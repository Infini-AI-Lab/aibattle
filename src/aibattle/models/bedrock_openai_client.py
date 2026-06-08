"""OpenAI model client through AWS Bedrock Mantle Responses."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import random
import threading
from pathlib import Path
from typing import Optional, Union

from .base import ModelClient, ModelOutput


_SIGV4_CREDENTIALS_LOCK = threading.Lock()
_SIGV4_CREDENTIALS_CACHE = {}
_SIGV4_REFRESH_MARGIN = timedelta(
    minutes=int(os.environ.get("BEDROCK_CREDENTIAL_REFRESH_MARGIN_MINUTES", "10"))
)


def _cached_sigv4_credentials(profile: Optional[str], region: str):
    """Return frozen AWS credentials without invoking credential_process per call."""
    import boto3

    key = (profile or "", region)
    now = datetime.now(timezone.utc)
    with _SIGV4_CREDENTIALS_LOCK:
        cached = _SIGV4_CREDENTIALS_CACHE.get(key)
        if cached is not None:
            frozen, expiry = cached
            if expiry is None or expiry - now > _SIGV4_REFRESH_MARGIN:
                return frozen

        session_kwargs = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)
        creds = session.get_credentials()
        if creds is None:
            raise RuntimeError("No AWS credentials available for Bedrock SigV4 call")
        frozen = creds.get_frozen_credentials()
        expiry = getattr(creds, "_expiry_time", None)
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        _SIGV4_CREDENTIALS_CACHE[key] = (frozen, expiry)
        return frozen


class BedrockOpenAIClient(ModelClient):
    def __init__(
        self,
        *,
        model_id: str,
        aws_region: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout: float = 900.0,
    ):
        self.model_id = model_id
        self.aws_region = aws_region
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        self._backend = os.environ.get("BEDROCK_OPENAI_BACKEND", "sigv4").lower()
        self._aws_profile = os.environ.get("AWS_PROFILE")
        self._client = None
        self._proc = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        if self._backend == "converse":
            from .bedrock_converse import make_bedrock_runtime

            self._client = make_bedrock_runtime(aws_region, timeout=timeout)

    async def generate(
        self,
        prompt: Union[str, list],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        if self._backend == "converse":
            return await self._generate_converse(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        if self._backend == "node":
            return await self._generate_node(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        return await self._generate_sigv4(
            prompt, temperature=temperature, max_tokens=max_tokens
        )

    def _prompt_text(self, prompt: Union[str, list]) -> str:
        if isinstance(prompt, list):
            parts = []
            for msg in prompt:
                role = msg.get("role", "user")
                parts.append(f"{role}: {msg.get('content', '')}")
            input_text = "\n\n".join(parts)
        else:
            input_text = prompt
        if self._system_prompt:
            input_text = f"{self._system_prompt}\n\n{input_text}"
        return input_text

    async def _generate_sigv4(
        self,
        prompt: Union[str, list],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        effort = "xhigh" if self._reasoning_effort == "max" else (
            self._reasoning_effort or "xhigh"
        )
        payload = await self._post_sigv4_with_retries(
            {
                "model": self.model_id,
                "input": self._prompt_text(prompt),
                "reasoning": {"effort": effort},
                "max_output_tokens": (
                    self._default_max_tokens if max_tokens is None else max_tokens
                ),
                "store": False,
                **(
                    {"temperature": temp}
                    if (temp := (
                        self._default_temperature
                        if temperature is None
                        else temperature
                    )) is not None
                    else {}
                ),
            }
        )
        return self._response_payload_to_model_output(payload)

    async def _post_sigv4_with_retries(self, body: dict, *, attempts: int = 8) -> dict:
        for attempt in range(attempts):
            try:
                return await asyncio.to_thread(self._post_sigv4, body)
            except Exception:  # noqa: BLE001
                if attempt == attempts - 1:
                    raise
                base = min(60, 2 ** attempt)
                await asyncio.sleep(base * (0.5 + random.random()))
        raise RuntimeError("unreachable")

    def _post_sigv4(self, body: dict) -> dict:
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        frozen = _cached_sigv4_credentials(self._aws_profile, self.aws_region)
        data = json.dumps(body)
        endpoint = (
            f"https://bedrock-mantle.{self.aws_region}.api.aws/openai/v1/responses"
        )
        req = AWSRequest(
            method="POST",
            url=endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(frozen, "bedrock", self.aws_region).add_auth(req)
        resp = requests.post(
            endpoint, data=data, headers=dict(req.headers), timeout=self._timeout
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"{resp.status_code} {resp.text}") from e
        return resp.json()

    @staticmethod
    def _response_payload_to_model_output(payload: dict) -> ModelOutput:
        content = payload.get("output_text")
        if not isinstance(content, str):
            chunks = []
            for item in payload.get("output") or []:
                for part in item.get("content") or []:
                    if isinstance(part.get("text"), str):
                        chunks.append(part["text"])
                    elif isinstance(part.get("refusal"), str):
                        chunks.append(part["refusal"])
            content = "".join(chunks)

        reasoning_parts = []
        for item in payload.get("output") or []:
            if item.get("type") == "reasoning":
                for part in item.get("summary") or []:
                    if isinstance(part.get("text"), str):
                        reasoning_parts.append(part["text"])
        usage = payload.get("usage") or {}
        return ModelOutput(
            content=content or "",
            reasoning=("".join(reasoning_parts) or None),
            finish_reason=payload.get("status"),
            completion_tokens=usage.get("output_tokens"),
            prompt_tokens=usage.get("input_tokens"),
        )

    async def _generate_converse(
        self,
        prompt: Union[str, list],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        from .bedrock_converse import (
            converse_with_retries,
            extract_converse_output,
            text_message,
        )

        inference_config = {
            "maxTokens": (
                self._default_max_tokens if max_tokens is None else max_tokens
            )
        }
        temp = self._default_temperature if temperature is None else temperature
        if temp is not None:
            inference_config["temperature"] = temp

        kwargs = {
            "modelId": self.model_id,
            "messages": text_message(prompt, system_prompt=self._system_prompt),
            "inferenceConfig": inference_config,
        }
        if self._reasoning_effort:
            kwargs["additionalModelRequestFields"] = {
                "reasoning": {"effort": self._reasoning_effort}
            }
        resp = await converse_with_retries(self._client, **kwargs)
        return extract_converse_output(resp)

    async def _start_worker(self):
        if self._proc is not None and self._proc.returncode is None:
            return
        repo_root = Path(__file__).resolve().parents[3]
        worker = repo_root / "scripts" / "bedrock_openai_worker.mjs"
        self._proc = await asyncio.create_subprocess_exec(
            "node",
            str(worker),
            cwd=str(repo_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _generate_node(
        self,
        prompt: Union[str, list],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        input_text = self._prompt_text(prompt)

        async with self._lock:
            await self._start_worker()
            self._next_id += 1
            req = {
                "id": self._next_id,
                "awsRegion": self.aws_region,
                "model": self.model_id,
                "input": input_text,
                "maxOutputTokens": (
                    self._default_max_tokens if max_tokens is None else max_tokens
                ),
                "reasoningEffort": self._reasoning_effort,
            }
            temp = self._default_temperature if temperature is None else temperature
            if temp is not None:
                req["temperature"] = temp
            line = json.dumps(req, ensure_ascii=False) + "\n"
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=self._timeout + 30
                )
            except asyncio.TimeoutError as e:
                raise TimeoutError(
                    f"Timed out waiting for BedrockOpenAI response for {self.model_id}"
                ) from e
            if not raw:
                stderr = ""
                if self._proc.stderr is not None:
                    stderr = (await self._proc.stderr.read()).decode("utf-8", "replace")
                raise RuntimeError(
                    f"BedrockOpenAI worker exited before responding. {stderr}".strip()
                )
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            usage = payload.get("usage") or {}
            return ModelOutput(
                content=payload.get("content") or "",
                reasoning=payload.get("reasoning"),
                finish_reason=payload.get("finish_reason"),
                completion_tokens=usage.get("output_tokens"),
                prompt_tokens=usage.get("input_tokens"),
            )
