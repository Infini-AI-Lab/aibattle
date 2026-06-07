"""OpenAI model client through AWS Bedrock.

The JavaScript OpenAI SDK exposes ``BedrockOpenAI`` and calls
``responses.create`` for model ids such as ``openai.gpt-5.5``. This repository's
runner is Python, so this adapter keeps the model boundary in Python while
delegating GPT calls to a small Node worker that uses that exact SDK path.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Union

from .base import ModelClient, ModelOutput


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
        self._backend = os.environ.get("BEDROCK_OPENAI_BACKEND", "node").lower()
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
        return await self._generate_node(
            prompt, temperature=temperature, max_tokens=max_tokens
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
