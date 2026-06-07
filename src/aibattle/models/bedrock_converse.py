"""Shared helpers for AWS Bedrock Runtime Converse clients."""

from __future__ import annotations

import asyncio
import random
from typing import Optional

from botocore.config import Config

from .base import ModelOutput


def make_bedrock_runtime(region_name: str, *, timeout: float):
    """Create a Bedrock Runtime client with per-call timeouts."""
    import boto3

    cfg = Config(
        read_timeout=timeout,
        connect_timeout=min(timeout, 30.0),
        retries={"max_attempts": 1},
    )
    return boto3.client("bedrock-runtime", region_name=region_name, config=cfg)


def text_message(prompt, *, system_prompt: Optional[str] = None):
    """Normalize a string or chat-style list into Bedrock Converse messages."""
    if isinstance(prompt, str):
        text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return [{"role": "user", "content": [{"text": text}]}]
    return prompt


def extract_converse_output(resp: dict) -> ModelOutput:
    """Pull text, reasoning text, stop reason, and token counts from Converse."""
    content_blocks = (
        resp.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    texts: list[str] = []
    reasoning_parts: list[str] = []
    for block in content_blocks:
        if "text" in block:
            texts.append(block["text"])
        reasoning = block.get("reasoningContent")
        if isinstance(reasoning, dict):
            reasoning_text = reasoning.get("reasoningText")
            if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                reasoning_parts.append(reasoning_text["text"])
            elif isinstance(reasoning.get("text"), str):
                reasoning_parts.append(reasoning["text"])
        thinking = block.get("thinking")
        if isinstance(thinking, str):
            reasoning_parts.append(thinking)

    usage = resp.get("usage") or {}
    stop = resp.get("stopReason")
    return ModelOutput(
        content="".join(texts),
        reasoning=("".join(reasoning_parts) or None),
        finish_reason="length" if stop == "max_tokens" else stop,
        completion_tokens=usage.get("outputTokens"),
        prompt_tokens=usage.get("inputTokens"),
    )


async def converse_with_retries(client, *, attempts: int = 8, **kwargs) -> dict:
    """Run a blocking boto3 Converse call in a worker thread with backoff."""
    for attempt in range(attempts):
        try:
            return await asyncio.to_thread(client.converse, **kwargs)
        except Exception:  # noqa: BLE001 - boto3 exposes several transient types
            if attempt == attempts - 1:
                raise
            base = min(60, 2 ** attempt)
            await asyncio.sleep(base * (0.5 + random.random()))
    raise RuntimeError("unreachable")
