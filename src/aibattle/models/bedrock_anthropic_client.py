"""Anthropic Claude client through AWS Bedrock Runtime Converse."""

from __future__ import annotations

from typing import Optional, Union

from .base import ModelClient, ModelOutput
from .bedrock_converse import (
    converse_with_retries,
    extract_converse_output,
    make_bedrock_runtime,
    text_message,
)


class BedrockAnthropicClient(ModelClient):
    def __init__(
        self,
        *,
        model_id: str,
        aws_region: str,
        temperature: Optional[float] = None,
        max_tokens: int = 128000,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        timeout: float = 900.0,
    ):
        self.model_id = model_id
        self.aws_region = aws_region
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._reasoning_effort = reasoning_effort
        self._thinking_budget_tokens = thinking_budget_tokens
        self._client = make_bedrock_runtime(aws_region, timeout=timeout)

    async def generate(
        self,
        prompt: Union[str, list],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        out_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        inference_config = {"maxTokens": out_tokens}
        temp = self._default_temperature if temperature is None else temperature
        if temp is not None:
            inference_config["temperature"] = temp
        kwargs = {
            "modelId": self.model_id,
            "messages": text_message(prompt),
            "inferenceConfig": inference_config,
        }
        if self._system_prompt:
            kwargs["system"] = [{"text": self._system_prompt}]

        extra = {}
        if self._reasoning_effort:
            # Claude 4.6/4.8 on Bedrock use adaptive thinking with an effort
            # knob instead of the older explicit thinking-token budget shape.
            extra["thinking"] = {"type": "adaptive"}
            extra["output_config"] = {"effort": self._reasoning_effort}
        elif self._thinking_budget_tokens:
            extra["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(
                    self._thinking_budget_tokens,
                    max(1024, out_tokens - 1),
                ),
            }
        if extra:
            kwargs["additionalModelRequestFields"] = extra

        resp = await converse_with_retries(self._client, **kwargs)
        return extract_converse_output(resp)
