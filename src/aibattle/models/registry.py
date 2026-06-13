"""Model client factory.

Resolves a parsed ``model`` config block into a ``ModelClient``. API keys are
read from the environment variable named by ``api_key_env`` (never inlined in
YAML). A ``.fireworks`` file in the project root is used as a convenience
fallback for the Fireworks key.
"""

from __future__ import annotations

import os

from .base import ModelClient

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def _resolve_api_key(cfg: dict) -> str:
    env_name = cfg.get("api_key_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]

    provider = cfg.get("provider", "openai")
    # Convenience fallback: read the project-local .fireworks file.
    if provider in ("fireworks", "openai") and "fireworks" in (cfg.get("base_url") or _FIREWORKS_BASE_URL):
        for path in (".fireworks", os.path.expanduser("~/.fireworks")):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    key = fh.read().strip()
                if key:
                    return key
    raise ValueError(
        f"No API key found. Set env var {env_name!r} "
        "or provide a .fireworks file for Fireworks."
    )


def make_client(cfg: dict) -> ModelClient:
    provider = cfg.get("provider", "openai")

    if provider in ("openai", "fireworks"):
        from .openai_client import OpenAIClient

        base_url = cfg.get("base_url")
        if provider == "fireworks" and not base_url:
            base_url = _FIREWORKS_BASE_URL
        return OpenAIClient(
            model_id=cfg["model_id"],
            api_key=_resolve_api_key({**cfg, "base_url": base_url}),
            base_url=base_url,
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 256)),
            system_prompt=cfg.get("system_prompt"),
            timeout=float(cfg.get("timeout_s", 300)),
        )

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            model_id=cfg["model_id"],
            api_key=_resolve_api_key(cfg),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 256)),
            system_prompt=cfg.get("system_prompt"),
        )

    if provider == "bedrock_openai":
        from .bedrock_openai_client import BedrockOpenAIClient

        return BedrockOpenAIClient(
            model_id=cfg["model_id"],
            aws_region=cfg.get("aws_region", cfg.get("region", "us-east-2")),
            temperature=(
                float(cfg["temperature"])
                if cfg.get("temperature") is not None
                else None
            ),
            max_tokens=int(cfg.get("max_tokens", 128000)),
            system_prompt=cfg.get("system_prompt"),
            reasoning_effort=cfg.get("reasoning_effort"),
            timeout=float(cfg.get("timeout_s", 900)),
        )

    if provider == "bedrock_anthropic":
        from .bedrock_anthropic_client import BedrockAnthropicClient

        return BedrockAnthropicClient(
            model_id=cfg["model_id"],
            aws_region=cfg.get("aws_region", cfg.get("region", "us-east-1")),
            temperature=(
                float(cfg["temperature"])
                if cfg.get("temperature") is not None
                else None
            ),
            max_tokens=int(cfg.get("max_tokens", 128000)),
            system_prompt=cfg.get("system_prompt"),
            reasoning_effort=cfg.get("reasoning_effort"),
            thinking_budget_tokens=(
                int(cfg["thinking_budget_tokens"])
                if cfg.get("thinking_budget_tokens") is not None
                else None
            ),
            timeout=float(cfg.get("timeout_s", 900)),
        )

    raise ValueError(f"Unknown model provider {provider!r}")
