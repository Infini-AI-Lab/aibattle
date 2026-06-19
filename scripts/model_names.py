"""Shared model-name normalization for the report pipeline.

Coached is now the canonical (and only) run set, so the ``-coached`` suffix that
the coached tournaments append to every model id is pure noise. The new-games
runs never carried it, so without stripping it the cross-game leaderboard would
treat ``kimi-k2p6-coached`` and ``kimi-k2p6`` as two different models. Every
analyzer normalizes its loaded run data through ``strip_coached`` so names line
up across all games.
"""

from __future__ import annotations

_SUFFIX = "-coached"

# Canonical, human-readable model names for display in reports/charts. The run
# data keeps the compact Fireworks slugs (e.g. ``glm-5p1`` where ``p`` is the
# decimal point); these are the official names verified against the Fireworks
# catalog. Display-only — internal lookups still key on the slug.
DISPLAY_NAMES = {
    "glm-5p1": "GLM-5.1",
    "glm-5p2": "GLM-5.2",
    "kimi-k2p6": "Kimi K2.6",
    "minimax-m2p7": "MiniMax-M2.7",
    "minimax-m3": "MiniMax-M3",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "qwen3p7-plus": "Qwen3.7 Plus",
    "gpt-oss-120b": "GPT-OSS 120B",
    "claude-opus-4.8": "Claude Opus 4.8",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
}


def display_name(model):
    """Official display name for a model slug; falls back to the slug itself
    (after stripping any ``-coached`` suffix) so unknown ids still render."""
    if not isinstance(model, str):
        return model
    slug = model[: -len(_SUFFIX)] if model.endswith(_SUFFIX) else model
    return DISPLAY_NAMES.get(slug, slug)


def strip_coached(obj):
    """Recursively strip a trailing ``-coached`` from every string in a loaded
    JSON structure. Only model-name ids end with that suffix in the run data
    (board cells are ``player_0``/``player_1``, etc.), so this is safe to apply
    to a whole parsed ``data.json`` / episode file."""
    if isinstance(obj, str):
        return obj[: -len(_SUFFIX)] if obj.endswith(_SUFFIX) else obj
    if isinstance(obj, list):
        return [strip_coached(x) for x in obj]
    if isinstance(obj, dict):
        return {strip_coached(k): strip_coached(v) for k, v in obj.items()}
    return obj
