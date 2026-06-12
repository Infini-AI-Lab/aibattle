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
