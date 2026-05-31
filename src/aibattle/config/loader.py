"""YAML config loading + validation.

The YAML file is the primary v0 interface. This loader parses it, validates the
shape, and fails fast with clear messages on unknown games/agents or missing
keys. API keys are NOT read here — they are resolved from the environment (or
the .fireworks fallback) at model-client construction time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

from ..games.registry import available_games


@dataclass
class GameConfig:
    name: str
    version: Optional[str] = None
    params: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    episodes: int = 100
    seed: int = 0
    seat_swap: bool = True
    on_invalid_action: str = "fallback"
    max_concurrency: int = 32


@dataclass
class OutputConfig:
    dir: str = "./runs/exp"
    save_full_log: bool = True
    save_summary: bool = True
    save_trajectories: bool = False     # one combined JSON of all episode trajectories
    trajectories_file: str = "trajectories.json"
    save_transcripts: bool = False      # one plain-text file per episode (human-readable)
    transcripts_dir: str = "transcripts"


@dataclass
class ArenaConfig:
    game: GameConfig
    players: dict          # {"player_0": <agent cfg dict>, "player_1": ...}
    run: RunConfig
    output: OutputConfig
    raw: dict = field(default_factory=dict)


_VALID_PLAYERS = ["player_0", "player_1"]
_VALID_INVALID_POLICIES = {"fallback", "forfeit"}


def load_config(path: str) -> ArenaConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # --- game ---
    g = data.get("game") or {}
    if "name" not in g:
        raise ValueError("config.game.name is required")
    if g["name"] not in available_games():
        raise ValueError(
            f"Unknown game {g['name']!r}. Available: {available_games()}"
        )
    game = GameConfig(name=g["name"], version=g.get("version"),
                      params=g.get("params") or {})

    # --- players ---
    players = data.get("players") or {}
    for pid in _VALID_PLAYERS:
        if pid not in players:
            raise ValueError(f"config.players.{pid} is required")
        agent_cfg = (players[pid] or {}).get("agent")
        if not agent_cfg or "type" not in agent_cfg:
            raise ValueError(f"config.players.{pid}.agent.type is required")
        if agent_cfg["type"] not in ("builtin", "model", "external", "human"):
            raise ValueError(
                f"config.players.{pid}.agent.type must be "
                "builtin|model|external|human"
            )

    # --- run ---
    r = data.get("run") or {}
    policy = r.get("on_invalid_action", "fallback")
    if policy not in _VALID_INVALID_POLICIES:
        raise ValueError(
            f"run.on_invalid_action must be one of {_VALID_INVALID_POLICIES}"
        )
    # Accept legacy `concurrency` as a fallback for `max_concurrency`.
    max_concurrency = int(r.get("max_concurrency", r.get("concurrency", 32)))
    run = RunConfig(
        episodes=int(r.get("episodes", 100)),
        seed=int(r.get("seed", 0)),
        seat_swap=bool(r.get("seat_swap", True)),
        on_invalid_action=policy,
        max_concurrency=max_concurrency,
    )

    # --- output ---
    o = data.get("output") or {}
    output = OutputConfig(
        dir=o.get("dir", "./runs/exp"),
        save_full_log=bool(o.get("save_full_log", True)),
        save_summary=bool(o.get("save_summary", True)),
        save_trajectories=bool(o.get("save_trajectories", False)),
        trajectories_file=o.get("trajectories_file", "trajectories.json"),
        save_transcripts=bool(o.get("save_transcripts", False)),
        transcripts_dir=o.get("transcripts_dir", "transcripts"),
    )

    return ArenaConfig(
        game=game,
        players={pid: players[pid]["agent"] for pid in _VALID_PLAYERS},
        run=run,
        output=output,
        raw=data,
    )
