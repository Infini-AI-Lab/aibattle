"""Agent factory registry.

Builds an ``Agent`` from a parsed config block. Kinds:
  - builtin : random / kuhn_heuristic
  - model   : ModelClient + game template (default model-backed agent)
  - local   : in-process reasoning harness (CoT / vote / two-stage / self-refine)
  - external: in-process import path or HTTP endpoint
  - human   : interactive terminal player
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import Agent
from .random_agent import RandomAgent
from .heuristic_agent import KuhnHeuristicAgent
from .holdem_agents import RandomHoldemAgent, HoldemHeuristicAgent
from .board_agents import (
    RandomBoardAgent, Connect4HeuristicAgent, GomokuHeuristicAgent,
)

_BUILTINS = {
    "random": RandomAgent,
    "kuhn_heuristic": KuhnHeuristicAgent,
    "holdem_random": RandomHoldemAgent,
    "holdem_heuristic": HoldemHeuristicAgent,
    "board_random": RandomBoardAgent,
    "connect4_heuristic": Connect4HeuristicAgent,
    "gomoku_heuristic": GomokuHeuristicAgent,
}


def _build_model_agent(cfg: dict, game_name: str) -> Agent:
    from ..models.registry import make_client
    from .model_agent import ModelAgent
    from .templates.registry import make_template

    model_cfg = cfg.get("model") or {}
    client = make_client(model_cfg)
    template = make_template(game_name, coached=bool(cfg.get("coached", False)))
    return ModelAgent(
        client=client,
        template=template,
        name=cfg.get("name", model_cfg.get("model_id", "model")),
        max_retries=int(cfg.get("max_retries", 2)),
    )


def _build_local_agent(cfg: dict, game_name: str) -> Agent:
    """Build an in-process reasoning-harness agent (``type: local``).

    Reuses the same ModelClient + GameTemplate as ModelAgent, but wraps them in a
    multi-step harness selected by ``cfg["harness"]``. Harness-specific knobs
    (n, temperature, rounds, custom prompts) come from ``cfg["harness_args"]``.
    """
    from ..models.registry import make_client
    from .local import make_harness
    from .templates.registry import make_template

    harness = cfg.get("harness")
    if not harness:
        raise ValueError("local agent needs a 'harness' (e.g. cot, self_consistency, "
                         "two_stage, self_refine)")
    model_cfg = cfg.get("model") or {}
    client = make_client(model_cfg)
    template = make_template(game_name)
    return make_harness(
        harness,
        client=client,
        template=template,
        name=cfg.get("name", model_cfg.get("model_id", harness)),
        max_retries=int(cfg.get("max_retries", 2)),
        harness_args=cfg.get("harness_args") or {},
    )


def _build_external_agent(cfg: dict) -> Agent:
    if "entrypoint" in cfg:
        module_path, _, cls_name = cfg["entrypoint"].partition(":")
        if not cls_name:
            raise ValueError(
                f"external entrypoint must be 'module:ClassName', got {cfg['entrypoint']!r}"
            )
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
        return cls(**cfg.get("args", {}))
    if "http" in cfg:
        from .http_agent import HttpAgent
        http = cfg["http"]
        return HttpAgent(
            name=cfg.get("name", "external_http"),
            url=http["url"],
            timeout_s=float(http.get("timeout_s", 30)),
        )
    raise ValueError("external agent needs 'entrypoint' or 'http'")


def make_agent(cfg: dict, *, game_name: str, seed: int | None = None) -> Agent:
    atype = cfg.get("type")
    if atype == "human":
        from .human_agent import HumanAgent
        return HumanAgent(name=cfg.get("name", "human"))
    if atype == "builtin":
        name = cfg.get("name")
        if name not in _BUILTINS:
            raise ValueError(
                f"Unknown builtin agent {name!r}. Available: {sorted(_BUILTINS)}"
            )
        kwargs: dict[str, Any] = {"seed": seed}
        return _BUILTINS[name](name=name, **kwargs)
    if atype == "model":
        return _build_model_agent(cfg, game_name)
    if atype == "local":
        return _build_local_agent(cfg, game_name)
    if atype == "external":
        return _build_external_agent(cfg)
    raise ValueError(
        f"Unknown agent type {atype!r} (expected builtin|model|local|external|human)"
    )
