"""Live smoke test of the four reasoning harnesses against a real model.

Runs ONE short Kuhn hand per harness via Fireworks (OpenAI-compatible), printing
the chosen action and the harness's intermediate artifacts (metadata["harness"]).
Requires FIREWORKS_API_KEY in the environment (or a .fireworks file).

Usage:
  FIREWORKS_API_KEY=... python scripts/smoke_harness.py
"""
from __future__ import annotations

import asyncio
import os

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

MODEL_ID = os.environ.get("SMOKE_MODEL", "accounts/fireworks/models/gpt-oss-120b")
HARNESSES = ["cot", "self_consistency", "two_stage", "self_refine"]


def acfg(harness):
    args = {"self_consistency": {"n": 3, "temperature": 0.7}}.get(harness, {})
    return {"type": "local", "harness": harness, "name": f"{harness}-bot",
            "model": {"provider": "fireworks", "model_id": MODEL_ID,
                      "api_key_env": "FIREWORKS_API_KEY",
                      # generous budget: reasoning models need room for the
                      # chain-of-thought before the final action word.
                      "temperature": 0.3, "max_tokens": 8192, "timeout_s": 300},
            "harness_args": args, "max_retries": 1}


async def one(harness):
    runner = Runner(lambda: make_game("kuhn_poker"), on_invalid_action="fallback")
    agent_a = make_agent(acfg(harness), game_name="kuhn_poker")
    agent_b = make_agent({"type": "builtin", "name": "kuhn_heuristic"},
                         game_name="kuhn_poker", seed=1)
    with MatchLogger(None) as lg:
        res = await runner.run_match(agent_a, agent_b, episodes=1, seed=2025,
                                     seat_swap=False, logger=lg)
    ep = res.episodes[0] if res.episodes else None
    print(f"\n=== {harness}: failures={res.failures} ===", flush=True)
    if not ep:
        print("  (no episode)"); return
    for s in ep.get("steps", []):
        resp = s.get("response") or {}
        meta = resp.get("metadata") or {}
        h = meta.get("harness") or {}
        if s.get("agent_name", "").endswith("-bot"):
            print(f"  step{s.get('step')} -> {s.get('selected_action')} "
                  f"{s.get('selected_amount') or ''}  harness={ {k: h[k] for k in h if k != 'history'} }")


async def main():
    if not os.environ.get("FIREWORKS_API_KEY") and os.path.exists(".fireworks"):
        os.environ["FIREWORKS_API_KEY"] = open(".fireworks").read().strip()
    for h in HARNESSES:
        try:
            await one(h)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {h}: ERROR {type(e).__name__}: {e} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
