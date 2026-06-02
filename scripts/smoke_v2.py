"""Tiny end-to-end smoke test of the v2 Hold'em config (temp 0.6, 100bb stacks,
per-hand betting history in the prompt). Runs ONE short game per mode into a
throwaway dir (runs/_smoke_v2) so the real run folders stay clean. Verifies:
prompts render (with history), models respond, parsing works, episodes persist.
Short hand counts keep it fast; the real runs use the full MAX_HANDS.
"""
from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

FIVE = ["deepseek-v4-pro", "gpt-oss-120b", "kimi-k2p6", "glm-5p1", "minimax-m2p7"]
TWO = ["deepseek-v4-pro", "gpt-oss-120b"]
OUT = "runs/_smoke_v2"


def acfg(name):  # mirrors the real scripts' v2 config
    return {"type": "model", "name": name,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{name}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": 0.6, "max_tokens": 131072, "timeout_s": 900},
            "max_retries": 2}


def sample(res, label):
    eps = res.episodes
    print(f"\n=== {label}: episodes={len(eps)} drops={res.failures} ===", flush=True)
    if not eps:
        print("  (no episode completed)"); return
    e = eps[0]
    trunc = sum(1 for s in e.get("steps", [])
                if ((s.get("response") or {}).get("metadata") or {}).get("truncated"))
    inv = e.get("invalid_count")
    print(f"  winner={e.get('winner_name') or e.get('ranking')} "
          f"length={e.get('length')} decisions invalid={inv} truncated_steps={trunc}")
    for s in e.get("steps", [])[:4]:
        m = (s.get("response") or {}).get("metadata") or {}
        print(f"    step{s.get('step_index')} {s.get('agent_name')} -> "
              f"{s.get('selected_action')} {s.get('selected_amount') or ''} "
              f"(toks={m.get('completion_tokens')}, finish={m.get('finish_reason')})")


async def lite():
    r = Runner(lambda: make_game("holdem", {"starting_stack": 200}),
               on_invalid_action="fallback")
    with MatchLogger(None) as lg:
        res = await r.run_match(make_agent(acfg(TWO[0]), game_name="holdem"),
                                make_agent(acfg(TWO[1]), game_name="holdem"),
                                episodes=2, seed=1000, seat_swap=True, logger=lg,
                                episode_dir=os.path.join(OUT, "lite"))
    sample(res, "LITE (2 hands)")


async def match():
    r = Runner(lambda: make_game("holdem_match", {"starting_stack": 200, "max_hands": 4}),
               on_invalid_action="fallback")
    with MatchLogger(None) as lg:
        res = await r.run_match(make_agent(acfg(TWO[0]), game_name="holdem_match"),
                                make_agent(acfg(TWO[1]), game_name="holdem_match"),
                                episodes=1, seed=7000, seat_swap=False, logger=lg,
                                episode_dir=os.path.join(OUT, "match"))
    sample(res, "MATCH (1 match x 4 hands)")


async def table():
    g = lambda: make_game("holdem_table", {"num_players": 5, "starting_stack": 200,
                                           "max_hands": 4})
    r = Runner(g, on_invalid_action="fallback")
    agents = [make_agent(acfg(m), game_name="holdem_table") for m in FIVE]
    with MatchLogger(None) as lg:
        res = await r.run_table(agents, episodes=1, seed=8000, logger=lg,
                                episode_dir=os.path.join(OUT, "table"), seat_rotate=True)
    sample(res, "TABLE (1 session x 4 hands, 5 players)")


async def main():
    os.makedirs(OUT, exist_ok=True)
    print("v2 smoke: temp=0.6, 100bb stacks, history prompt, max_tokens=131072\n", flush=True)
    await asyncio.gather(lite(), match(), table())
    print("\nSMOKE DONE.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
