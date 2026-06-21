"""Add 100 MORE independent_blackjack hands for glm-5p1 vs the builtin dealer,
extending the existing 100-hand series to 200 total. Same dir + same per-call
settings (coached, temp 0.6, max_tokens 131072, 900s, 2 retries) so per-episode
resume skips the on-disk ep000-ep099 and fills ep100-ep199 — no contamination.

Runs as its own process in parallel with the blotto top-off. Modest concurrency
cap (default 16) so the two processes together stay near the 64-slot account
request ceiling rather than doubling it (128 -> 429 storm).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback

if "FIREWORKS_API_KEY" not in os.environ and os.path.exists(".fireworks"):
    with open(".fireworks", encoding="utf-8") as fh:
        os.environ["FIREWORKS_API_KEY"] = fh.read().strip()

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "16"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "131072"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "900"))
MAX_RETRIES = 2

MODEL = "glm-5p1"
TOTAL_EPISODES = int(os.environ.get("BLACKJACK_EPISODES", "200"))  # 100 existing + 100 new
OUT = f"runs/new_games_experiment/independent_blackjack/{MODEL}__vs__dealer"


def acfg(name, timeout):
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{name}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout_s": timeout},
            "max_retries": MAX_RETRIES}


def dealer_cfg():
    return {"type": "builtin", "name": "blackjack_dealer"}


def step_tracker(path, pair):
    def on_step(info):
        rec = {"t": round(time.time(), 1), "pair": pair, "ep": info["episode"],
               "step": info["step"], "agent": info["agent_name"],
               "action": (info.get("action") or "")[:40]}
        try:
            open(path, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return on_step


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    print(f"extra blackjack: {MODEL} vs dealer -> {TOTAL_EPISODES} total "
          f"(resume fills only the missing), cap {MAX_CONCURRENCY}\n", flush=True)
    runner = Runner(lambda: make_game("independent_blackjack"), on_invalid_action="fallback")
    steps_path = "runs/new_games_experiment/independent_blackjack/steps.jsonl"
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(MODEL, MODEL_TIMEOUT_S), game_name="independent_blackjack"),
                make_agent(dealer_cfg(), game_name="independent_blackjack"),
                episodes=TOTAL_EPISODES, seat_swap=False,
                logger=lg, semaphore=sem, episode_dir=OUT,
                on_step=step_tracker(steps_path, f"{MODEL}__vs__dealer"))
        drop = f"  DROPPED {res.failures}/{TOTAL_EPISODES}" if res.failures else ""
        print(f"blackjack {MODEL} vs dealer: {len(res.episodes)}/{TOTAL_EPISODES}{drop}", flush=True)
    except Exception as ex:
        print(f"blackjack {MODEL} FAILED: {ex}", flush=True)
        traceback.print_exc()
    print(f"\nEXTRA BLACKJACK DONE in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
