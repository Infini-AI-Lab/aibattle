"""Bring EVERY 7-set model's independent_blackjack series up to 500 hands vs the
builtin dealer. Same dirs + same per-call settings (coached, temp 0.6, max_tokens
131072, 900s, 2 retries) so per-episode resume skips on-disk ep files and fills
only the missing ones up to ep499 — no contamination, fully restart-safe.

One shared semaphore (modest cap, default 16) across all models so this short
side run plus the blotto top-off (cap 64) stay near the 64-slot account request
ceiling rather than blowing past it (128 -> 429 storm). Blackjack hands are short
(~4 decisions) so it still finishes quickly.
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
TOTAL_EPISODES = int(os.environ.get("BLACKJACK_EPISODES", "500"))

_DEFAULT_MODELS = ["deepseek-v4-pro", "glm-5p1", "glm-5p2", "gpt-oss-120b",
                   "kimi-k2p6", "minimax-m3", "qwen3p7-plus"]
MODELS = ([m.strip() for m in os.environ["BLACKJACK_MODELS"].split(",") if m.strip()]
          if os.environ.get("BLACKJACK_MODELS") else _DEFAULT_MODELS)
BASE = "runs/new_games_experiment/independent_blackjack"
STEPS = os.path.join(BASE, "steps.jsonl")

sem = None


def acfg(name, timeout):
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{name}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout_s": timeout},
            "max_retries": MAX_RETRIES}


def dealer_cfg():
    return {"type": "builtin", "name": "blackjack_dealer"}


def step_tracker(pair):
    def on_step(info):
        rec = {"t": round(time.time(), 1), "pair": pair, "ep": info["episode"],
               "step": info["step"], "agent": info["agent_name"],
               "action": (info.get("action") or "")[:40]}
        try:
            open(STEPS, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return on_step


async def play(model, done_box, total):
    gdir = os.path.join(BASE, f"{model}__vs__dealer")
    os.makedirs(gdir, exist_ok=True)
    runner = Runner(lambda: make_game("independent_blackjack"), on_invalid_action="fallback")
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(model, MODEL_TIMEOUT_S), game_name="independent_blackjack"),
                make_agent(dealer_cfg(), game_name="independent_blackjack"),
                episodes=TOTAL_EPISODES, seat_swap=False,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(f"{model}__vs__dealer"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{TOTAL_EPISODES}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] blackjack {model} vs dealer: {len(res.episodes)}/{TOTAL_EPISODES}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] blackjack {model} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    os.makedirs(BASE, exist_ok=True)
    t0 = time.perf_counter()
    done = [0]
    total = len(MODELS)
    print(f"extra blackjack ALL: {MODELS} -> {TOTAL_EPISODES} each (resume fills missing), "
          f"shared cap {MAX_CONCURRENCY}\n", flush=True)
    await asyncio.gather(*(play(m, done, total) for m in MODELS))
    print(f"\nEXTRA BLACKJACK ALL DONE in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
