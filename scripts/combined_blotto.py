"""Add blotto (repeated_colonel_blotto) coverage for the two newest models —
qwen3p7-plus and glm-5p2 — so blotto becomes a full 7-model field too.

Only the NEW pairs run; every incumbent-vs-incumbent blotto pair is already on
disk and skipped by per-episode resume. Bare dir labels (coached agents), 20
episodes/pair, seat_swap, temp 0.6 — matching the original blotto conventions so
the leaderboard aggregator finds the ep files.

New pairs (11): each new model vs the 5 in-set incumbents + qwen-vs-glm-5p2.
Dir: runs/new_games_experiment/repeated_colonel_blotto/<opp>__vs__<new>.
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

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "64"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "131072"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "900"))
MAX_RETRIES = 2
BLOTTO_EPISODES = int(os.environ.get("BLOTTO_EPISODES", "20"))

INCUMBENTS = ["deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "glm-5p1", "gpt-oss-120b"]
NEW = ["qwen3p7-plus", "glm-5p2"]
OUT = "runs/new_games_experiment/repeated_colonel_blotto"

sem = None


def acfg(name, timeout):
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{name}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout_s": timeout},
            "max_retries": MAX_RETRIES}


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


async def play_blotto(a, b, seed, done_box, total):
    gdir = os.path.join(OUT, f"{a}__vs__{b}")
    os.makedirs(gdir, exist_ok=True)
    runner = Runner(lambda: make_game("repeated_colonel_blotto"), on_invalid_action="fallback")
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(a, MODEL_TIMEOUT_S), game_name="repeated_colonel_blotto"),
                make_agent(acfg(b, MODEL_TIMEOUT_S), game_name="repeated_colonel_blotto"),
                episodes=BLOTTO_EPISODES, seed=seed, seat_swap=True,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(os.path.join(OUT, "steps.jsonl"), f"{a}__vs__{b}"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{BLOTTO_EPISODES}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] blotto {a} vs {b}: {len(res.episodes)}/{BLOTTO_EPISODES}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] blotto {a} vs {b} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = [0]
    # Build new pairs: each new model vs the 5 incumbents + qwen-vs-glm-5p2.
    pairs = []
    for n in NEW:
        for o in INCUMBENTS:
            pairs.append((o, n))              # <opp>__vs__<new>
    pairs.append(("qwen3p7-plus", "glm-5p2"))  # the two new models head-to-head
    total = len(pairs)
    print(f"blotto eval: {NEW} vs field — {total} new pairs ({BLOTTO_EPISODES}/pair), "
          f"shared cap {MAX_CONCURRENCY}, resume on\n", flush=True)

    tasks = [play_blotto(a, b, 9300 + i, done, total) for i, (a, b) in enumerate(pairs)]
    await asyncio.gather(*tasks)
    print(f"\nBLOTTO EVAL DONE ({total} pairs) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
