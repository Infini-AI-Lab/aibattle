"""Extend the existing leduc_poker pair coverage in place: 50 -> 100 episodes/pair
for the 7-model set.

Each pair's (a, b) order and its output dir are taken from the EXISTING on-disk
dir name (bare ``A__vs__B``), so per-episode resume skips ep000-ep049 already
there and fills only ep050-ep099 — no duplicate/reversed dirs, no recompute.

Methodology matches the original new_games_experiment leduc run exactly (verified
against saved episodes): coached agents with BARE names (name=model id, not
``model-coached``), leduc_poker params {}, temp 0.6, max_tokens 131072, 900s
timeout, 2 retries, seat_swap=True. Leduc episodes are short (2-6 decisions), so
this is fast and not prone to the long-sequential hang seen in blotto/gomoku.

DRY_RUN=1 prints the target pairs + current counts and exits without running.
"""
from __future__ import annotations

import asyncio
import glob
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
TARGET = int(os.environ.get("LEDUC_TARGET", "100"))
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

BASE = "runs/new_games_experiment/leduc_poker"
STEPS = os.path.join(BASE, "steps.jsonl")
SEVEN = {"deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "glm-5p1", "gpt-oss-120b",
         "qwen3p7-plus", "glm-5p2"}

sem = None


def acfg(name, timeout):
    # BARE name (matches original leduc run), coached=True (verified in saved eps).
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{name}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout_s": timeout},
            "max_retries": MAX_RETRIES}


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


def parse_pair(dirname):
    """'A__vs__B' -> (a, b) if both in 7-set, else None."""
    if "__vs__" not in dirname:
        return None
    a, b = dirname.split("__vs__", 1)
    if a in SEVEN and b in SEVEN and a != b:
        return a, b
    return None


async def play(gdir, a, b, done_box, total):
    runner = Runner(lambda: make_game("leduc_poker", {}), on_invalid_action="fallback")
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(a, MODEL_TIMEOUT_S), game_name="leduc_poker"),
                make_agent(acfg(b, MODEL_TIMEOUT_S), game_name="leduc_poker"),
                episodes=TARGET, seed=None, seat_swap=True,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(f"{a}__vs__{b}"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{TARGET}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] leduc {a} vs {b}: {len(res.episodes)}/{TARGET}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] leduc {a} vs {b} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    pairs = []
    for d in sorted(glob.glob(BASE + "/*")):
        if not os.path.isdir(d):
            continue
        pr = parse_pair(os.path.basename(d))
        if not pr:
            continue
        n = len(glob.glob(os.path.join(d, "ep*.json")))
        if n == 0:
            print(f"  skip empty dir {os.path.basename(d)}", flush=True)
            continue
        pairs.append((d, pr, n))
    total = len(pairs)
    print(f"extend leduc: {total} 7-set pairs -> {TARGET}/pair, cap {MAX_CONCURRENCY}, resume on", flush=True)
    for d, (a, b), n in pairs:
        print(f"  {a} vs {b}: {n}/{TARGET}", flush=True)
    if DRY_RUN:
        print("DRY_RUN — not launching.", flush=True)
        return
    if not total:
        print("nothing to do.", flush=True)
        return
    t0 = time.perf_counter()
    done = [0]
    await asyncio.gather(*(play(d, a, b, done, total) for d, (a, b), n in pairs))
    print(f"\nEXTEND LEDUC DONE ({total} pairs) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
