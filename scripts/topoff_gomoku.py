"""Top off the gomoku tail for the new models (qwen3p7-plus, glm-5p2): re-run any
7-set new-model gomoku pair that is under 10 episodes, filling only the missing
ones via per-episode resume. Methodology matches the original gomoku runs:
coached, random_open=2, seat_swap=True, temp 0.6, max_tokens 131072, 900s, 2
retries. Pair (a,b) order + dir taken from the existing on-disk dir name.
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

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "131072"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
BOARD_TIMEOUT = float(os.environ.get("BOARD_TIMEOUT", "900"))
MAX_RETRIES = 2
GOMOKU_EPISODES = 10
RANDOM_OPEN = 2

SEVEN = {"glm-5p2", "qwen3p7-plus", "minimax-m3", "deepseek-v4-pro",
         "kimi-k2p6", "glm-5p1", "gpt-oss-120b"}
NEW = {"glm-5p2", "qwen3p7-plus"}

sem = None


def acfg(name, base, timeout):
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{base}",
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


def strip(n):
    return n[: -len("-coached")] if n.endswith("-coached") else n


async def play(gdir, a, b, seed, done_box, total):
    runner = Runner(lambda: make_game("gomoku", {"random_open": RANDOM_OPEN}),
                    on_invalid_action="fallback")
    na, nb = f"{a}-coached", f"{b}-coached"
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(na, a, BOARD_TIMEOUT), game_name="gomoku"),
                make_agent(acfg(nb, b, BOARD_TIMEOUT), game_name="gomoku"),
                episodes=GOMOKU_EPISODES, seed=seed, seat_swap=True,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker("runs/gomoku/steps.jsonl", f"{na}__vs__{nb}"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{GOMOKU_EPISODES}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] gomoku {a} vs {b}: {len(res.episodes)}/{GOMOKU_EPISODES}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] gomoku {a} vs {b} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    done = [0]
    short = []
    for d in sorted(glob.glob("runs/gomoku/gomoku__*")):
        if not os.path.isdir(d):
            continue
        nm = os.path.basename(d)[len("gomoku__"):]
        if "__vs__" not in nm:
            continue
        a, b = [strip(x) for x in nm.split("__vs__")]
        if not ({a, b} <= SEVEN) or not ({a, b} & NEW):
            continue
        n = len(glob.glob(os.path.join(d, "ep*.json")))
        if n < GOMOKU_EPISODES:
            short.append((d, a, b, n))
    total = len(short)
    print(f"gomoku top-off: {total} short pairs, cap {MAX_CONCURRENCY}, resume on", flush=True)
    for d, a, b, n in short:
        print(f"  {a} vs {b}: {n}/{GOMOKU_EPISODES}", flush=True)
    if not total:
        print("nothing to do.", flush=True)
        return
    await asyncio.gather(*(play(d, a, b, 5500 + i, done, total)
                          for i, (d, a, b, n) in enumerate(short)))
    print("\nGOMOKU TOPOFF DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
