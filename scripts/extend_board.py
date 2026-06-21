"""Extend connect4 + gomoku pair coverage in place: 10 -> 30 episodes/pair for the
7-model set.

Each pair's (a, b) order and output dir are taken from the EXISTING on-disk dir
name (``<game>__A-coached__vs__B-coached``), so per-episode resume skips ep000-009
already there and fills only ep010-029 — no duplicate/reversed dirs, no recompute.
Strictly the 7-model set (minimax-m2p7 and the foreign claude-* pairs are skipped).

Methodology matches the original board runs: coached agents (names ``A-coached``),
random_open=2, seat_swap=True, temp 0.6, max_tokens 131072, 900s timeout, 2 retries.
connect4 episodes are short; gomoku episodes are long and hang-prone (the timeout x
retry cascade) — restart on stall as usual.

Env: BOARD_TARGET (default 30), GAMES (default "connect4,gomoku"), DRY_RUN=1.
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
BOARD_TIMEOUT = float(os.environ.get("BOARD_TIMEOUT", "900"))
MAX_RETRIES = 2
RANDOM_OPEN = 2
TARGET = int(os.environ.get("BOARD_TARGET", "30"))
GAME_LIST = [g.strip() for g in os.environ.get("GAMES", "connect4,gomoku").split(",") if g.strip()]
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

SEVEN = {"deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "glm-5p1", "gpt-oss-120b",
         "qwen3p7-plus", "glm-5p2"}

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


def parse_pair(game, dirname):
    """'<game>__A-coached__vs__B-coached' -> (a,b) bare, if both in 7-set."""
    pref = game + "__"
    if not dirname.startswith(pref):
        return None
    nm = dirname[len(pref):]
    if "__vs__" not in nm:
        return None
    def bare(x):
        return x[: -len("-coached")] if x.endswith("-coached") else x
    a, b = (bare(x) for x in nm.split("__vs__", 1))
    if a in SEVEN and b in SEVEN and a != b:
        return a, b
    return None


async def play(game, gdir, a, b, seed, steps_path, done_box, total):
    runner = Runner(lambda: make_game(game, {"random_open": RANDOM_OPEN}),
                    on_invalid_action="fallback")
    na, nb = f"{a}-coached", f"{b}-coached"
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(na, a, BOARD_TIMEOUT), game_name=game),
                make_agent(acfg(nb, b, BOARD_TIMEOUT), game_name=game),
                episodes=TARGET, seed=seed, seat_swap=True,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(steps_path, f"{na}__vs__{nb}"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{TARGET}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] {game} {a} vs {b}: {len(res.episodes)}/{TARGET}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] {game} {a} vs {b} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks_meta = []  # (game, dir, a, b)
    for game in GAME_LIST:
        base = f"runs/{game}"
        for d in sorted(glob.glob(base + f"/{game}__*")):
            if not os.path.isdir(d):
                continue
            pr = parse_pair(game, os.path.basename(d))
            if not pr:
                continue
            n = len(glob.glob(os.path.join(d, "ep*.json")))
            if n == 0:
                print(f"  skip empty {os.path.basename(d)}", flush=True)
                continue
            tasks_meta.append((game, d, pr[0], pr[1], n))
    total = len(tasks_meta)
    print(f"extend board: -> {TARGET}/pair, games={GAME_LIST}, {total} pairs, cap {MAX_CONCURRENCY}, resume on", flush=True)
    for game, d, a, b, n in tasks_meta:
        print(f"  {game}: {a} vs {b}: {n}/{TARGET}", flush=True)
    if DRY_RUN:
        print("DRY_RUN — not launching.", flush=True)
        return
    if not total:
        print("nothing to do.", flush=True)
        return
    t0 = time.perf_counter()
    done = [0]
    await asyncio.gather(*(
        play(game, d, a, b, 7000 + i, f"runs/{game}/steps.jsonl", done, total)
        for i, (game, d, a, b, n) in enumerate(tasks_meta)))
    print(f"\nEXTEND BOARD DONE ({total} pairs) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
