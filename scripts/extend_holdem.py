"""Extend the existing hold'em pair coverage in place:
  - holdem_1hand : 50 -> 100 episodes/pair
  - holdem_match : 40 ->  60 episodes/pair

Covers the 8-model set (7-set + minimax-m2p7). Each pair's (a, b) order and its
output dir are taken from the EXISTING on-disk dir name, so per-episode resume
skips the episodes already there and fills only the new ones — no duplicate or
reversed dirs, no recompute. Methodology matches the original runs exactly:
coached agents, seat_swap=False (fixed seat layout already on disk), holdem
starting_stack 200, holdem_match max_hands 30, temp 0.6, max_tokens 131072,
900s timeout, 2 retries. Fast game (1hand) launched before the long one (match).
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
HOLDEM_TIMEOUT = float(os.environ.get("HOLDEM_TIMEOUT", "900"))
MAX_RETRIES = 2
STARTING_STACK = 200
MAX_HANDS = 30

H1_TARGET = int(os.environ.get("H1_TARGET", "100"))
MATCH_TARGET = int(os.environ.get("MATCH_TARGET", "60"))

EIGHT = {"deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "glm-5p1", "gpt-oss-120b",
         "qwen3p7-plus", "glm-5p2", "minimax-m2p7"}

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
        pub = info.get("public") or {}
        rec = {"t": round(time.time(), 1), "pair": pair, "ep": info["episode"],
               "step": info["step"], "agent": info["agent_name"],
               "action": (info.get("action") or "")[:40]}
        h = pub.get("match_hand") or pub.get("table_hand")
        if h is not None:
            rec["hand"] = h
        try:
            open(path, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return on_step


def parse_pair(dirname):
    """('A-coached__vs__B-coached[__r0]') -> (a_bare, b_bare) or None if not 8-set."""
    name = dirname
    if name.endswith("__r0"):
        name = name[: -len("__r0")]
    if "__vs__" not in name:
        return None
    left, right = name.split("__vs__", 1)
    def bare(x):
        return x[: -len("-coached")] if x.endswith("-coached") else x
    a, b = bare(left), bare(right)
    if a in EIGHT and b in EIGHT and a != b:
        return a, b
    return None


async def play(game, gdir, a, b, episodes, gmaker, steps_path, done_box, total):
    os.makedirs(gdir, exist_ok=True)
    runner = Runner(gmaker, on_invalid_action="fallback")
    na, nb = f"{a}-coached", f"{b}-coached"
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(na, a, HOLDEM_TIMEOUT), game_name=game),
                make_agent(acfg(nb, b, HOLDEM_TIMEOUT), game_name=game),
                episodes=episodes, seed=None, seat_swap=False,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(steps_path, f"{na}__vs__{nb}"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{episodes}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] {game} {a} vs {b}: {len(res.episodes)}/{episodes}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] {game} {a} vs {b} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = [0]

    # Enumerate existing 8-set pair dirs for each game (preserve a/b order from disk).
    def collect(base):
        out = []
        for d in sorted(glob.glob(base + "/*")):
            if not os.path.isdir(d):
                continue
            pr = parse_pair(os.path.basename(d))
            if not pr:
                continue
            n = len(glob.glob(os.path.join(d, "ep*.json")))
            if n == 0:
                # Stale/empty dir (e.g. bare-label leftover) — skip so we don't
                # create a duplicate logical pair the aggregator would double-count.
                print(f"  skip empty dir {os.path.basename(d)}", flush=True)
                continue
            out.append((d, pr))
        return out

    h1_dirs = collect("runs/holdem_1hand")
    match_dirs = collect("runs/holdem_match")

    # Two m2p7 pairs were never played (m2p7 was dropped before glm-5p2 / qwen
    # joined). Create them from scratch so m2p7 is a complete 8th member. Order
    # follows the convention "older opponent first, newer model second".
    MISSING = [("minimax-m2p7", "glm-5p2"), ("minimax-m2p7", "qwen3p7-plus")]
    h1_have = {os.path.basename(d) for d, _ in h1_dirs}
    for a, b in MISSING:
        nm = f"{a}-coached__vs__{b}-coached__r0"
        if nm not in h1_have:
            h1_dirs.append((os.path.join("runs/holdem_1hand", nm), (a, b)))
            print(f"  new pair (from 0) holdem_1hand: {nm}", flush=True)
    match_have = {os.path.basename(d) for d, _ in match_dirs}
    for a, b in MISSING:
        nm = f"{a}-coached__vs__{b}-coached"
        if nm not in match_have:
            match_dirs.append((os.path.join("runs/holdem_match", nm), (a, b)))
            print(f"  new pair (from 0) holdem_match: {nm}", flush=True)

    total = len(h1_dirs) + len(match_dirs)
    print(f"extend holdem: holdem_1hand {len(h1_dirs)} pairs -> {H1_TARGET}/pair, "
          f"holdem_match {len(match_dirs)} pairs -> {MATCH_TARGET}/pair "
          f"({total} tasks), cap {MAX_CONCURRENCY}, resume on\n", flush=True)

    tasks = []
    # FAST first: holdem_1hand
    for d, (a, b) in h1_dirs:
        tasks.append(play("holdem", d, a, b, H1_TARGET,
                          lambda: make_game("holdem", {"starting_stack": STARTING_STACK}),
                          "runs/holdem_1hand/steps.jsonl", done, total))
    # SLOW: holdem_match
    for d, (a, b) in match_dirs:
        tasks.append(play("holdem_match", d, a, b, MATCH_TARGET,
                          lambda: make_game("holdem_match", {"starting_stack": STARTING_STACK, "max_hands": MAX_HANDS}),
                          "runs/holdem_match/steps.jsonl", done, total))

    await asyncio.gather(*tasks)
    print(f"\nEXTEND HOLDEM DONE ({total} pairs) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
