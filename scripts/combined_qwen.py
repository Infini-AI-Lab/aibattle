"""qwen3p7-plus comparison vs minimax-m3 / deepseek-v4-pro / kimi-k2p6 on
holdem (1-hand), holdem_match, independent_blackjack — all under ONE shared
semaphore so the 64-call budget stays saturated and holdem_match (the long pole)
isn't starved.

Only qwen's NEW pairings actually run; the incumbent trio already played each
other in these games during the m3 eval (same coached/temp/token settings), so
per-episode resume skips those instantly. Faithful dir naming so resume reuses:
- holdem (1-hand) : runs/holdem_1hand/<a>-coached__vs__<b>-coached__r0, no seed,
                    50 hands/pair (random deals).
- holdem_match    : runs/holdem_match/<a>-coached__vs__<b>-coached, no seed,
                    40 matches/pair (<=30 hands each).
- blackjack       : runs/new_games_experiment/independent_blackjack/<m>__vs__dealer
                    BARE label, player_0=model vs builtin dealer, 100 hands.

Settings frozen: coached, temp 0.6, max_tokens 131072, 900s timeout, 2 retries.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
import traceback
from collections import defaultdict

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
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "900"))
MAX_RETRIES = 2

# 4-model roster. Order fixes combinations() so incumbent pair dirs match on disk.
MODELS = os.environ.get(
    "MODELS", "deepseek-v4-pro,kimi-k2p6,minimax-m3,qwen3p7-plus").split(",")
COACHED = [m + "-coached" for m in MODELS]

HANDS = int(os.environ.get("HANDS", "50"))            # holdem 1-hand
MATCH_EPISODES = int(os.environ.get("MATCH_EPISODES", "40"))
BLACKJACK_EPISODES = int(os.environ.get("BLACKJACK_EPISODES", "100"))
MAX_HANDS = 30
STARTING_STACK = 200

H1_OUT = "runs/holdem_1hand"
MATCH_OUT = "runs/holdem_match"
BJ_OUT = "runs/new_games_experiment/independent_blackjack"


def acfg(name: str, base: str, timeout: float) -> dict:
    return {
        "type": "model", "name": name, "coached": True,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{base}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "timeout_s": timeout,
        },
        "max_retries": MAX_RETRIES,
    }


def dealer_cfg() -> dict:
    return {"type": "builtin", "name": "blackjack_dealer"}


def step_tracker(path: str, pair: str):
    def on_step(info):
        pub = info.get("public") or {}
        rec = {"t": round(time.time(), 1), "pair": pair, "ep": info["episode"],
               "step": info["step"], "agent": info["agent_name"],
               "action": (info.get("action") or "")[:40]}
        hand = pub.get("match_hand") or pub.get("table_hand")
        if hand is not None:
            rec["hand"] = hand
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return on_step


def _trim_match(e):
    return {k: e[k] for k in ("episode", "seat_assignment", "returns", "winner",
                              "winner_name", "length", "hands_played",
                              "final_stacks", "stack_diff", "reason",
                              "hand_summaries") if k in e}


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = 0

    pairs = list(itertools.combinations(COACHED, 2))   # 6 (3 incumbent + 3 qwen)
    total = len(pairs) * 2 + len(MODELS)               # h1 + match + blackjack
    print(f"qwen eval: holdem_1hand {len(pairs)} + holdem_match {len(pairs)} pairs "
          f"+ blackjack {len(MODELS)} vs dealer, ONE shared semaphore cap "
          f"{MAX_CONCURRENCY}, per-episode resume on\n", flush=True)

    async def play_h1(a, b):
        nonlocal done
        gdir = os.path.join(H1_OUT, f"{a}__vs__{b}__r0")
        os.makedirs(gdir, exist_ok=True)
        ba, bb = a[:-len("-coached")], b[:-len("-coached")]
        runner = Runner(lambda: make_game("holdem", {"starting_stack": STARTING_STACK}),
                        on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, ba, HOLDEM_TIMEOUT), game_name="holdem"),
                    make_agent(acfg(b, bb, HOLDEM_TIMEOUT), game_name="holdem"),
                    episodes=HANDS, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(os.path.join(H1_OUT, "steps.jsonl"), f"{a}__vs__{b}"))
            done += 1
            drop = f"  DROPPED {res.failures}/{HANDS}" if res.failures else ""
            print(f"[{done}/{total}] holdem_1hand {a} vs {b}: {len(res.episodes)}/{HANDS}{drop}", flush=True)
        except Exception as ex:
            done += 1; print(f"[{done}/{total}] holdem_1hand {a} vs {b} FAILED: {ex}", flush=True); traceback.print_exc()

    def match_factory():
        return make_game("holdem_match", {"starting_stack": STARTING_STACK, "max_hands": MAX_HANDS})

    async def play_match(a, b):
        nonlocal done
        gdir = os.path.join(MATCH_OUT, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        ba, bb = a[:-len("-coached")], b[:-len("-coached")]
        runner = Runner(match_factory, on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, ba, HOLDEM_TIMEOUT), game_name="holdem_match"),
                    make_agent(acfg(b, bb, HOLDEM_TIMEOUT), game_name="holdem_match"),
                    episodes=MATCH_EPISODES, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(os.path.join(MATCH_OUT, "steps.jsonl"), f"{a}__vs__{b}"))
            done += 1
            drop = f"  DROPPED {res.failures}/{MATCH_EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] holdem_match {a} vs {b}: {len(res.episodes)}/{MATCH_EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1; print(f"[{done}/{total}] holdem_match {a} vs {b} FAILED: {ex}", flush=True); traceback.print_exc()

    async def play_bj(m):  # bare label
        nonlocal done
        gdir = os.path.join(BJ_OUT, f"{m}__vs__dealer")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game("independent_blackjack"), on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(m, m, MODEL_TIMEOUT_S), game_name="independent_blackjack"),
                    make_agent(dealer_cfg(), game_name="independent_blackjack"),
                    episodes=BLACKJACK_EPISODES, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(os.path.join(BJ_OUT, "steps.jsonl"), f"{m}__vs__dealer"))
            done += 1
            drop = f"  DROPPED {res.failures}/{BLACKJACK_EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] blackjack {m} vs dealer: {len(res.episodes)}/{BLACKJACK_EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1; print(f"[{done}/{total}] blackjack {m} FAILED: {ex}", flush=True); traceback.print_exc()

    tasks = [play_h1(a, b) for a, b in pairs]
    tasks += [play_match(a, b) for a, b in pairs]
    tasks += [play_bj(m) for m in MODELS]
    await asyncio.gather(*tasks)
    print(f"\nQWEN EVAL DONE (holdem_1hand + holdem_match + blackjack) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
