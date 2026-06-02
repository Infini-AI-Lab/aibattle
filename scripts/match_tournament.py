"""Round-robin Heads-Up Hold'em MATCH-mode tournament.

Each unique model pair plays EPISODES matches (a match = up to MAX_HANDS hands,
stacks carried, match-level winner). The primary metric is match win rate, which
is far less variance-prone than single-hand chip delta. Each match is one
episode and is persisted per-episode (resume: relaunch to continue).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time
import traceback
from collections import defaultdict

os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

# qwen3p6-plus excluded (restrictive per-model 429 limit on this account).
MODELS = os.environ.get("MODELS",
    "deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7").split(",")
EPISODES = int(os.environ.get("EPISODES", "10"))  # matches per pair (independent deals);
                            # raise later to add more — per-episode resume reuses.
MAX_HANDS = 30
STARTING_STACK = 200            # 100bb (blinds 1/2)
MAX_CONCURRENCY = 128
OUT = "runs/match_tournament"
os.makedirs(OUT, exist_ok=True)
# Deals are fully random and independent: every match draws its own OS-entropy
# deal seeds inside the runner (seed=None), saved per ep<NNN>.json. No run-level
# seed to log or re-pass — a resumed run fills missing matches with fresh deals.


def acfg(name: str) -> dict:
    return {
        "type": "model", "name": name,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{name}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": 0.6, "max_tokens": 131072,
            "timeout_s": int(os.environ.get("HOLDEM_TIMEOUT", "900")),
        },
        "max_retries": 2,
    }


def game_factory():
    return make_game("holdem_match",
                     {"starting_stack": STARTING_STACK, "max_hands": MAX_HANDS})


async def main():
    pairs = list(itertools.combinations(MODELS, 2))
    total = len(pairs)
    data = {"mode": "match", "models": MODELS, "episodes_per_pair": EPISODES,
            "max_hands": MAX_HANDS, "starting_stack": STARTING_STACK,
            "pairs": []}
    done = 0
    global_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    print(f"Match tournament: {total} pairs x {EPISODES} matches "
          f"({MAX_HANDS} hands each), random independent deals, temp=0.6, "
          f"global cap {MAX_CONCURRENCY}, per-episode resume on\n", flush=True)

    def save():
        json.dump(data, open(os.path.join(OUT, "match_data.json"), "w"))

    async def play(a, b):
        nonlocal done
        gdir = os.path.join(OUT, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(game_factory, on_invalid_action="fallback")
        ta = time.perf_counter()
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a), game_name="holdem_match"),
                    make_agent(acfg(b), game_name="holdem_match"),
                    episodes=EPISODES, seat_swap=False,
                    logger=lg, semaphore=global_sem, episode_dir=gdir,
                )
            # Match win rate per model (by seat assignment).
            wins = defaultdict(int); draws = 0
            for e in res.episodes:
                w = e.get("winner_name")
                if w:
                    wins[w] += 1
                else:
                    draws += 1
            data["pairs"].append({"a": a, "b": b,
                                  "episodes": [_trim(e) for e in res.episodes]})
            save()
            done += 1
            got = len(res.episodes)
            drop = f"  DROPPED {res.failures}/{EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] {a} vs {b} done in {time.perf_counter()-ta:.0f}s "
                  f"| matches={got}/{EPISODES}{drop} | wins={dict(wins)} draws={draws}",
                  flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    specs = list(pairs)
    random.Random(99).shuffle(specs)  # deterministic launch order (slow/fast interleave)
    await asyncio.gather(*(play(a, b) for a, b in specs))
    save()
    print(f"\nMATCH TOURNAMENT DONE in {time.perf_counter()-t0:.0f}s", flush=True)


def _trim(e: dict) -> dict:
    """Keep match-level fields + per-hand summaries; drop heavy step traces."""
    return {k: e[k] for k in ("episode", "seat_assignment", "returns", "winner",
                              "winner_name", "length", "hands_played",
                              "final_stacks", "stack_diff", "reason",
                              "hand_summaries") if k in e}


if __name__ == "__main__":
    asyncio.run(main())
