"""Round-robin tournament for the board games (Connect Four + Gomoku).

For each game: 10 unique model pairs x EPISODES games each, all run in parallel
under one global semaphore. Per-game aggregate saved incrementally to
runs/board_tournament/<game>_data.json; full per-match logs under
runs/board_tournament/<game>__<a>__vs__<b>/match.jsonl.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
import traceback
from collections import defaultdict

os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

GAMES = ["connect4", "gomoku"]
MODELS = ["deepseek-v4-pro", "gpt-oss-120b", "kimi-k2p6", "glm-5p1", "qwen3p6-plus"]
EPISODES = 50
MAX_CONCURRENCY = 128           # global cap; 300+ over-runs the Fireworks limit
RANDOM_OPEN = 2
OUT = "runs/board_tournament"
os.makedirs(OUT, exist_ok=True)


def acfg(name: str) -> dict:
    return {
        "type": "model", "name": name,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{name}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": 0.0, "max_tokens": 16384, "timeout_s": 300,
        },
        "max_retries": 2,
    }


def trim(episodes: list) -> list:
    out = []
    for e in episodes:
        e2 = dict(e)
        steps = []
        for s in e.get("steps", []):
            s2 = dict(s)
            resp = dict(s2.get("response") or {})
            resp.pop("raw_output", None)   # full copy lives in match.jsonl
            s2["response"] = resp
            steps.append(s2)
        e2["steps"] = steps
        out.append(e2)
    return out


async def main():
    pairs = list(itertools.combinations(MODELS, 2))  # 10
    data = {g: {"game": g, "episodes_per_pair": EPISODES, "models": MODELS,
                "games": []} for g in GAMES}
    done = 0
    total = len(GAMES) * len(pairs)
    global_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    print(f"Board tournament: {len(GAMES)} games x {len(pairs)} pairs x "
          f"{EPISODES} episodes = {total * EPISODES} games, global cap "
          f"{MAX_CONCURRENCY}\n", flush=True)

    def save(game):
        json.dump(data[game], open(os.path.join(OUT, f"{game}_data.json"), "w"))

    async def play(game, a, b, seed):
        nonlocal done
        gdir = os.path.join(OUT, f"{game}__{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda g=game: make_game(g, {"random_open": RANDOM_OPEN}),
                        on_invalid_action="fallback")
        ta = time.perf_counter()
        try:
            with MatchLogger(os.path.join(gdir, "match.jsonl")) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a), game_name=game),
                    make_agent(acfg(b), game_name=game),
                    episodes=EPISODES, seed=seed, seat_swap=True,
                    logger=lg, semaphore=global_sem,
                )
            data[game]["games"].append({"a": a, "b": b, "seed": seed,
                                        "episodes": trim(res.episodes)})
            save(game)
            wins = defaultdict(int); draws = 0
            for e in res.episodes:
                w = e.get("winner_name")
                if w:
                    wins[w] += 1
                else:
                    draws += 1
            done += 1
            got = len(res.episodes)
            drop = f"  DROPPED {res.failures}/{EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] {game}: {a} vs {b} done in "
                  f"{time.perf_counter() - ta:.0f}s | episodes={got}/{EPISODES}{drop} "
                  f"| wins={dict(wins)} draws={draws}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] {game}: {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    tasks = []
    seed = 5000
    for game in GAMES:
        for a, b in pairs:
            tasks.append(play(game, a, b, seed))
            seed += 1
    await asyncio.gather(*tasks)
    for g in GAMES:
        save(g)
    print(f"\nBOARD TOURNAMENT DONE in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
