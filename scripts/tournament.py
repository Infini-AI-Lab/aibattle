"""Round-robin Hold'em tournament across 5 Fireworks models.

10 unique pairs x 2 reps = 20 games, 30 hands each. Every hand is an
independent deal (no seat-swap duplicate): each episode draws its own cards. Full
per-game logs go to runs/tournament/<a>__vs__<b>__rN/match.jsonl; a trimmed
aggregate (no raw chain-of-thought, to stay loadable) is saved incrementally to
runs/tournament/tournament_data.json for analysis.
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

# qwen3p6-plus excluded (restrictive per-model 429 limit on this account).
MODELS = os.environ.get("MODELS",
    "deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7").split(",")
HANDS = int(os.environ.get("HANDS", "20"))   # hands per game (Hold'em Lite)
REPS = 1
MAX_CONCURRENCY = 128  # GLOBAL cap on concurrent hands across all games
OUT = "runs/tournament"
os.makedirs(OUT, exist_ok=True)
# Deals are fully random and independent: every hand draws its own OS-entropy
# deal seed inside the runner (seed=None), and that seed is saved in each
# ep<NNN>.json. There is no run-level seed to log or re-pass — a resumed run just
# fills missing hands with fresh independent deals.


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


def trim(episodes: list) -> list:
    """Drop the heavy raw chain-of-thought from the aggregate (kept per-episode)."""
    out = []
    for e in episodes:
        e2 = dict(e)
        steps = []
        for s in e.get("steps", []):
            s2 = dict(s)
            resp = dict(s2.get("response") or {})
            resp.pop("raw_output", None)   # huge; full copy lives in the ep file
            resp.pop("prompt", None)       # repeats rules each step; kept in ep file
            s2["response"] = resp
            steps.append(s2)
        e2["steps"] = steps
        out.append(e2)
    return out


async def main():
    pairs = list(itertools.combinations(MODELS, 2))  # 10
    games = []
    gid = 0
    for rep in range(REPS):
        for a, b in pairs:
            games.append((gid, a, b, rep))
            gid += 1

    all_games = []
    done = 0
    t0 = time.perf_counter()
    # All games run concurrently, sharing ONE global semaphore so the total
    # number of in-flight model calls is bounded (avoids rate limits) while the
    # wall-clock collapses from sum-of-games to ~slowest-game.
    global_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    print(f"Starting tournament: {len(games)} games x {HANDS} hands, "
          f"random independent deals, temp=0.6, ALL IN PARALLEL (global cap "
          f"{MAX_CONCURRENCY} concurrent calls, per-episode resume on)\n", flush=True)

    def save():
        json.dump({"models": MODELS, "hands": HANDS, "reps": REPS,
                   "games": all_games},
                  open(os.path.join(OUT, "tournament_data.json"), "w"))

    async def play(gid, a, b, rep):
        nonlocal done
        gdir = os.path.join(OUT, f"{a}__vs__{b}__r{rep}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game("holdem", {"starting_stack": 200}),
                        on_invalid_action="fallback")
        ta = time.perf_counter()
        try:
            # No-op logger: each hand persists its own self-contained file under
            # gdir via episode_dir, which is also the resume unit. No shared
            # match.jsonl to rewrite or corrupt.
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a), game_name="holdem"),
                    make_agent(acfg(b), game_name="holdem"),
                    episodes=HANDS, seat_swap=False,
                    logger=lg, semaphore=global_sem, episode_dir=gdir,
                )
            # append + save in one synchronous (await-free) block -> race-safe
            all_games.append({"gid": gid, "a": a, "b": b, "rep": rep,
                              "episodes": trim(res.episodes)})
            save()
            tally = defaultdict(float)
            for e in res.episodes:
                for seat, nm in e["seat_assignment"].items():
                    tally[nm] += e["returns"][seat]
            done += 1
            print(f"[{done}/{len(games)}] {a} vs {b} (r{rep}) done in "
                  f"{time.perf_counter() - ta:.0f}s | chips: "
                  f"{ {k: round(v, 1) for k, v in tally.items()} }", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{len(games)}] {a} vs {b} (r{rep}) FAILED: {ex}", flush=True)
            traceback.print_exc()

    await asyncio.gather(*(play(*g) for g in games))
    save()
    print(f"\nTOURNAMENT DONE in {time.perf_counter() - t0:.0f}s, "
          f"{len(all_games)}/{len(games)} games completed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
