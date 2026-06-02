"""Multi-Agent Hold'em TABLE-mode tournament.

All models sit at one table and play SESSIONS table sessions (a session = up to
MAX_HANDS hands, ranking output). Seat assignment rotates each session (and the
button is randomized per session), neutralizing positional bias over repeated
trials. The primary metric is average finishing rank; secondary: top-1 rate,
average final stack. Each session is one episode, persisted per-episode (resume:
relaunch to continue).
"""

from __future__ import annotations

import asyncio
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
MODELS = ["deepseek-v4-pro", "gpt-oss-120b", "kimi-k2p6", "glm-5p1", "minimax-m2p7"]
SESSIONS = int(os.environ.get("TABLE_SESSIONS", "50"))  # seat-rotated; raise
                            # later to add more — per-episode resume reuses sessions.
MAX_HANDS = 40
STARTING_STACK = 200            # 100bb (blinds 1/2)
MAX_CONCURRENCY = 128
OUT = "runs/table_tournament"
os.makedirs(OUT, exist_ok=True)
# Per-run base seed: random by default (fresh deals each run), overridable via
# RUN_SEED, and LOGGED (banner + data.json) for on-demand reproducibility. It
# seeds the whole session/deal sequence. (RUN_SEED=8000 reproduces v1 deals.)
RUN_SEED = int(os.environ.get("RUN_SEED", random.SystemRandom().randrange(2**31)))


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
    return make_game("holdem_table", {"num_players": len(MODELS),
                                      "starting_stack": STARTING_STACK,
                                      "max_hands": MAX_HANDS})


async def main():
    n = len(MODELS)
    runner = Runner(game_factory, on_invalid_action="fallback")
    global_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    print(f"Table tournament: {n}-player table, {SESSIONS} sessions x "
          f"{MAX_HANDS} hands, RUN_SEED={RUN_SEED}, temp=0.6, global cap "
          f"{MAX_CONCURRENCY}, per-episode resume on\n", flush=True)

    agents = [make_agent(acfg(m), game_name="holdem_table") for m in MODELS]
    gdir = os.path.join(OUT, "table")
    os.makedirs(gdir, exist_ok=True)
    with MatchLogger(None) as lg:
        res = await runner.run_table(
            agents, episodes=SESSIONS, seed=RUN_SEED, logger=lg,
            semaphore=global_sem, episode_dir=gdir, seat_rotate=True,
        )

    # Aggregate by model NAME across sessions (seats rotate, so map via the
    # session's seat_assignment + rank_of).
    rank_sum = defaultdict(float); top1 = defaultdict(int)
    stack_sum = defaultdict(float); appearances = defaultdict(int)
    sessions = []
    for e in res.episodes:
        seat_to_name = e["seat_assignment"]          # player_i -> model name
        rank_of = e["rank_of"]                       # player_i -> rank
        final = e["final_stacks"]
        per = {}
        for seat, name in seat_to_name.items():
            r = rank_of[seat]
            rank_sum[name] += r
            stack_sum[name] += final[seat]
            appearances[name] += 1
            if r == 1:
                top1[name] += 1
            per[name] = {"rank": r, "final_stack": final[seat]}
        sessions.append({"episode": e["episode"], "reason": e.get("reason"),
                         "hands_played": e.get("hands_played"), "result": per})

    summary = []
    for m in MODELS:
        ap = appearances[m] or 1
        summary.append({
            "model": m,
            "avg_rank": round(rank_sum[m] / ap, 3),
            "top1_rate": round(top1[m] / ap, 3),
            "avg_final_stack": round(stack_sum[m] / ap, 1),
            "sessions": appearances[m],
        })
    summary.sort(key=lambda r: r["avg_rank"])

    data = {"mode": "table", "models": MODELS, "num_players": n,
            "sessions": SESSIONS, "max_hands": MAX_HANDS, "run_seed": RUN_SEED,
            "starting_stack": STARTING_STACK, "summary": summary,
            "session_results": sessions, "failures": res.failures}
    json.dump(data, open(os.path.join(OUT, "table_data.json"), "w"))

    drop = f"  ({res.failures} sessions DROPPED)" if res.failures else ""
    print(f"Completed {len(res.episodes)}/{SESSIONS} sessions{drop} in "
          f"{time.perf_counter()-t0:.0f}s\n", flush=True)
    print(f"{'model':<18} avg_rank  top1%   avg_stack", flush=True)
    for r in summary:
        print(f"{r['model']:<18} {r['avg_rank']:>7}  {r['top1_rate']*100:>4.0f}%  "
              f"{r['avg_final_stack']:>9}", flush=True)
    print(f"\nTABLE TOURNAMENT DONE in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
