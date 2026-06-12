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
import time
import traceback
from collections import defaultdict

os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

import _heartbeat  # debug per-move heartbeat -> data-log/ (gitignored)

# qwen3p6-plus excluded (restrictive per-model 429 limit on this account).
MODELS = os.environ.get("MODELS",
    "deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7").split(",")
# Coaching: COACHED=1 coaches every model; "<name>#coached" coaches just that one.
# A coached model is an independent participant named "<base>-coached".
_COACH_ALL = os.environ.get("COACHED", "").lower() not in ("", "0", "false", "no")
def _coach_label(spec):
    base = spec.split("#", 1)[0].strip()
    return f"{base}-coached" if (_COACH_ALL or spec.strip().endswith("#coached")) else base
MODELS = [_coach_label(s) for s in MODELS]
SESSIONS = int(os.environ.get("TABLE_SESSIONS", "50"))  # seat-rotated; raise
                            # later to add more — per-episode resume reuses sessions.
MAX_HANDS = 40
STARTING_STACK = 200            # 100bb (blinds 1/2)
MAX_CONCURRENCY = 128
OUT = os.environ.get("OUT", "runs/holdem_table")
os.makedirs(OUT, exist_ok=True)
# Deals are fully random and independent: every session draws its own OS-entropy
# deal seed inside the runner (seed=None), and seats are a random permutation
# derived from that seed. The seed is saved per ep<NNN>.json; a resumed run fills
# missing sessions with fresh independent deals. No run-level seed to log.


def acfg(label: str) -> dict:
    coached = label.endswith("-coached")
    base = label[: -len("-coached")] if coached else label
    return {
        "type": "model", "name": label, "coached": coached,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{base}",
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
    hb_fh, hb_path = _heartbeat.open_log("table")
    print(f"Table tournament: {n}-player table, {SESSIONS} sessions x "
          f"{MAX_HANDS} hands, random independent deals + random seats, temp=0.6, "
          f"global cap {MAX_CONCURRENCY}, per-episode resume on\n"
          f"debug heartbeat -> {hb_path}\n", flush=True)

    agents = [make_agent(acfg(m), game_name="holdem_table") for m in MODELS]
    gdir = os.path.join(OUT, "table")
    os.makedirs(gdir, exist_ok=True)
    with MatchLogger(None) as lg:
        res = await runner.run_table(
            agents, episodes=SESSIONS, logger=lg,
            semaphore=global_sem, episode_dir=gdir, seat_rotate=True,
            on_step=_heartbeat.make_cb(hb_fh, "table"),
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
            "sessions": SESSIONS, "max_hands": MAX_HANDS,
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
