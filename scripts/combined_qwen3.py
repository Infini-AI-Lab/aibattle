"""Extend qwen3p7-plus to the FULL 6-model field: add its two missing opponents
(glm-5p1, gpt-oss-120b) across the 6 head-to-head games. Only qwen's 2 new pairs
per game run (all other pairs already on disk; blackjack vs-dealer already done
for all three, so it is excluded here). One shared semaphore; fast games first.

Per-game dirs/labels/temps match the existing conventions so the ep files land
where the leaderboard aggregator expects them:
- kuhn        : runs/kuhn_poker/<o>-coached__vs__qwen3p7-plus-coached, 30/pair, temp 0.0
- leduc       : runs/new_games_experiment/leduc_poker/<o>__vs__qwen3p7-plus (BARE), 50/pair, 0.6
- holdem_1hand: runs/holdem_1hand/<o>-coached__vs__qwen3p7-plus-coached__r0, 50/pair, 0.6, no seat_swap
- connect4/gomoku: runs/<g>/<g>__<o>-coached__vs__qwen3p7-plus-coached, 10/pair, 0.6, random_open=2
- holdem_match: runs/holdem_match/<o>-coached__vs__qwen3p7-plus-coached, 40/pair, 0.6, no seat_swap
where <o> in {glm-5p1, gpt-oss-120b}. Frozen: coached, max_tokens 131072, 900s, 2 retries.
"""
from __future__ import annotations

import asyncio
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
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "900"))
BOARD_TIMEOUT = float(os.environ.get("BOARD_TIMEOUT", "900"))
HOLDEM_TIMEOUT = float(os.environ.get("HOLDEM_TIMEOUT", "900"))
MAX_RETRIES = 2

OPPS = ["glm-5p1", "gpt-oss-120b"]
QWEN = "qwen3p7-plus"
RANDOM_OPEN = 2
STARTING_STACK = 200
MAX_HANDS = 30

sem = None


def acfg(name, base, temp, timeout):
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{base}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": temp, "max_tokens": MAX_TOKENS, "timeout_s": timeout},
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


def _trim_match(e):
    return {k: e[k] for k in ("episode", "seat_assignment", "returns", "winner",
                              "winner_name", "length", "hands_played", "final_stacks",
                              "stack_diff", "reason", "hand_summaries") if k in e}


async def play(game, gdir, a, b, ba, bb, episodes, temp, timeout, seat_swap,
               seed, steps_path, gmaker, done_box, total):
    os.makedirs(gdir, exist_ok=True)
    runner = Runner(gmaker, on_invalid_action="fallback")
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(a, ba, temp, timeout), game_name=game),
                make_agent(acfg(b, bb, temp, timeout), game_name=game),
                episodes=episodes, seed=seed, seat_swap=seat_swap,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(steps_path, f"{a}__vs__{b}"))
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
    total = len(OPPS) * 6
    print(f"qwen eval-3: qwen3p7-plus vs {OPPS} on 6 games ({total} new pairs), "
          f"shared cap {MAX_CONCURRENCY}, resume on\n", flush=True)

    tasks = []
    # FAST games first: kuhn, leduc, holdem_1hand
    for i, o in enumerate(OPPS):
        oc, qc = o + "-coached", QWEN + "-coached"
        tasks.append(play("kuhn_poker", f"runs/kuhn_poker/{oc}__vs__{qc}", oc, qc, o, QWEN,
                          30, 0.0, MODEL_TIMEOUT_S, True, 9500 + i,
                          "runs/kuhn_poker/steps.jsonl", lambda: make_game("kuhn_poker"), done, total))
        tasks.append(play("leduc_poker", f"runs/new_games_experiment/leduc_poker/{o}__vs__{QWEN}",
                          o, QWEN, o, QWEN, 50, 0.6, MODEL_TIMEOUT_S, True, 9500 + i,
                          "runs/new_games_experiment/leduc_poker/steps.jsonl",
                          lambda: make_game("leduc_poker"), done, total))
        tasks.append(play("holdem", f"runs/holdem_1hand/{oc}__vs__{qc}__r0", oc, qc, o, QWEN,
                          50, 0.6, HOLDEM_TIMEOUT, False, None, "runs/holdem_1hand/steps.jsonl",
                          lambda: make_game("holdem", {"starting_stack": STARTING_STACK}), done, total))
    # SLOW games: connect4, gomoku, holdem_match
    for i, o in enumerate(OPPS):
        oc, qc = o + "-coached", QWEN + "-coached"
        for g in ("connect4", "gomoku"):
            tasks.append(play(g, f"runs/{g}/{g}__{oc}__vs__{qc}", oc, qc, o, QWEN,
                              10, 0.6, BOARD_TIMEOUT, True, 5500 + i,
                              f"runs/{g}/steps.jsonl",
                              (lambda gg=g: make_game(gg, {"random_open": RANDOM_OPEN})), done, total))
        tasks.append(play("holdem_match", f"runs/holdem_match/{oc}__vs__{qc}", oc, qc, o, QWEN,
                          40, 0.6, HOLDEM_TIMEOUT, False, None, "runs/holdem_match/steps.jsonl",
                          lambda: make_game("holdem_match", {"starting_stack": STARTING_STACK, "max_hands": MAX_HANDS}),
                          done, total))

    await asyncio.gather(*tasks)
    print(f"\nQWEN EVAL3 DONE (qwen vs glm-5p1 + gpt-oss-120b on 6 games) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
