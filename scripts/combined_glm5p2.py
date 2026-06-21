"""glm-5p2 as a NEW full-field member: pair it against the 6 incumbents
(deepseek-v4-pro, kimi-k2p6, minimax-m3, glm-5p1, gpt-oss-120b, qwen3p7-plus)
across the 6 head-to-head games, plus 100 hands of independent blackjack vs the
builtin dealer. Only glm-5p2's NEW pairs run; every incumbent-vs-incumbent pair
is already on disk and skipped by per-episode resume.

One shared semaphore; fast games first (kuhn, leduc, holdem_1hand, blackjack)
then slow (connect4, gomoku, holdem_match) to avoid FIFO-starving short episodes.
Per-game dirs/labels/temps match existing conventions so the leaderboard
aggregator finds the ep files:
- kuhn        : runs/kuhn_poker/<o>-coached__vs__glm-5p2-coached, 30/pair, temp 0.0, seat_swap
- leduc       : runs/new_games_experiment/leduc_poker/<o>__vs__glm-5p2 (BARE), 50/pair, 0.6, seat_swap
- holdem_1hand: runs/holdem_1hand/<o>-coached__vs__glm-5p2-coached__r0, 50/pair, 0.6, no seat_swap
- blackjack   : runs/new_games_experiment/independent_blackjack/glm-5p2__vs__dealer, 100, 0.6
- connect4/gomoku: runs/<g>/<g>__<o>-coached__vs__glm-5p2-coached, 10/pair, 0.6, random_open=2, seat_swap
- holdem_match: runs/holdem_match/<o>-coached__vs__glm-5p2-coached, 40/pair, 0.6, no seat_swap
where <o> in the 6 incumbents. Frozen: coached, max_tokens 131072, 900s, 2 retries.
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

NEW = "glm-5p2"
OPPS = ["deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "glm-5p1", "gpt-oss-120b", "qwen3p7-plus"]
RANDOM_OPEN = 2
STARTING_STACK = 200
MAX_HANDS = 30
BLACKJACK_EPISODES = 100

sem = None


def acfg(name, base, temp, timeout):
    return {"type": "model", "name": name, "coached": True,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{base}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": temp, "max_tokens": MAX_TOKENS, "timeout_s": timeout},
            "max_retries": MAX_RETRIES}


def dealer_cfg():
    return {"type": "builtin", "name": "blackjack_dealer"}


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


async def play_bj(done_box, total):
    gdir = f"runs/new_games_experiment/independent_blackjack/{NEW}__vs__dealer"
    os.makedirs(gdir, exist_ok=True)
    runner = Runner(lambda: make_game("independent_blackjack"), on_invalid_action="fallback")
    steps_path = "runs/new_games_experiment/independent_blackjack/steps.jsonl"
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(acfg(NEW, NEW, 0.6, MODEL_TIMEOUT_S), game_name="independent_blackjack"),
                make_agent(dealer_cfg(), game_name="independent_blackjack"),
                episodes=BLACKJACK_EPISODES, seat_swap=False,
                logger=lg, semaphore=sem, episode_dir=gdir,
                on_step=step_tracker(steps_path, f"{NEW}__vs__dealer"))
        done_box[0] += 1
        drop = f"  DROPPED {res.failures}/{BLACKJACK_EPISODES}" if res.failures else ""
        print(f"[{done_box[0]}/{total}] blackjack {NEW} vs dealer: {len(res.episodes)}/{BLACKJACK_EPISODES}{drop}", flush=True)
    except Exception as ex:
        done_box[0] += 1
        print(f"[{done_box[0]}/{total}] blackjack {NEW} FAILED: {ex}", flush=True)
        traceback.print_exc()


async def main():
    global sem
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = [0]
    total = len(OPPS) * 6 + 1  # 6 head-to-head games per opp + 1 blackjack
    print(f"glm-5p2 eval: glm-5p2 vs {OPPS} on 6 games + blackjack ({total} new tasks), "
          f"shared cap {MAX_CONCURRENCY}, resume on\n", flush=True)

    nc = NEW + "-coached"
    tasks = []
    # FAST games first: kuhn, leduc, holdem_1hand
    for i, o in enumerate(OPPS):
        oc = o + "-coached"
        tasks.append(play("kuhn_poker", f"runs/kuhn_poker/{oc}__vs__{nc}", oc, nc, o, NEW,
                          30, 0.0, MODEL_TIMEOUT_S, True, 9500 + i,
                          "runs/kuhn_poker/steps.jsonl", lambda: make_game("kuhn_poker"), done, total))
        tasks.append(play("leduc_poker", f"runs/new_games_experiment/leduc_poker/{o}__vs__{NEW}",
                          o, NEW, o, NEW, 50, 0.6, MODEL_TIMEOUT_S, True, 9500 + i,
                          "runs/new_games_experiment/leduc_poker/steps.jsonl",
                          lambda: make_game("leduc_poker"), done, total))
        tasks.append(play("holdem", f"runs/holdem_1hand/{oc}__vs__{nc}__r0", oc, nc, o, NEW,
                          50, 0.6, HOLDEM_TIMEOUT, False, None, "runs/holdem_1hand/steps.jsonl",
                          lambda: make_game("holdem", {"starting_stack": STARTING_STACK}), done, total))
    # blackjack (fast, vs dealer)
    tasks.append(play_bj(done, total))
    # SLOW games: connect4, gomoku, holdem_match
    for i, o in enumerate(OPPS):
        oc = o + "-coached"
        for g in ("connect4", "gomoku"):
            tasks.append(play(g, f"runs/{g}/{g}__{oc}__vs__{nc}", oc, nc, o, NEW,
                              10, 0.6, BOARD_TIMEOUT, True, 5500 + i,
                              f"runs/{g}/steps.jsonl",
                              (lambda gg=g: make_game(gg, {"random_open": RANDOM_OPEN})), done, total))
        tasks.append(play("holdem_match", f"runs/holdem_match/{oc}__vs__{nc}", oc, nc, o, NEW,
                          40, 0.6, HOLDEM_TIMEOUT, False, None, "runs/holdem_match/steps.jsonl",
                          lambda: make_game("holdem_match", {"starting_stack": STARTING_STACK, "max_hands": MAX_HANDS}),
                          done, total))

    await asyncio.gather(*tasks)
    print(f"\nGLM5P2 EVAL DONE (glm-5p2 vs 6 incumbents on 6 games + blackjack) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
