"""qwen3p7-plus comparison vs minimax-m3 / deepseek-v4-pro / kimi-k2p6 on
kuhn, connect4, gomoku, leduc — all under ONE shared semaphore.

Only qwen's NEW pairings run; the incumbent trio already played each other in
these games during the m3 eval (same settings), so per-episode resume skips
them. Faithful dir naming + per-game model ORDER so resume reuses incumbent dirs:
- kuhn      : runs/kuhn_poker/<a>-coached__vs__<b>-coached, seat_swap, 30/pair,
              temp 0.0. Order deepseek<kimi<m3 (OLD_MODELS) so pair dirs match.
- connect4  : runs/connect4/connect4__<a>-coached__vs__<b>-coached, seat_swap,
              10/pair, temp 0.6, random_open=2. Same order.
- gomoku    : runs/gomoku/gomoku__<a>-coached__vs__<b>-coached, 10/pair, temp 0.6.
- leduc     : runs/new_games_experiment/leduc_poker/<a>__vs__<b>, BARE labels,
              seat_swap, 50/pair, temp 0.6. Order kimi<deepseek<m3 (NG_MODELS)
              so pair dirs match the existing 'kimi-k2p6__vs__deepseek-v4-pro' etc.

Settings frozen: coached, max_tokens 131072, 900s timeout, 2 retries. Writes ep
files only (does NOT rewrite the full *_data.json aggregates); leaderboards are
aggregated from ep files, filtered to the 4 models.
"""
from __future__ import annotations

import asyncio
import itertools
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
MAX_RETRIES = 2

# Per-game model order so itertools.combinations reproduces the incumbent dirs.
BOARD_ORDER = ["deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "qwen3p7-plus"]   # -coached
LEDUC_ORDER = ["kimi-k2p6", "deepseek-v4-pro", "minimax-m3", "qwen3p7-plus"]   # bare

KUHN_EPISODES = int(os.environ.get("KUHN_EPISODES", "30"))
BOARD_EPISODES = int(os.environ.get("BOARD_EPISODES", "10"))
LEDUC_EPISODES = int(os.environ.get("LEDUC_EPISODES", "50"))
RANDOM_OPEN = 2

KUHN_OUT = "runs/kuhn_poker"
LEDUC_OUT = "runs/new_games_experiment/leduc_poker"
RUNS_BASE = "runs"


def acfg(name: str, base: str, temp: float, timeout: float) -> dict:
    return {
        "type": "model", "name": name, "coached": True,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{base}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": temp, "max_tokens": MAX_TOKENS,
            "timeout_s": timeout,
        },
        "max_retries": MAX_RETRIES,
    }


def step_tracker(path: str, pair: str):
    def on_step(info):
        rec = {"t": round(time.time(), 1), "pair": pair, "ep": info["episode"],
               "step": info["step"], "agent": info["agent_name"],
               "action": (info.get("action") or "")[:40]}
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass
    return on_step


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = 0

    coached = [m + "-coached" for m in BOARD_ORDER]
    board_pairs = list(itertools.combinations(coached, 2))      # 6 (3 incumbent + 3 qwen)
    leduc_pairs = list(itertools.combinations(LEDUC_ORDER, 2))  # 6 (bare)
    total = len(board_pairs) * 3 + len(leduc_pairs)            # kuhn+connect4+gomoku + leduc
    print(f"qwen eval-2: kuhn+connect4+gomoku {len(board_pairs)} pairs each + leduc "
          f"{len(leduc_pairs)} pairs, ONE shared semaphore cap {MAX_CONCURRENCY}, "
          f"per-episode resume on\n", flush=True)

    async def play(game, gdir, a, b, ba, bb, episodes, temp, timeout, seed, steps_path, gmaker):
        nonlocal done
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(gmaker, on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, ba, temp, timeout), game_name=game),
                    make_agent(acfg(b, bb, temp, timeout), game_name=game),
                    episodes=episodes, seed=seed, seat_swap=True,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(steps_path, f"{a}__vs__{b}"))
            done += 1
            drop = f"  DROPPED {res.failures}/{episodes}" if res.failures else ""
            print(f"[{done}/{total}] {game} {a} vs {b}: {len(res.episodes)}/{episodes}{drop}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] {game} {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    tasks = []
    # FAST card games FIRST (kuhn, leduc) so they win the semaphore race and
    # finish quickly, THEN the slow board games — otherwise the long board
    # episodes monopolize the 64 slots and FIFO-starve the short card episodes.
    # kuhn (temp 0.0, coached)
    for pi, (a, b) in enumerate(board_pairs):
        ba, bb = a[:-len("-coached")], b[:-len("-coached")]
        tasks.append(play("kuhn_poker", os.path.join(KUHN_OUT, f"{a}__vs__{b}"),
                          a, b, ba, bb, KUHN_EPISODES, 0.0, MODEL_TIMEOUT_S,
                          9000 + pi, os.path.join(KUHN_OUT, "steps.jsonl"),
                          lambda: make_game("kuhn_poker")))
    # leduc (temp 0.6, bare labels, leduc order)
    for pi, (a, b) in enumerate(leduc_pairs):
        tasks.append(play("leduc_poker", os.path.join(LEDUC_OUT, f"{a}__vs__{b}"),
                          a, b, a, b, LEDUC_EPISODES, 0.6, MODEL_TIMEOUT_S,
                          9000 + pi, os.path.join(LEDUC_OUT, "steps.jsonl"),
                          lambda: make_game("leduc_poker")))
    # board games last (connect4, gomoku — temp 0.6)
    for gi, game in enumerate(["connect4", "gomoku"]):
        for pi, (a, b) in enumerate(board_pairs):
            ba, bb = a[:-len("-coached")], b[:-len("-coached")]
            gdir = os.path.join(RUNS_BASE, game, f"{game}__{a}__vs__{b}")
            tasks.append(play(game, gdir, a, b, ba, bb, BOARD_EPISODES, 0.6, BOARD_TIMEOUT,
                              5000 + gi * 100 + pi, os.path.join(RUNS_BASE, game, "steps.jsonl"),
                              (lambda g=game: make_game(g, {"random_open": RANDOM_OPEN}))))

    await asyncio.gather(*tasks)
    print(f"\nQWEN EVAL2 DONE (kuhn + connect4 + gomoku + leduc) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
