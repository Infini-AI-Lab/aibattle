"""Combined recovery runner for ALL FOUR under-target m3 games at once
(connect4, gomoku, colonel_blotto, holdem_match) under ONE shared semaphore.

Why one process: each original script owns its own asyncio.Semaphore(MAX), so
running them in sequence (or even the board games as a separate stage) leaves the
64-call budget idle at every stage's tail — and worse, the board games' m3
episodes intermittently hang on dead sockets, which would block everything queued
behind them. Launching all four games' pairs against a single shared semaphore
keeps total Fireworks concurrency at <=64 AND lets blotto/holdem fill any slot a
stalled board episode would otherwise waste. No stage can block another.

Faithful to the original scripts so per-episode resume reuses existing files:
- connect4 / gomoku : runs/<game>/<game>__<a>-coached__vs__<b>-coached, seat_swap,
                      seed=5000+gi*15+pi (gi: connect4=0, gomoku=1; pi: pair idx
                      over combinations(OLD coached)), random_open=2, 10 eps.
- blotto            : runs/new_games_experiment/repeated_colonel_blotto/<a>__vs__<b>
                      BARE labels, seat_swap, seed=9000+pair_index over
                      combinations(NG_MODELS), 20 eps.
- holdem_match      : runs/holdem_match/<a>-coached__vs__<b>-coached, no seat_swap,
                      no seed (random deals), 40 matches (<=30 hands each).

Settings identical to run_m3_eval.sh: coached, temp 0.6, max_tokens 131072, 900s
timeout, 2 retries. Concurrency is the single shared cap (default 64).
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
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "900"))
BOARD_TIMEOUT = float(os.environ.get("BOARD_TIMEOUT", "900"))
HOLDEM_TIMEOUT = float(os.environ.get("HOLDEM_TIMEOUT", "900"))
MAX_RETRIES = 2

NG_MODELS = os.environ.get(
    "NG_MODELS", "kimi-k2p6,deepseek-v4-pro,glm-5p1,minimax-m2p7,gpt-oss-120b,minimax-m3").split(",")
OLD_MODELS = os.environ.get(
    "OLD_MODELS", "deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7,minimax-m3").split(",")
OLD_COACHED = [m + "-coached" for m in OLD_MODELS]

BLOTTO_EPISODES = int(os.environ.get("BLOTTO_EPISODES", "20"))
MATCH_EPISODES = int(os.environ.get("MATCH_EPISODES", "40"))
BOARD_EPISODES = int(os.environ.get("BOARD_EPISODES", "10"))
MAX_HANDS = 30
STARTING_STACK = 200
RANDOM_OPEN = 2
BOARD_GAMES = ["connect4", "gomoku"]   # order fixes the 5000+gi*15 seed base

NG_OUT = os.environ.get("NG_OUT", "runs/new_games_experiment")
MATCH_OUT = os.environ.get("MATCH_OUT", "runs/holdem_match")
RUNS_BASE = os.environ.get("RUNS_DIR", "runs")


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


def step_tracker(path: str, pair: str):
    """Append one JSONL line per decision to `path` (live in-flight telemetry).
    Records the hand number from obs.public for hand-based games (holdem) so the
    status table can show 'hand X/30'; pure observability, never blocks a step."""
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


def _trim_steps(episodes):
    out = []
    for e in episodes:
        e2 = dict(e); steps = []
        for s in e.get("steps", []):
            s2 = dict(s); resp = dict(s2.get("response") or {})
            resp.pop("raw_output", None); resp.pop("prompt", None)
            s2["response"] = resp; steps.append(s2)
        e2["steps"] = steps; out.append(e2)
    return out


def _trim_match(e):
    return {k: e[k] for k in ("episode", "seat_assignment", "returns", "winner",
                              "winner_name", "length", "hands_played",
                              "final_stacks", "stack_diff", "reason",
                              "hand_summaries") if k in e}


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = 0

    board_pairs = list(itertools.combinations(OLD_COACHED, 2))   # 15
    blotto_pairs = list(itertools.combinations(NG_MODELS, 2))    # 15 (bare)
    match_pairs = list(itertools.combinations(OLD_COACHED, 2))   # 15
    total = len(BOARD_GAMES) * len(board_pairs) + len(blotto_pairs) + len(match_pairs)
    print(f"Combined recovery: connect4+gomoku ({len(BOARD_GAMES)*len(board_pairs)}) "
          f"+ blotto ({len(blotto_pairs)}) + holdem_match ({len(match_pairs)}) pairs, "
          f"ONE shared semaphore cap {MAX_CONCURRENCY}, per-episode resume on\n", flush=True)

    board_results = {g: [] for g in BOARD_GAMES}
    blotto_results = []
    match_results = []

    async def play_board(game, a, b, seed):
        nonlocal done
        gdir = os.path.join(RUNS_BASE, game, f"{game}__{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        base_a, base_b = a[:-len("-coached")], b[:-len("-coached")]
        runner = Runner(lambda g=game: make_game(g, {"random_open": RANDOM_OPEN}),
                        on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, base_a, BOARD_TIMEOUT), game_name=game),
                    make_agent(acfg(b, base_b, BOARD_TIMEOUT), game_name=game),
                    episodes=BOARD_EPISODES, seed=seed, seat_swap=True,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(os.path.join(RUNS_BASE, game, "steps.jsonl"), f"{a}__vs__{b}"))
            board_results[game].append({"a": a, "b": b, "seed": seed,
                                        "episodes": _trim_steps(res.episodes)})
            done += 1
            drop = f"  DROPPED {res.failures}/{BOARD_EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] {game} {a} vs {b}: {len(res.episodes)}/{BOARD_EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] {game} {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    async def play_blotto(a, b, seed):
        nonlocal done
        gdir = os.path.join(NG_OUT, "repeated_colonel_blotto", f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game("repeated_colonel_blotto"), on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, a, MODEL_TIMEOUT_S), game_name="repeated_colonel_blotto"),
                    make_agent(acfg(b, b, MODEL_TIMEOUT_S), game_name="repeated_colonel_blotto"),
                    episodes=BLOTTO_EPISODES, seed=seed, seat_swap=True,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(os.path.join(NG_OUT, "repeated_colonel_blotto", "steps.jsonl"), f"{a}__vs__{b}"))
            blotto_results.append({"a": a, "b": b, "seed": seed, "episodes": res.episodes})
            done += 1
            drop = f"  DROPPED {res.failures}" if res.failures else ""
            print(f"[{done}/{total}] blotto {a} vs {b}: {len(res.episodes)}/{BLOTTO_EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] blotto {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    def match_factory():
        return make_game("holdem_match", {"starting_stack": STARTING_STACK, "max_hands": MAX_HANDS})

    async def play_match(a, b):
        nonlocal done
        gdir = os.path.join(MATCH_OUT, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        base_a, base_b = a[:-len("-coached")], b[:-len("-coached")]
        runner = Runner(match_factory, on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, base_a, HOLDEM_TIMEOUT), game_name="holdem_match"),
                    make_agent(acfg(b, base_b, HOLDEM_TIMEOUT), game_name="holdem_match"),
                    episodes=MATCH_EPISODES, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=step_tracker(os.path.join(MATCH_OUT, "steps.jsonl"), f"{a}__vs__{b}"))
            match_results.append({"a": a, "b": b, "episodes": [_trim_match(e) for e in res.episodes]})
            done += 1
            drop = f"  DROPPED {res.failures}/{MATCH_EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] match {a} vs {b}: {len(res.episodes)}/{MATCH_EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] match {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    tasks = []
    for gi, game in enumerate(BOARD_GAMES):
        for pi, (a, b) in enumerate(board_pairs):
            tasks.append(play_board(game, a, b, 5000 + gi * len(board_pairs) + pi))
    for i, (a, b) in enumerate(blotto_pairs):
        tasks.append(play_blotto(a, b, 9000 + i))
    for a, b in match_pairs:
        tasks.append(play_match(a, b))

    await asyncio.gather(*tasks)

    _save_board(board_results)
    _save_blotto(blotto_results)
    _save_match(match_results)
    print(f"\nM3 RECOVERY DONE (combined all-4) in {time.perf_counter()-t0:.0f}s", flush=True)


def _save_board(results):
    for game, games_list in results.items():
        path = os.path.join(RUNS_BASE, game, f"{game}_data.json")
        merged = {g["a"] + "\x00" + g["b"]: g for g in games_list}
        if os.path.exists(path):
            try:
                prev = json.load(open(path))
                for g in prev.get("games", []):
                    merged.setdefault(g["a"] + "\x00" + g["b"], g)
            except (json.JSONDecodeError, OSError):
                pass
        out = {"game": game, "episodes_per_pair": BOARD_EPISODES,
               "models": OLD_COACHED, "games": list(merged.values())}
        tmp = path + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, path)


def _save_blotto(pairs):
    out = os.path.join(NG_OUT, "repeated_colonel_blotto")
    hands = defaultdict(int); wins = defaultdict(int); net = defaultdict(float)
    dec = defaultdict(int); inval = defaultdict(int)
    for g in pairs:
        for e in g["episodes"]:
            seat = e["seat_assignment"]
            for p in ("player_0", "player_1"):
                nm = seat[p]; hands[nm] += 1; net[nm] += e["returns"][p]
            if e.get("winner_name"):
                wins[e["winner_name"]] += 1
            for s in e.get("steps", []):
                nm = s.get("agent_name") or seat.get(s.get("player"))
                dec[nm] += 1
                if s.get("invalid"):
                    inval[nm] += 1
    lb = [{"model": m, "games": hands[m], "win_rate": round(wins[m]/(hands[m] or 1), 3),
           "net_per_game": round(net[m]/(hands[m] or 1), 3),
           "invalid_rate": round(inval[m]/(dec[m] or 1), 4)} for m in hands]
    lb.sort(key=lambda r: r["net_per_game"], reverse=True)
    data = {"game": "repeated_colonel_blotto", "models": NG_MODELS,
            "episodes_per_pair": BLOTTO_EPISODES, "structure": "round_robin_seat_swap",
            "pairs": pairs, "leaderboard": lb}
    json.dump(data, open(os.path.join(out, "data.json"), "w"))


def _save_match(pairs):
    data = {"mode": "match", "models": OLD_MODELS, "episodes_per_pair": MATCH_EPISODES,
            "max_hands": MAX_HANDS, "starting_stack": STARTING_STACK, "pairs": pairs}
    json.dump(data, open(os.path.join(MATCH_OUT, "match_data.json"), "w"))


if __name__ == "__main__":
    asyncio.run(main())
