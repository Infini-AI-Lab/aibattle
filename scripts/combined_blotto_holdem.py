"""Combined recovery runner: Colonel Blotto + Hold'em match under ONE shared
concurrency budget.

The two games normally run from separate scripts, each with its own
asyncio.Semaphore(MAX_CONCURRENCY). Running them back-to-back wastes the idle
slots at each stage's tail. This runner launches every pair of BOTH games against
a single shared semaphore, so the 64-call budget flows to whichever game has work
— total Fireworks concurrency stays <=64 the whole time, and no slot sits idle
while the other game still has episodes to play.

Faithful to the original scripts so per-episode resume reuses existing files:
- Blotto   : dirs runs/new_games_experiment/repeated_colonel_blotto/<a>__vs__<b>
             with BARE model labels, seat_swap=True, seed=9000+pair_index over
             combinations(NG_MODELS), 20 eps/pair. (NG_MODELS order fixes seeds.)
- Hold'em  : dirs runs/holdem_match/<a>-coached__vs__<b>-coached, seat_swap=False,
             no seed (random deals), 40 matches/pair (<=30 hands each).

Settings are identical to run_m3_eval.sh / run_m3_recovery.sh: coached, temp 0.6,
max_tokens 131072, 900s timeout, 2 retries. Concurrency is the single shared cap.
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
HOLDEM_TIMEOUT = float(os.environ.get("HOLDEM_TIMEOUT", "900"))
MAX_RETRIES = 2

# Same orders the original scripts used, so combinations() -> identical pair dirs
# (and, for blotto, identical seeds) -> per-episode resume reuses everything.
NG_MODELS = os.environ.get(
    "NG_MODELS", "kimi-k2p6,deepseek-v4-pro,glm-5p1,minimax-m2p7,gpt-oss-120b,minimax-m3").split(",")
OLD_MODELS = os.environ.get(
    "OLD_MODELS", "deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7,minimax-m3").split(",")

BLOTTO_EPISODES = int(os.environ.get("BLOTTO_EPISODES", "20"))
MATCH_EPISODES = int(os.environ.get("MATCH_EPISODES", "40"))
MAX_HANDS = 30
STARTING_STACK = 200

NG_OUT = os.environ.get("NG_OUT", "runs/new_games_experiment")
MATCH_OUT = os.environ.get("MATCH_OUT", "runs/holdem_match")


def acfg(name: str, base: str, timeout: float, coached: bool) -> dict:
    return {
        "type": "model", "name": name, "coached": coached,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{base}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "timeout_s": timeout,
        },
        "max_retries": MAX_RETRIES,
    }


def blotto_step_tracker(pair: str):
    path = os.path.join(NG_OUT, "repeated_colonel_blotto", "steps.jsonl")
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


def _trim_match(e: dict) -> dict:
    return {k: e[k] for k in ("episode", "seat_assignment", "returns", "winner",
                              "winner_name", "length", "hands_played",
                              "final_stacks", "stack_diff", "reason",
                              "hand_summaries") if k in e}


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    done = 0

    blotto_pairs = list(itertools.combinations(NG_MODELS, 2))   # bare labels
    match_pairs = list(itertools.combinations(OLD_MODELS, 2))   # -> coached dirs
    total = len(blotto_pairs) + len(match_pairs)
    print(f"Combined runner: {len(blotto_pairs)} blotto + {len(match_pairs)} "
          f"holdem_match pairs, ONE shared semaphore cap {MAX_CONCURRENCY}, "
          f"per-episode resume on\n", flush=True)

    blotto_results = []
    match_results = []

    async def play_blotto(a, b, seed):
        nonlocal done
        out = os.path.join(NG_OUT, "repeated_colonel_blotto")
        gdir = os.path.join(out, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game("repeated_colonel_blotto"),
                        on_invalid_action="fallback")
        ta = time.perf_counter()
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, a, MODEL_TIMEOUT_S, True), game_name="repeated_colonel_blotto"),
                    make_agent(acfg(b, b, MODEL_TIMEOUT_S, True), game_name="repeated_colonel_blotto"),
                    episodes=BLOTTO_EPISODES, seed=seed, seat_swap=True,
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=blotto_step_tracker(f"{a}__vs__{b}"))
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

    async def play_match(a, b):  # a,b already '<name>-coached'
        nonlocal done
        gdir = os.path.join(MATCH_OUT, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        base_a, base_b = a[:-len("-coached")], b[:-len("-coached")]
        runner = Runner(match_factory, on_invalid_action="fallback")
        ta = time.perf_counter()
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a, base_a, HOLDEM_TIMEOUT, True), game_name="holdem_match"),
                    make_agent(acfg(b, base_b, HOLDEM_TIMEOUT, True), game_name="holdem_match"),
                    episodes=MATCH_EPISODES, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir)
            match_results.append({"a": a, "b": b, "episodes": [_trim_match(e) for e in res.episodes]})
            done += 1
            drop = f"  DROPPED {res.failures}/{MATCH_EPISODES}" if res.failures else ""
            print(f"[{done}/{total}] match {a} vs {b}: {len(res.episodes)}/{MATCH_EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] match {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    tasks = [play_blotto(a, b, 9000 + i) for i, (a, b) in enumerate(blotto_pairs)]
    tasks += [play_match(a, b) for a, b in match_pairs]
    await asyncio.gather(*tasks)

    # Refresh each game's data.json aggregate from the (complete) results.
    _save_blotto(blotto_results)
    _save_match(match_results)
    print(f"\nM3 RECOVERY DONE (combined blotto+holdem_match) in {time.perf_counter()-t0:.0f}s", flush=True)


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
