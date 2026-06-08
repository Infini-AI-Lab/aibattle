"""Four-model Fireworks comparison across the four new games.

Games and structure (per the implementation plan, Milestone E):
- Agent-vs-agent (othello_lite_6x6, leduc_poker, repeated_colonel_blotto): a
  round-robin over the four verified Fireworks models (C(4,2)=6 pairs), seat-swap
  on, per-episode resume.
- Agent-vs-environment (independent_blackjack): each model independently plays as
  player_0 against the built-in blackjack_dealer, seat_swap off; reported with the
  dedicated player-seat-only analysis (the dealer is never ranked).

Only the four verified-available model ids are used; the unavailable
``minimax-m2p7`` / ``deepseek-flash`` are intentionally absent.

Per-episode resume is on (``episode_dir``), so the experiment can be re-run to
continue after an interruption. Results are stored under ``runs/<exp>/`` and an
aggregated report is written under ``reports/``.

Usage:
  PYTHONPATH=src python scripts/new_games_experiment.py [--episodes N] [--games g1,g2]
Environment:
  EPISODES   override episodes per pair / per model (default small for cost)
  OUT        output root (default runs/new_games_experiment)
"""

from __future__ import annotations

import argparse
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

# The four verified-available Fireworks models (DEC-1). Do NOT add minimax-m2p7
# or deepseek-flash — they are not in this account.
MODELS = ["kimi-k2p6", "deepseek-v4-pro", "glm-5p1", "gpt-oss-120b"]

VERSUS_GAMES = ["othello_lite_6x6", "leduc_poker", "repeated_colonel_blotto"]
ENV_GAMES = ["independent_blackjack"]

OUT = os.environ.get("OUT", "runs/new_games_experiment")
REPORT_DIR = "reports"
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))


def acfg(label: str) -> dict:
    return {
        "type": "model", "name": label,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{label}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": 0.0, "max_tokens": 16384, "timeout_s": 120,
        },
        "max_retries": 2,
    }


def dealer_cfg() -> dict:
    return {"type": "builtin", "name": "blackjack_dealer"}


def _aggregate_versus(pairs_data: list) -> list:
    hands = defaultdict(int); wins = defaultdict(int); net = defaultdict(float)
    decisions = defaultdict(int); invalid = defaultdict(int)
    for g in pairs_data:
        for e in g["episodes"]:
            seat = e["seat_assignment"]
            wname = e.get("winner_name")
            for p in ("player_0", "player_1"):
                nm = seat[p]
                hands[nm] += 1
                net[nm] += e["returns"][p]
            if wname:
                wins[wname] += 1
            for s in e.get("steps", []):
                nm = s.get("agent_name") or seat.get(s.get("player"))
                decisions[nm] += 1
                if s.get("invalid"):
                    invalid[nm] += 1
    rows = []
    for m in set(hands):
        h = hands[m] or 1; d = decisions[m] or 1
        rows.append({"model": m, "games": hands[m],
                     "win_rate": round(wins[m] / h, 3),
                     "net_per_game": round(net[m] / h, 3),
                     "invalid_rate": round(invalid[m] / d, 4)})
    rows.sort(key=lambda r: r["net_per_game"], reverse=True)
    return rows


def _aggregate_blackjack(model_runs: dict) -> list:
    """model -> list of episodes (player_0 == the model)."""
    rows = []
    for m, eps in model_runs.items():
        hands = len(eps) or 1
        profit = sum(e["returns"]["player_0"] for e in eps)
        wins = sum(1 for e in eps if e["returns"]["player_0"] > 0)
        losses = sum(1 for e in eps if e["returns"]["player_0"] < 0)
        pushes = sum(1 for e in eps if e["returns"]["player_0"] == 0)
        decisions = sum(1 for e in eps for s in e.get("steps", [])
                        if s.get("player") == "player_0")
        invalid = sum(1 for e in eps for s in e.get("steps", [])
                      if s.get("player") == "player_0" and s.get("invalid"))
        rows.append({"model": m, "hands": len(eps),
                     "profit": round(profit, 2),
                     "mean_per_hand": round(profit / hands, 3),
                     "win_rate": round(wins / hands, 3),
                     "loss_rate": round(losses / hands, 3),
                     "push_rate": round(pushes / hands, 3),
                     "invalid_rate": round(invalid / max(1, decisions), 4)})
    rows.sort(key=lambda r: r["mean_per_hand"], reverse=True)
    return rows


async def run_versus_game(game: str, episodes: int, sem) -> dict:
    out = os.path.join(OUT, game)
    os.makedirs(out, exist_ok=True)
    pairs = list(itertools.combinations(MODELS, 2))
    data = {"game": game, "models": MODELS, "episodes_per_pair": episodes,
            "structure": "round_robin_seat_swap", "pairs": []}

    def save():
        json.dump(data, open(os.path.join(out, "data.json"), "w"))

    async def play(a, b, seed):
        gdir = os.path.join(out, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a), game_name=game),
                    make_agent(acfg(b), game_name=game),
                    episodes=episodes, seed=seed, seat_swap=True,
                    logger=lg, semaphore=sem, episode_dir=gdir)
            data["pairs"].append({"a": a, "b": b, "seed": seed,
                                  "episodes": res.episodes})
            save()
            print(f"  [{game}] {a} vs {b}: {len(res.episodes)}/{episodes} hands",
                  flush=True)
        except Exception as ex:
            print(f"  [{game}] {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    specs = [(a, b, 9000 + i) for i, (a, b) in enumerate(pairs)]
    if os.environ.get("SERIAL_PAIRS", "").lower() in ("1", "true", "yes"):
        # True serial: one model pair at a time. Concentrates throughput on a
        # single match (fewer concurrent slow reasoning calls) so progress is
        # visible and one stalled model can't block the whole batch.
        for a, b, s in specs:
            await play(a, b, s)
            save()
    else:
        await asyncio.gather(*(play(a, b, s) for a, b, s in specs))
    save()
    rows = _aggregate_versus(data["pairs"])
    data["leaderboard"] = rows
    save()
    return data


async def run_blackjack(episodes: int, sem) -> dict:
    game = "independent_blackjack"
    out = os.path.join(OUT, game)
    os.makedirs(out, exist_ok=True)
    data = {"game": game, "models": MODELS, "episodes_per_model": episodes,
            "structure": "independent_vs_dealer", "model_runs": {}}

    def save():
        json.dump(data, open(os.path.join(out, "data.json"), "w"))

    model_runs = {}

    async def play(m, seed):
        gdir = os.path.join(out, f"{m}__vs__dealer")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(m), game_name=game),       # player_0 = model
                    make_agent(dealer_cfg(), game_name=game),  # player_1 = dealer
                    episodes=episodes, seed=seed, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir)
            model_runs[m] = res.episodes
            data["model_runs"][m] = {"hands": len(res.episodes)}
            save()
            print(f"  [blackjack] {m} vs dealer: {len(res.episodes)}/{episodes} hands",
                  flush=True)
        except Exception as ex:
            print(f"  [blackjack] {m} FAILED: {ex}", flush=True)
            traceback.print_exc()

    await asyncio.gather(*(play(m, 9100 + i) for i, m in enumerate(MODELS)))
    rows = _aggregate_blackjack(model_runs)
    data["leaderboard"] = rows
    save()
    return data


# A fast local baseline per game, used for the model-vs-baseline mode (only the
# model seat calls the API, so long games like Blotto/Othello can actually
# finish). These are the game's uniform-random builtins.
_BASELINE = {
    "othello_lite_6x6": "board_random",
    "repeated_colonel_blotto": "blotto_random",
    "leduc_poker": "leduc_random",
}


def _aggregate_vs_baseline(model_runs: dict) -> list:
    """model -> list of episodes (player_0 == the model, player_1 == baseline)."""
    rows = []
    for m, eps in model_runs.items():
        n = len(eps) or 1
        net = sum(e["returns"]["player_0"] for e in eps)
        wins = sum(1 for e in eps if e["returns"]["player_0"] > 0)
        losses = sum(1 for e in eps if e["returns"]["player_0"] < 0)
        decisions = sum(1 for e in eps for s in e.get("steps", [])
                        if s.get("player") == "player_0")
        invalid = sum(1 for e in eps for s in e.get("steps", [])
                      if s.get("player") == "player_0" and s.get("invalid"))
        rows.append({"model": m, "games": len(eps),
                     "win_rate": round(wins / n, 3),
                     "loss_rate": round(losses / n, 3),
                     "net_per_game": round(net / n, 3),
                     "invalid_rate": round(invalid / max(1, decisions), 4)})
    rows.sort(key=lambda r: r["net_per_game"], reverse=True)
    return rows


async def run_versus_baseline(game: str, episodes: int, sem) -> dict:
    """Each model plays as player_0 against the game's fast local baseline
    (player_1). Only the model seat calls the API, so long games complete."""
    out = os.path.join(OUT, game)
    os.makedirs(out, exist_ok=True)
    baseline = _BASELINE[game]
    data = {"game": game, "models": MODELS, "episodes_per_model": episodes,
            "structure": f"model_vs_{baseline}", "model_runs": {}}
    model_runs = {}

    def save():
        json.dump(data, open(os.path.join(out, "data.json"), "w"))

    async def play(m, seed):
        gdir = os.path.join(out, f"{m}__vs__{baseline}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(m), game_name=game),               # player_0 = model
                    make_agent({"type": "builtin", "name": baseline},
                               game_name=game),                        # player_1 = baseline
                    episodes=episodes, seed=seed, seat_swap=False,
                    logger=lg, semaphore=sem, episode_dir=gdir)
            model_runs[m] = res.episodes
            data["model_runs"][m] = {"games": len(res.episodes)}
            # Update the leaderboard incrementally as each model finishes, so a
            # slow model never prevents the already-finished ones from being
            # stored and reported.
            data["leaderboard"] = _aggregate_vs_baseline(model_runs)
            save()
            print(f"  [{game}] {m} vs {baseline}: {len(res.episodes)}/{episodes}",
                  flush=True)
        except Exception as ex:
            print(f"  [{game}] {m} FAILED: {ex}", flush=True)
            traceback.print_exc()

    # Concurrent across models: each model plays only against the LOCAL baseline
    # (seconds per step), so the four runs are independent — a slow model
    # (e.g. kimi-k2p6) does not block the fast ones from finishing and storing.
    await asyncio.gather(*(play(m, 9200 + i) for i, m in enumerate(MODELS)))
    data["leaderboard"] = _aggregate_vs_baseline(model_runs)
    save()
    return data


def _load_all_stored() -> dict:
    """Read every per-game data.json under OUT so the report reflects all games
    completed across runs (per-episode resume keeps partial progress)."""
    stored = {}
    for game in VERSUS_GAMES + ENV_GAMES:
        f = os.path.join(OUT, game, "data.json")
        if os.path.exists(f):
            try:
                stored[game] = json.load(open(f))
            except Exception:
                pass
    return stored


def write_report(all_data: dict):
    # Merge in any games stored on disk from prior runs, then de-dupe by game.
    merged = dict(_load_all_stored())
    merged.update(all_data)
    all_data = merged
    os.makedirs(REPORT_DIR, exist_ok=True)
    lines = ["# AI Battle Arena — New Games Four-Model Experiment", ""]
    lines.append(f"Models: {', '.join(MODELS)}  ")
    lines.append("Unavailable ids (minimax-m2p7, deepseek-flash) are out of scope.")
    lines.append("")
    for game, data in all_data.items():
        lines.append(f"## {game}")
        lb = data.get("leaderboard", [])
        if not lb:
            lines.append("_(no results)_\n")
            continue
        cols = list(lb[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in lb:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        lines.append("")
    md = "\n".join(lines)
    for p in (os.path.join(OUT, "report.md"),
              os.path.join(REPORT_DIR, "new_games_experiment_report.md")):
        open(p, "w", encoding="utf-8").write(md)
    json.dump({g: d.get("leaderboard", []) for g, d in all_data.items()},
              open(os.path.join(REPORT_DIR, "new_games_experiment.json"), "w"),
              indent=2)
    print(f"\nReport written to {REPORT_DIR}/new_games_experiment_report.md")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int,
                    default=int(os.environ.get("EPISODES", "6")))
    ap.add_argument("--games", type=str, default="")
    args = ap.parse_args()
    episodes = args.episodes
    want = set(args.games.split(",")) if args.games else None

    os.makedirs(OUT, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    all_data = {}

    # Blackjack and Leduc finish quickest (short hands); Blotto is medium; Othello
    # games are long (~30 plies) with slow reasoning models, so run it LAST. Every
    # game uses per-episode resume, so a later re-run continues where it left off.
    # OTHELLO_EPISODES lets Othello use a smaller count so the run completes.
    fast_first = ENV_GAMES + ["leduc_poker", "repeated_colonel_blotto",
                              "othello_lite_6x6"]
    othello_episodes = int(os.environ.get("OTHELLO_EPISODES", str(episodes)))
    # BASELINE_MODE=1 evaluates the long games (Blotto/Othello) as model-vs-local-
    # baseline instead of model-vs-model, so only one seat calls the API and the
    # games actually finish. Optionally restrict to specific games via a CSV in
    # BASELINE_GAMES (default: both long games).
    baseline_mode = os.environ.get("BASELINE_MODE", "").lower() in ("1", "true", "yes")
    baseline_games = set(
        os.environ.get("BASELINE_GAMES",
                       "repeated_colonel_blotto,othello_lite_6x6").split(","))

    for game in fast_first:
        if want and game not in want:
            continue
        if game in ENV_GAMES:
            print(f"== {game} (independent vs dealer, {episodes} hands/model) ==",
                  flush=True)
            all_data[game] = await run_blackjack(episodes, sem)
        elif baseline_mode and game in baseline_games:
            n = othello_episodes if game == "othello_lite_6x6" else episodes
            print(f"== {game} (model vs baseline, {n} games/model) ==", flush=True)
            all_data[game] = await run_versus_baseline(game, n, sem)
        else:
            n = othello_episodes if game == "othello_lite_6x6" else episodes
            print(f"== {game} (round-robin, {n} hands/pair) ==", flush=True)
            all_data[game] = await run_versus_game(game, n, sem)
        write_report(all_data)   # incremental report so partial progress is saved

    print(f"\nEXPERIMENT DONE in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
