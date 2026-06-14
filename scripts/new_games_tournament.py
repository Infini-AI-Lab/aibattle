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
  PYTHONPATH=src python scripts/new_games_tournament.py [--episodes N] [--games g1,g2]
Environment:
  EPISODES            override episodes per pair / per model (default small for cost)
  BLACKJACK_EPISODES  per-model blackjack hand count (default: EPISODES)
  OTHELLO_EPISODES    per-pair othello episode count (default: EPISODES)
  PARALLEL_GAMES      1 = run all selected games concurrently under the one
                      shared MAX_CONCURRENCY semaphore (default: sequential)
  OUT                 output root (default runs/new_games_experiment)
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

if "FIREWORKS_API_KEY" not in os.environ and os.path.exists(".fireworks"):
    with open(".fireworks", encoding="utf-8") as fh:
        os.environ["FIREWORKS_API_KEY"] = fh.read().strip()

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

# Default Fireworks models (DEC-1 era). Override with a comma-separated MODELS
# env var; verify availability first (e.g. minimax-m2p7 became available later,
# deepseek-flash is still not in this account).
MODELS = [m for m in os.environ.get(
    "MODELS", "kimi-k2p6,deepseek-v4-pro,glm-5p1,gpt-oss-120b").split(",") if m]

VERSUS_GAMES = ["othello_lite_6x6", "leduc_poker", "repeated_colonel_blotto"]
ENV_GAMES = ["independent_blackjack"]

OUT = os.environ.get("OUT", "runs/new_games_experiment")
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "128"))
# Model call timeout. Long games (Blotto/Othello) with slow reasoning models can
# legitimately take a while per step, so this defaults high and is env-tunable.
# (A previous hard-coded 120s could fail expected-slow long-game decisions.)
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "900"))
# Per-call output token cap (128k default so long reasoning is never truncated).
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "131072"))
# Inject the per-game coaching line into the prompt template (COACHED=0 disables).
COACHED = os.environ.get("COACHED", "1").lower() in ("1", "true", "yes")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
MAX_RETRIES = 2


def acfg(label: str) -> dict:
    return {
        "type": "model", "name": label, "coached": COACHED,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{label}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "timeout_s": MODEL_TIMEOUT_S,
        },
        "max_retries": MAX_RETRIES,
    }


def run_settings() -> dict:
    """Every knob that shaped model behavior, stamped into each game's data.json
    so a stored run is self-describing (COACHED in particular is otherwise
    unrecoverable after the fact)."""
    return {
        "coached": COACHED,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "model_timeout_s": MODEL_TIMEOUT_S,
        "max_retries": MAX_RETRIES,
        "max_concurrency": MAX_CONCURRENCY,
        "provider": "fireworks",
        "on_invalid_action": "fallback",
    }


def dealer_cfg() -> dict:
    return {"type": "builtin", "name": "blackjack_dealer"}


def make_step_tracker(game: str, pair: str):
    """A Runner on_step callback that appends one compact JSONL line per
    completed decision to OUT/<game>/steps.jsonl. Pure observability (no model
    behavior change): lets a monitor report in-game positions of in-flight
    episodes, e.g. which round each Blotto episode has reached."""
    path = os.path.join(OUT, game, "steps.jsonl")
    def on_step(info):
        rec = {"t": round(time.time(), 1), "pair": pair, "ep": info["episode"],
               "step": info["step"], "agent": info["agent_name"],
               "action": (info.get("action") or "")[:40]}
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass  # progress telemetry must never kill an episode
    return on_step


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
            "structure": "round_robin_seat_swap", "settings": run_settings(),
            "pairs": [], "leaderboard": []}

    def save():
        # Refresh the leaderboard from whatever pairs have completed so far, so an
        # interruption still leaves a truthful partial state on disk.
        data["leaderboard"] = _aggregate_versus(data["pairs"])
        json.dump(data, open(os.path.join(out, "data.json"), "w"))

    # Write the skeleton BEFORE any pair runs, so a game that is interrupted
    # before its first completion still has a visible 0/6 PARTIAL data.json.
    save()

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
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=make_step_tracker(game, f"{a}__vs__{b}"))
            data["pairs"].append({"a": a, "b": b, "seed": seed,
                                  "episodes": res.episodes})
            save()
            print(f"  [{game}] {a} vs {b}: {len(res.episodes)}/{episodes} hands",
                  flush=True)
        except Exception as ex:
            print(f"  [{game}] {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    specs = [(a, b, 9000 + i) for i, (a, b) in enumerate(pairs)]
    # Optionally run pairs involving a fast model first (e.g. gpt-oss-120b), so a
    # serial run stores at least one completed round-robin episode quickly even
    # when the other models are slow. Pure ordering — every pair still runs.
    fast = os.environ.get("FAST_FIRST_MODEL", "")
    if fast:
        specs.sort(key=lambda t: 0 if fast in (t[0], t[1]) else 1)
    if os.environ.get("SERIAL_PAIRS", "").lower() in ("1", "true", "yes"):
        # True serial: one model pair at a time. Concentrates throughput on a
        # single match (fewer concurrent slow reasoning calls) so progress is
        # visible and one stalled model can't block the whole batch.
        for a, b, s in specs:
            await play(a, b, s)
            save()
    else:
        await asyncio.gather(*(play(a, b, s) for a, b, s in specs))
    save()  # save() refreshes the leaderboard from completed pairs
    return data


async def run_blackjack(episodes: int, sem) -> dict:
    game = "independent_blackjack"
    out = os.path.join(OUT, game)
    os.makedirs(out, exist_ok=True)
    data = {"game": game, "models": MODELS, "episodes_per_model": episodes,
            "structure": "independent_vs_dealer", "settings": run_settings(),
            "model_runs": {}}

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
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=make_step_tracker(game, f"{m}__vs__dealer"))
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
            "structure": f"model_vs_{baseline}", "settings": run_settings(),
            "model_runs": {}}
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
                    logger=lg, semaphore=sem, episode_dir=gdir,
                    on_step=make_step_tracker(game, f"{m}__vs__{baseline}"))
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


# The AC-8 expected structure per game (so missing games are reported, not hidden).
EXPECTED_STRUCTURE = {
    "othello_lite_6x6": "round_robin_seat_swap",
    "leduc_poker": "round_robin_seat_swap",
    "repeated_colonel_blotto": "round_robin_seat_swap",
    "independent_blackjack": "independent_vs_dealer",
}


def _seat_directions(pair: dict) -> set:
    """Return which model seat directions appear in a pair result."""
    a = pair.get("a")
    b = pair.get("b")
    dirs = set()
    for ep in pair.get("episodes", []):
        seat = ep.get("seat_assignment", {})
        if seat.get("player_0") == a and seat.get("player_1") == b:
            dirs.add("a_as_player_0")
        if seat.get("player_0") == b and seat.get("player_1") == a:
            dirs.add("b_as_player_0")
    return dirs


def _coverage(game: str, data: dict) -> dict:
    """Expected-vs-actual coverage for a game, so PARTIAL/missing != COMPLETE.

    For round-robin games, COMPLETE requires every model pair to have both seat
    directions represented. A single episode with seat_swap=True is still only
    one direction because Runner interprets episodes as the total episode budget.
    """
    n_pairs = len(list(itertools.combinations(MODELS, 2)))  # 6 for four models
    n_models = len(MODELS)
    structure = data.get("structure") or EXPECTED_STRUCTURE.get(game, "")
    models_present = len(data.get("leaderboard", []))
    if EXPECTED_STRUCTURE.get(game) == "round_robin_seat_swap":
        pairs = data.get("pairs", [])
        pairs_done = len(pairs)
        seat_swapped = sum(1 for pair in pairs if len(_seat_directions(pair)) >= 2)
        complete = (pairs_done >= n_pairs and seat_swapped >= n_pairs
                    and models_present >= n_models)
        text = (f"{pairs_done}/{n_pairs} model pairs, "
                f"{seat_swapped}/{n_pairs} seat-swapped pairs, "
                f"{models_present}/{n_models} models")
        cov = {
            "expected_pairs": n_pairs,
            "pairs_done": pairs_done,
            "seat_swapped_pairs": seat_swapped,
        }
    else:  # model-vs-dealer (one run per model)
        runs_done = len(data.get("model_runs", {})) or models_present
        complete = runs_done >= n_models
        text = f"{runs_done}/{n_models} models"
        cov = {"expected_models": n_models, "models_done": runs_done}
    cov.update({"structure": structure, "status": "COMPLETE" if complete else "PARTIAL",
                "text": text})
    return cov


def write_report(all_data: dict):
    # Merge in any games stored on disk from prior runs.
    merged = dict(_load_all_stored())
    merged.update(all_data)
    # Seed EVERY expected game so a game with no data still appears as PARTIAL
    # (rather than vanishing from the report and looking complete).
    for game in VERSUS_GAMES + ENV_GAMES:
        merged.setdefault(game, {"game": game, "structure": EXPECTED_STRUCTURE.get(game),
                                 "leaderboard": [], "pairs": [], "model_runs": {}})
    all_data = merged
    os.makedirs(REPORT_DIR, exist_ok=True)

    lines = ["# AI Battle Arena — New Games Four-Model Experiment", ""]
    lines.append(f"Models: {', '.join(MODELS)}  ")
    absent = [m for m in ("minimax-m2p7", "deepseek-flash") if m not in MODELS]
    if absent:
        lines.append(f"Unavailable ids ({', '.join(absent)}) are out of scope.")
    lines.append("")
    json_out = {}
    # Report games in a stable, plan-aligned order.
    for game in ("independent_blackjack", "leduc_poker",
                 "repeated_colonel_blotto", "othello_lite_6x6"):
        data = all_data[game]
        cov = _coverage(game, data)
        lb = data.get("leaderboard", [])
        lines.append(f"## {game}")
        lines.append(f"_coverage: {cov['text']} — {cov['status']} "
                     f"(structure: {cov['structure'] or 'n/a'})_")
        if not lb:
            lines.append("_(no results yet)_\n")
        else:
            cols = list(lb[0].keys())
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
            for r in lb:
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
            lines.append("")
        json_out[game] = {"coverage": cov, "leaderboard": lb}

    md = "\n".join(lines)
    for p in (os.path.join(OUT, "report.md"),
              os.path.join(REPORT_DIR, "new_games_leaderboard.md")):
        open(p, "w", encoding="utf-8").write(md)
    json.dump(json_out,
              open(os.path.join(REPORT_DIR, "new_games_leaderboard.json"), "w"),
              indent=2)
    print(f"\nReport written to {REPORT_DIR}/new_games_leaderboard.md")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int,
                    default=int(os.environ.get("EPISODES", "20")))
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
    # OTHELLO_EPISODES / BLACKJACK_EPISODES override the per-game episode count.
    fast_first = ENV_GAMES + ["leduc_poker", "repeated_colonel_blotto",
                              "othello_lite_6x6"]
    othello_episodes = int(os.environ.get("OTHELLO_EPISODES", str(episodes)))
    blackjack_episodes = int(os.environ.get("BLACKJACK_EPISODES", str(episodes)))
    # BASELINE_MODE=1 evaluates the long games (Blotto/Othello) as model-vs-local-
    # baseline instead of model-vs-model, so only one seat calls the API and the
    # games actually finish. Optionally restrict to specific games via a CSV in
    # BASELINE_GAMES (default: both long games).
    baseline_mode = os.environ.get("BASELINE_MODE", "").lower() in ("1", "true", "yes")
    baseline_games = set(
        os.environ.get("BASELINE_GAMES",
                       "repeated_colonel_blotto,othello_lite_6x6").split(","))
    # PARALLEL_GAMES=1 runs ALL selected games concurrently under the single
    # shared semaphore, so MAX_CONCURRENCY is a true global model-call budget
    # (the per-game ordering above only matters for the sequential mode).
    parallel_games = os.environ.get("PARALLEL_GAMES", "").lower() in ("1", "true", "yes")

    async def run_one(game: str):
        n = othello_episodes if game == "othello_lite_6x6" else episodes
        if game in ENV_GAMES:
            print(f"== {game} (independent vs dealer, "
                  f"{blackjack_episodes} hands/model) ==", flush=True)
            all_data[game] = await run_blackjack(blackjack_episodes, sem)
        elif baseline_mode and game in baseline_games:
            print(f"== {game} (model vs baseline, {n} games/model) ==", flush=True)
            all_data[game] = await run_versus_baseline(game, n, sem)
        else:
            print(f"== {game} (round-robin, {n} hands/pair) ==", flush=True)
            all_data[game] = await run_versus_game(game, n, sem)
        write_report(all_data)   # incremental report so partial progress is saved

    selected = [g for g in fast_first if not want or g in want]
    if parallel_games:
        await asyncio.gather(*(run_one(g) for g in selected))
    else:
        for game in selected:
            await run_one(game)

    print(f"\nEXPERIMENT DONE in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
