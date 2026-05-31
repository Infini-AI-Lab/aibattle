"""Thin CLI wrapper around the YAML config.

  aibattle run config.yaml     # run a match, write logs + summary
  aibattle eval <run_dir>      # (re)compute summary from an existing log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

from .agents.registry import make_agent
from .config.loader import load_config
from .eval.evaluator import evaluate, format_summary
from .games.registry import make_game
from .logging.logger import MatchLogger
from .logging.transcript import render_transcript
from .runner.runner import Runner


def _progress_bar(label: str, width: int = 30) -> "callable":
    """Return a progress callback that renders an in-place bar to stdout."""
    def cb(done: int, total: int) -> None:
        frac = (done / total) if total else 1.0
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{label} [{bar}] {done}/{total}", end="", flush=True)
        if done >= total:
            print()  # finish the line
    return cb


def _build_game_factory(game_cfg):
    params = dict(game_cfg.params)

    def factory():
        return make_game(game_cfg.name, params)

    return factory


def _make_run_dir(base_dir: str) -> str:
    """Create a unique per-run subdirectory so concurrent/repeated runs can
    never write to the same files (which would corrupt each other's logs).

    Layout: <base_dir>/run_<timestamp>_<rand>/
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{stamp}_{uuid.uuid4().hex[:6]}")
    # exist_ok=False: the random suffix guarantees uniqueness; a collision
    # would indicate a real problem, so fail loudly rather than overwrite.
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


async def _run(config_path: str, rerun: bool = False) -> int:
    cfg = load_config(config_path)

    # Skip re-running if a prior run already exists for this output dir, unless
    # --rerun is passed. This avoids wasting (paid) model calls on a repeat.
    existing = _resolve_log(cfg.output.dir)
    if existing and not rerun:
        print(f"A run already exists for this config: {existing}")
        print("Pass --rerun to run it again. Showing the existing summary:\n")
        summary = evaluate(existing, progress=_progress_bar("Evaluating"))
        print()
        print(format_summary(summary))
        return 0

    run_dir = _make_run_dir(cfg.output.dir)

    # Derive distinct, reproducible seeds for builtin agents.
    agent_a = make_agent(cfg.players["player_0"], game_name=cfg.game.name,
                         seed=cfg.run.seed + 1)
    agent_b = make_agent(cfg.players["player_1"], game_name=cfg.game.name,
                         seed=cfg.run.seed + 2)

    log_path = (os.path.join(run_dir, "match.jsonl")
                if cfg.output.save_full_log else None)
    runner = Runner(_build_game_factory(cfg.game),
                    on_invalid_action=cfg.run.on_invalid_action)

    def _progress(done, total, result):
        winner = result.get("winner_name") or "tie"
        seats = "/".join(result["seat_assignment"][p] for p in ("player_0", "player_1"))
        print(f"  [{done:>{len(str(total))}}/{total}] {seats}  ->  "
              f"winner: {winner}  (len {result['length']})", flush=True)

    mc = cfg.run.max_concurrency
    mode = (f"parallel (up to {mc} episodes at once)" if mc > 1 else "sequential")
    print(f"Run dir: {run_dir}")
    print(f"Matchup: {agent_a.name} vs {agent_b.name}")
    print(f"Episodes: {cfg.run.episodes}  |  Execution: {mode}  |  "
          f"seat_swap: {cfg.run.seat_swap}  |  seed: {cfg.run.seed}", flush=True)
    with MatchLogger(log_path) as logger:
        result = await runner.run_match(
            agent_a, agent_b,
            episodes=cfg.run.episodes,
            seed=cfg.run.seed,
            seat_swap=cfg.run.seat_swap,
            logger=logger,
            max_concurrency=cfg.run.max_concurrency,
            progress=_progress,
        )

    print(f"Ran {len(result.episodes)} episodes.")

    # Combined trajectories JSON: all episodes (with nested steps) in one file.
    if cfg.output.save_trajectories:
        traj_path = os.path.join(run_dir, cfg.output.trajectories_file)
        with open(traj_path, "w", encoding="utf-8") as fh:
            json.dump({
                "game": cfg.game.name,
                "episodes": result.episodes,
            }, fh, indent=2)
        print(f"Trajectories: {traj_path}")

    # Per-episode human-readable transcripts (one plain-text file each).
    if cfg.output.save_transcripts:
        tdir = os.path.join(run_dir, cfg.output.transcripts_dir)
        os.makedirs(tdir, exist_ok=True)
        width = max(4, len(str(len(result.episodes) - 1)))
        for traj in result.episodes:
            fname = f"episode_{traj['episode']:0{width}d}.txt"
            with open(os.path.join(tdir, fname), "w", encoding="utf-8") as fh:
                fh.write(render_transcript(traj))
        print(f"Transcripts:  {tdir}/  ({len(result.episodes)} files)")

    if log_path:
        print(f"Match log: {log_path}")
        if cfg.output.save_summary:
            summary = evaluate(log_path, progress=_progress_bar("Evaluating"))
            summary_path = os.path.join(run_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            print(f"Summary:   {summary_path}\n")
            print(format_summary(summary))
    return 0


def _resolve_log(path: str) -> str | None:
    """Resolve a user-supplied path to a match.jsonl file.

    Accepts: a direct .jsonl file; a run dir containing match.jsonl; or a
    parent dir containing run_*/ subdirs (picks the most recent).
    """
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct = os.path.join(path, "match.jsonl")
        if os.path.exists(direct):
            return direct
        # Look for per-run subdirectories and pick the most recently modified.
        candidates = []
        for entry in os.listdir(path):
            sub = os.path.join(path, entry, "match.jsonl")
            if os.path.exists(sub):
                candidates.append(sub)
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None


def _eval(run_dir: str) -> int:
    log_path = _resolve_log(run_dir)
    if log_path is None:
        print(f"No match.jsonl found under {run_dir!r}", file=sys.stderr)
        return 1
    print(f"Evaluating: {log_path}")
    summary = evaluate(log_path, progress=_progress_bar("Evaluating"))
    print()
    print(format_summary(summary))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aibattle", description="AI Battle Arena v0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a match from a YAML config")
    p_run.add_argument("config", help="path to YAML config")
    p_run.add_argument("--rerun", action="store_true",
                       help="run again even if a prior run exists for this output dir")

    p_eval = sub.add_parser("eval", help="summarize an existing match log")
    p_eval.add_argument("run_dir", help="run directory or path to match.jsonl")

    args = parser.parse_args(argv)

    if args.command == "run":
        return asyncio.run(_run(args.config, rerun=args.rerun))
    if args.command == "eval":
        return _eval(args.run_dir)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
