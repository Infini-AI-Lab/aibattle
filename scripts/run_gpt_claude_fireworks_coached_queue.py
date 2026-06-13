"""Run the unfinished coached tournament pairs with a bounded active queue.

This supervisor keeps a limited number of pair-specific tournament workers
running at once. Each worker resumes from the existing per-episode logs, so the
queue can be restarted safely.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT.parent / "aibattle-logs" / "gpt_claude_fireworks_coached_tournament"
RUNNER = REPO_ROOT / "scripts" / "gpt_claude_fireworks_coached_tournament.py"
RUNNER_LOG_DIR = OUT_ROOT / "_runner_logs"

TARGETS = {
    "connect4": 50,
    "gomoku": 50,
    "holdem_match": 20,
}

GAME_PRIORITY = {
    "connect4": 0,
    "gomoku": 1,
    "holdem_match": 2,
}


def _freshness_bucket(age_s: float | None) -> int:
    if age_s is None:
        return 3
    if age_s <= 120:
        return 0
    if age_s <= 300:
        return 1
    if age_s <= 900:
        return 2
    return 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", default="")
    p.add_argument("--active-limit", type=int, default=6)
    p.add_argument("--poll-s", type=float, default=15.0)
    p.add_argument("--stale-seconds", type=float, default=420.0)
    p.add_argument("--no-progress-seconds", type=float, default=300.0)
    p.add_argument("--cooldown-seconds", type=float, default=600.0)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--anthropic-max-tokens", type=int, default=128000)
    p.add_argument("--fireworks-max-tokens", type=int, default=128000)
    p.add_argument("--worker-max-concurrency", type=int, default=1)
    p.add_argument("--pair-batch-size", type=int, default=1)
    p.add_argument("--holdem-match-max-concurrency", type=int, default=None)
    p.add_argument("--holdem-match-pair-batch-size", type=int, default=None)
    p.add_argument(
        "--state-file",
        default="/tmp/gpt_claude_fireworks_queue_state.json",
    )
    return p.parse_args()


def completed_count(pair_dir: Path) -> int:
    return len(
        [
            p for p in pair_dir.glob("ep*.json")
            if not p.name.endswith(".error.json")
        ]
    )


def heartbeat_age_s(game: str, pair: str) -> float | None:
    path = OUT_ROOT / "_heartbeat" / f"{game}_{pair}.log"
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _allowed_games(raw: str) -> set[str] | None:
    keep = {x.strip() for x in raw.split(",") if x.strip()}
    return keep or None


def unfinished_pairs(*, allowed_games: set[str] | None = None) -> list[dict]:
    rows = []
    for game, target in TARGETS.items():
        if allowed_games is not None and game not in allowed_games:
            continue
        game_dir = OUT_ROOT / game
        for pair_dir in sorted(p for p in game_dir.glob("*__vs__*") if p.is_dir()):
            done = completed_count(pair_dir)
            if done >= target:
                continue
            a, b = pair_dir.name.split("__vs__")
            rows.append(
                {
                    "game": game,
                    "pair": pair_dir.name,
                    "models": [a, b],
                    "done": done,
                    "target": target,
                    "remaining": target - done,
                    "heartbeat_age_s": heartbeat_age_s(game, pair_dir.name),
                }
            )
    rows.sort(
        key=lambda row: (
            GAME_PRIORITY.get(row["game"], 99),
            _freshness_bucket(row["heartbeat_age_s"]),
            row["remaining"],
            row["heartbeat_age_s"] if row["heartbeat_age_s"] is not None else 1e18,
            row["pair"],
        )
    )
    return rows


def launch(spec: dict, args: argparse.Namespace) -> subprocess.Popen:
    RUNNER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNNER_LOG_DIR / f"{spec['game']}__{spec['pair']}.log"
    fh = open(log_path, "ab", buffering=0)
    env = {**os.environ, "PYTHONPATH": "src"}
    worker_max_concurrency = args.worker_max_concurrency
    pair_batch_size = args.pair_batch_size
    if spec["game"] == "holdem_match":
        if args.holdem_match_max_concurrency is not None:
            worker_max_concurrency = args.holdem_match_max_concurrency
        if args.holdem_match_pair_batch_size is not None:
            pair_batch_size = args.holdem_match_pair_batch_size
    cmd = [
        "python",
        str(RUNNER),
        "--games",
        spec["game"],
        "--models",
        ",".join(spec["models"]),
        "--anthropic-max-tokens",
        str(args.anthropic_max_tokens),
        "--fireworks-max-tokens",
        str(args.fireworks_max_tokens),
        "--timeout-s",
        str(args.timeout_s),
        "--max-concurrency",
        str(worker_max_concurrency),
        "--pair-batch-size",
        str(pair_batch_size),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_fh = fh  # type: ignore[attr-defined]
    proc._log_path = str(log_path)  # type: ignore[attr-defined]
    return proc


def close_proc(proc: subprocess.Popen) -> None:
    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        fh.close()


def save_state(
    state_file: Path,
    active: dict[
        tuple[str, str],
        tuple[dict, subprocess.Popen, float, int, float, float | None],
    ],
    cooldowns: dict[tuple[str, str], float],
    *,
    allowed_games: set[str] | None,
) -> None:
    payload = {
        "updated_at": time.time(),
        "active": [
            {
                "game": spec["game"],
                "pair": spec["pair"],
                "pid": proc.pid,
                "done": spec["done"],
                "target": spec["target"],
                "remaining": spec["remaining"],
                "heartbeat_age_s": heartbeat_age_s(spec["game"], spec["pair"]),
                "last_progress_age_s": time.time() - last_progress_at,
                "started_at": launched_at,
                "done_at_launch": done_at_launch,
                "log_path": getattr(proc, "_log_path", None),
            }
            for (_, _), (
                spec,
                proc,
                launched_at,
                done_at_launch,
                last_progress_at,
                _,
            ) in active.items()
        ],
        "cooldowns": [
            {
                "game": game,
                "pair": pair,
                "remaining_cooldown_s": max(0.0, until - time.time()),
            }
            for (game, pair), until in sorted(cooldowns.items())
            if until > time.time()
        ],
        "unfinished": unfinished_pairs(allowed_games=allowed_games),
    }
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(state_file)


def main() -> None:
    args = parse_args()
    state_file = Path(args.state_file).resolve()
    allowed_games = _allowed_games(args.games)
    active: dict[
        tuple[str, str],
        tuple[dict, subprocess.Popen, float, int, float, float | None],
    ] = {}
    cooldowns: dict[tuple[str, str], float] = {}

    try:
        while True:
            rows = unfinished_pairs(allowed_games=allowed_games)
            if not rows and not active:
                print("All requested pairs are complete.", flush=True)
                return

            now = time.time()
            rows_by_key = {(row["game"], row["pair"]): row for row in rows}

            for key in list(cooldowns):
                if cooldowns[key] <= now:
                    cooldowns.pop(key, None)

            # Refresh progress bookkeeping for active workers.
            for key, (
                spec,
                proc,
                launched_at,
                done_at_launch,
                last_progress_at,
                last_hb_age,
            ) in list(active.items()):
                row = rows_by_key.get(key)
                current_done = row["done"] if row is not None else spec["target"]
                current_hb_age = (
                    row["heartbeat_age_s"] if row is not None else heartbeat_age_s(spec["game"], spec["pair"])
                )
                saw_heartbeat_progress = (
                    current_hb_age is not None
                    and (
                        last_hb_age is None
                        or current_hb_age < last_hb_age
                        or current_hb_age <= max(2 * args.poll_s, 30.0)
                    )
                )
                if current_done > spec["done"]:
                    spec = dict(spec)
                    if row is not None:
                        spec.update(row)
                    else:
                        spec["done"] = current_done
                        spec["remaining"] = max(0, spec["target"] - current_done)
                    active[key] = (
                        spec,
                        proc,
                        launched_at,
                        done_at_launch,
                        now,
                        current_hb_age,
                    )
                elif saw_heartbeat_progress:
                    active[key] = (
                        spec,
                        proc,
                        launched_at,
                        done_at_launch,
                        now,
                        current_hb_age,
                    )
                else:
                    active[key] = (
                        spec,
                        proc,
                        launched_at,
                        done_at_launch,
                        last_progress_at,
                        current_hb_age,
                    )

            preferred = [
                row for row in rows
                if cooldowns.get((row["game"], row["pair"]), 0.0) <= now
            ]
            if len(preferred) < args.active_limit:
                preferred = rows
            wanted = {
                (row["game"], row["pair"]): row
                for row in preferred[: args.active_limit]
            }

            # Stop workers that are no longer in the active frontier.
            for key, (
                spec,
                proc,
                launched_at,
                done_at_launch,
                last_progress_at,
                last_hb_age,
            ) in list(active.items()):
                if key in wanted:
                    continue
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                close_proc(proc)
                active.pop(key, None)

            # Restart stale workers.
            for key, (
                spec,
                proc,
                launched_at,
                done_at_launch,
                last_progress_at,
                last_hb_age,
            ) in list(active.items()):
                hb_age = heartbeat_age_s(spec["game"], spec["pair"])
                if hb_age is None or hb_age <= args.stale_seconds:
                    continue
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                close_proc(proc)
                active.pop(key, None)
                cooldowns[key] = now + args.cooldown_seconds

            # Rotate pairs that keep producing heartbeats but no completed episodes.
            for key, (
                spec,
                proc,
                launched_at,
                done_at_launch,
                last_progress_at,
                last_hb_age,
            ) in list(active.items()):
                if now - last_progress_at <= args.no_progress_seconds:
                    continue
                hb_age = heartbeat_age_s(spec["game"], spec["pair"])
                if hb_age is not None and hb_age <= max(2 * args.poll_s, 30.0):
                    continue
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                close_proc(proc)
                active.pop(key, None)
                cooldowns[key] = now + args.cooldown_seconds

            # Reap exited workers.
            for key, (
                spec,
                proc,
                launched_at,
                done_at_launch,
                last_progress_at,
                last_hb_age,
            ) in list(active.items()):
                if proc.poll() is None:
                    continue
                close_proc(proc)
                active.pop(key, None)

            # Launch the desired frontier.
            for key, spec in wanted.items():
                if key in active:
                    continue
                proc = launch(spec, args)
                active[key] = (
                    spec,
                    proc,
                    now,
                    spec["done"],
                    now,
                    heartbeat_age_s(spec["game"], spec["pair"]),
                )

            save_state(
                state_file,
                active,
                cooldowns,
                allowed_games=allowed_games,
            )
            live = ", ".join(
                f"{spec['game']}:{spec['pair']} pid={proc.pid}"
                for spec, proc, _, _, _, _ in active.values()
            )
            print(
                f"active={len(active)} unfinished={len(rows)} {live}",
                flush=True,
            )
            time.sleep(args.poll_s)
    finally:
        for spec, proc, _, _, _, _ in active.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            close_proc(proc)


if __name__ == "__main__":
    main()
