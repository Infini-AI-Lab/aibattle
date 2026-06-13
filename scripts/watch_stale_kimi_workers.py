"""Recycle stale tournament workers from the shared queue state.

This watchdog is operational glue for long-running tournament batches. It reads
the queue supervisor state file, and when a worker has gone too long without
progress for its game type, it terminates that worker process group so the
queue can relaunch it cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


DEFAULT_STATE = "/tmp/gpt_claude_fireworks_queue_state.json"
DEFAULT_LOG = (
    Path(__file__).resolve().parents[1].parent
    / "aibattle-logs"
    / "gpt_claude_fireworks_coached_tournament"
    / "_runner_logs"
    / "stale_kimi_watchdog.log"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-file", default=DEFAULT_STATE)
    p.add_argument("--log-path", default=str(DEFAULT_LOG))
    p.add_argument("--short-game-threshold-s", type=float, default=240.0)
    p.add_argument("--holdem-threshold-s", type=float, default=450.0)
    p.add_argument("--sleep-s", type=float, default=120.0)
    return p.parse_args()


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def _threshold_for(row: dict, *, short_game_threshold_s: float, holdem_threshold_s: float) -> float | None:
    game = row.get("game")
    if game in {"connect4", "gomoku"}:
        return short_game_threshold_s
    if game in {"holdem_match", "holdem_1hand"}:
        return holdem_threshold_s
    return None


def recycle_stale_workers(
    state_file: Path,
    *,
    short_game_threshold_s: float,
    holdem_threshold_s: float,
    log_path: Path,
) -> None:
    if not state_file.exists():
        log_line(log_path, f"state file missing: {state_file}")
        return

    payload = json.loads(state_file.read_text())
    killed = []
    for row in payload.get("active", []):
        threshold_s = _threshold_for(
            row,
            short_game_threshold_s=short_game_threshold_s,
            holdem_threshold_s=holdem_threshold_s,
        )
        if threshold_s is None:
            continue
        pair = row.get("pair", "")
        age = float(row.get("last_progress_age_s", 0.0) or 0.0)
        if age < threshold_s:
            continue
        pid = int(row["pid"])
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        killed.append((pid, row.get("game"), pair, round(age, 1)))

    if killed:
        for pid, game, pair, age in killed:
            log_line(
                log_path,
                f"recycled pid={pid} game={game} pair={pair} age={age}",
            )
        return

    log_line(log_path, "no stale workers")


def main() -> None:
    args = parse_args()
    state_file = Path(args.state_file).resolve()
    log_path = Path(args.log_path).resolve()
    while True:
        try:
            recycle_stale_workers(
                state_file,
                short_game_threshold_s=args.short_game_threshold_s,
                holdem_threshold_s=args.holdem_threshold_s,
                log_path=log_path,
            )
        except Exception as exc:  # noqa: BLE001
            log_line(log_path, f"watchdog error: {exc!r}")
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
