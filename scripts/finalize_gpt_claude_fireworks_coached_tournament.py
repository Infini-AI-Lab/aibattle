"""Wait for the coached tournament to finish, then summarize, commit, and push.

This script is intended to run in the background while the tournament queue is
still active. It polls the log directory until all requested episodes are
present, removes stale error sidecars for completed episodes, regenerates the
markdown summary, commits the repo changes, and pushes the current branch.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT.parent / "aibattle-logs" / "gpt_claude_fireworks_coached_tournament"
REPORT_PATH = REPO_ROOT / "reports" / "gpt_claude_fireworks_coached_tournament_summary.md"
TARGETS = {
    "connect4": 1050,
    "gomoku": 1050,
    "holdem_1hand": 2100,
    "holdem_match": 420,
}
COMMIT_PATHS = [
    "scripts/bedrock_coached_tournament.py",
    "src/aibattle/models/bedrock_anthropic_client.py",
    "src/aibattle/models/bedrock_openai_client.py",
    "src/aibattle/models/registry.py",
    "scripts/gpt_claude_fireworks_coached_tournament.py",
    "scripts/run_gpt_claude_fireworks_coached_queue.py",
    "scripts/summarize_tournament_markdown.py",
    "scripts/finalize_gpt_claude_fireworks_coached_tournament.py",
    "reports/gpt_claude_fireworks_coached_tournament_summary.md",
]
COMMIT_MESSAGE = "Add GPT Claude Fireworks coached tournament automation and results"
POLL_SECONDS = 60


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def _completed_count(game_dir: Path) -> int:
    return sum(
        1
        for p in game_dir.glob("**/ep*.json")
        if not p.name.endswith(".error.json")
    )


def _counts() -> dict[str, int]:
    return {game: _completed_count(OUT_DIR / game) for game in TARGETS}


def _clean_stale_error_files() -> int:
    removed = 0
    for path in OUT_DIR.glob("**/*.error.json"):
        if path.name == "pair.error.json":
            pair_dir = path.parent
            game = pair_dir.parent.name
            target = 100 if game == "holdem_1hand" else 50
            if game == "holdem_match":
                target = 20
            done = sum(
                1
                for p in pair_dir.glob("ep*.json")
                if not p.name.endswith(".error.json")
            )
            if done >= target:
                path.unlink(missing_ok=True)
                removed += 1
            continue

        completed_episode = path.with_suffix("")
        if completed_episode.exists():
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _queue_active() -> bool:
    proc = _run(
        [
            "bash",
            "-lc",
            "ps -ef | grep 'run_gpt_claude_fireworks_coached_queue.py' | grep -v grep",
        ],
        check=False,
    )
    return bool(proc.stdout.strip())


def _workers_active() -> bool:
    proc = _run(
        [
            "bash",
            "-lc",
            "ps -ef | grep 'gpt_claude_fireworks_coached_tournament.py' | grep -v grep",
        ],
        check=False,
    )
    return bool(proc.stdout.strip())


def _counts_complete(counts: dict[str, int]) -> bool:
    return all(counts[game] == target for game, target in TARGETS.items())


def _wait_for_completion() -> None:
    while True:
        removed = _clean_stale_error_files()
        counts = _counts()
        print(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "counts",
            json.dumps(counts, sort_keys=True),
            "removed_stale_errors",
            removed,
            flush=True,
        )
        if _counts_complete(counts):
            if not _queue_active() and not _workers_active():
                return
        time.sleep(POLL_SECONDS)


def _generate_report() -> None:
    _run(
        [
            "python",
            "scripts/summarize_tournament_markdown.py",
            str(OUT_DIR),
            "--output",
            str(REPORT_PATH),
            "--title",
            "GPT Claude Fireworks Coached Tournament Summary",
            "--timeout-s",
            "180",
        ]
    )


def _commit_and_push() -> None:
    _run(["git", "config", "user.name", "jsw-zorro"])
    _run(["git", "config", "user.email", "shuoweijin@gmail.com"])
    _run(["git", "add", "--", *COMMIT_PATHS])

    status = _run(["git", "status", "--short", "--", *COMMIT_PATHS])
    if not status.stdout.strip():
        print("No repo changes to commit.", flush=True)
        return

    _run(["git", "commit", "-m", COMMIT_MESSAGE])
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    _run(["git", "push", "origin", branch])


def main() -> None:
    _wait_for_completion()
    removed = _clean_stale_error_files()
    print(f"Final stale error cleanup removed {removed} files.", flush=True)
    counts = _counts()
    if not _counts_complete(counts):
        raise RuntimeError(f"Completion check failed after wait: {counts}")
    _generate_report()
    _commit_and_push()
    print("Tournament finalized.", flush=True)


if __name__ == "__main__":
    main()
