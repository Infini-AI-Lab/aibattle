"""Bridge old (match.jsonl + <game>_data.json) board-tournament output to the new
per-episode resume format.

The pre-resume runner wrote one match.jsonl per match plus an incremental
aggregate <game>_data.json containing every COMPLETED match's episodes. The new
runner resumes from per-episode files (``<match_dir>/ep<NNN>.json``). This script
reads the aggregate and writes those per-episode files for each completed match,
so relaunching board_tournament.py resumes instead of recomputing finished work.

Only fully-saved matches in the aggregate are migrated (a match is only appended
there once all its episodes finished), so this is a safe, lossless, idempotent
bridge. In-progress matches that never made it into the aggregate are simply
recomputed by the resumed run. Safe to run repeatedly; existing ep files are
left untouched.
"""

from __future__ import annotations

import json
import os

GAMES = ["connect4", "gomoku"]
OUT = "runs/board_tournament"


def _atomic_write(path: str, record: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> None:
    total_written = 0
    for game in GAMES:
        path = os.path.join(OUT, f"{game}_data.json")
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            print(f"skip {path}: unreadable")
            continue
        for m in data.get("games", []):
            a, b = m.get("a"), m.get("b")
            gdir = os.path.join(OUT, f"{game}__{a}__vs__{b}")
            os.makedirs(gdir, exist_ok=True)
            written = 0
            for e in m.get("episodes", []):
                idx = e.get("episode")
                if idx is None:
                    continue
                epath = os.path.join(gdir, f"ep{idx:03d}.json")
                if os.path.exists(epath):
                    continue  # already migrated / freshly played
                _atomic_write(epath, e)
                written += 1
            if written:
                print(f"{game}: {a} vs {b} -> wrote {written} episode files")
            total_written += written
    print(f"migration done: {total_written} episode files written")


if __name__ == "__main__":
    main()
