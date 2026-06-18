"""Rebuild a board-game `<game>_data.json` aggregate from per-pair ep*.json files.

board_tournament.py writes <game>_data.json incrementally as it plays. When only
the per-pair ep files are available (e.g. new matchups synced without an updated
aggregate), this reconstructs the structure analyze_board_tournament.py expects:

    {"game": g, "episodes_per_pair": N, "models": [...],
     "games": [{"a", "b", "seed", "episodes": [...]}, ...]}

Pair dirs are named "<game>__<a>__vs__<b>". Steps are trimmed exactly like
board_tournament.trim (drop raw_output/prompt) to keep the aggregate small.

Usage:
    python3 scripts/aggregate_board_data.py connect4
    python3 scripts/aggregate_board_data.py gomoku
"""

from __future__ import annotations

import glob
import json
import os
import sys


def _trim_episode(e: dict) -> dict:
    e2 = dict(e)
    steps = []
    for s in e.get("steps", []):
        s2 = dict(s)
        resp = dict(s2.get("response") or {})
        resp.pop("raw_output", None)
        resp.pop("prompt", None)
        s2["response"] = resp
        steps.append(s2)
    e2["steps"] = steps
    return e2


def main(game: str) -> None:
    data_dir = os.path.join("runs", game)
    prefix = f"{game}__"
    pair_dirs = sorted(d for d in glob.glob(os.path.join(data_dir, prefix + "*__vs__*"))
                       if os.path.isdir(d))
    if not pair_dirs:
        raise SystemExit(f"no pair directories under {data_dir}")

    models: list[str] = []
    games = []
    per_pair = 0
    for gdir in pair_dirs:
        name = os.path.basename(gdir)[len(prefix):]
        a, b = name.split("__vs__")
        eps = [_trim_episode(json.load(open(p)))
               for p in sorted(glob.glob(os.path.join(gdir, "ep*.json")))]
        # Drop incomplete episodes (e.g. timed-out games missing returns/length/
        # steps) so they don't break the analyzer downstream.
        eps = [e for e in eps if all(k in e for k in
                                     ("returns", "length", "steps", "seat_assignment"))]
        if not eps:
            continue
        for nm in (a, b):
            if nm not in models:
                models.append(nm)
        per_pair = max(per_pair, len(eps))
        seed = eps[0].get("seed")
        games.append({"a": a, "b": b, "seed": seed, "episodes": eps})

    out = {"game": game, "episodes_per_pair": per_pair, "models": models, "games": games}
    dest = os.path.join(data_dir, f"{game}_data.json")
    json.dump(out, open(dest, "w"))
    print(f"Wrote {dest}: {len(models)} models, {len(games)} games, "
          f"{per_pair} episodes/pair max")
    print(f"models: {models}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: aggregate_board_data.py <game> [<game> ...]")
    for g in sys.argv[1:]:
        main(g)
