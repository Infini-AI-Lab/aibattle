"""Rebuild a Hold'em-Lite `tournament_data.json` aggregate from per-episode files.

tournament.py writes tournament_data.json incrementally as it plays. When only
the per-pair ep*.json files are available (e.g. a run synced without an updated
aggregate, or new matchups added after the fact), this reconstructs the same
structure analyze_tournament.py expects:

    {"models": [...], "hands": N, "reps": R,
     "games": [{"gid", "a", "b", "rep", "episodes": [...]}, ...]}

Steps are trimmed exactly like tournament.trim (drop raw_output/prompt from each
step's response) to keep the aggregate small; the full copies live per-episode.

Usage:
    python3 scripts/aggregate_tournament_data.py runs/holdem_1hand
"""

from __future__ import annotations

import glob
import json
import os
import re
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


def main(data_dir: str) -> None:
    pair_dirs = sorted(d for d in glob.glob(os.path.join(data_dir, "*__vs__*"))
                       if os.path.isdir(d))
    if not pair_dirs:
        raise SystemExit(f"no pair directories under {data_dir}")

    models: list[str] = []
    games = []
    hands = 0
    for gid, gdir in enumerate(pair_dirs):
        name = os.path.basename(gdir)
        m = re.match(r"(.+)__vs__(.+?)(?:__r(\d+))?$", name)
        a, b, rep = m.group(1), m.group(2), int(m.group(3) or 0)
        for nm in (a, b):
            if nm not in models:
                models.append(nm)
        eps = []
        for ep_path in sorted(glob.glob(os.path.join(gdir, "ep*.json"))):
            eps.append(_trim_episode(json.load(open(ep_path))))
        if not eps:
            continue
        hands = max(hands, len(eps))
        games.append({"gid": gid, "a": a, "b": b, "rep": rep, "episodes": eps})

    reps = max((g["rep"] for g in games), default=0) + 1
    out = {"models": models, "hands": hands, "reps": reps, "games": games}
    dest = os.path.join(data_dir, "tournament_data.json")
    json.dump(out, open(dest, "w"))
    print(f"Wrote {dest}: {len(models)} models, {len(games)} games, "
          f"{hands} hands/game max")
    print(f"models: {models}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/holdem_1hand")
