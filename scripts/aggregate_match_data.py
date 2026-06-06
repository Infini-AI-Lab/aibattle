"""Rebuild a match-mode `match_data.json` aggregate from per-episode ep*.json files.

The match tournament runner (match_tournament.py) writes match_data.json
incrementally as it plays. When only the per-pair ep*.json files are available
(e.g. a run synced without its aggregate), this reconstructs the same structure
analyze_match_tournament.py expects.

Usage:
    python3 scripts/aggregate_match_data.py runs/match_tournament_coached
"""

from __future__ import annotations

import glob
import json
import os
import sys

# Episode fields the runner keeps in the aggregate (analyze reads a subset;
# extras are harmless but we mirror match_tournament._trim to stay faithful).
_KEEP = ("episode", "seat_assignment", "returns", "winner", "winner_name",
         "length", "hands_played", "final_stacks", "stack_diff", "reason",
         "hand_summaries")


def _models_from_dir(name: str) -> tuple[str, str]:
    a, b = name.split("__vs__")
    return a, b


def main(data_dir: str) -> None:
    pair_dirs = sorted(d for d in glob.glob(os.path.join(data_dir, "*__vs__*"))
                       if os.path.isdir(d))
    if not pair_dirs:
        raise SystemExit(f"no pair directories under {data_dir}")

    models: list[str] = []
    pairs = []
    max_hands = 0
    starting_stack = 0
    max_eps = 0

    for pd in pair_dirs:
        a, b = _models_from_dir(os.path.basename(pd))
        for m in (a, b):
            if m not in models:
                models.append(m)
        episodes = []
        for ep_path in sorted(glob.glob(os.path.join(pd, "ep*.json"))):
            e = json.load(open(ep_path))
            max_hands = max(max_hands, e.get("max_hands") or 0)
            fs = e.get("final_stacks") or {}
            if fs:
                starting_stack = max(starting_stack, sum(fs.values()) // len(fs))
            episodes.append({k: e[k] for k in _KEEP if k in e})
        max_eps = max(max_eps, len(episodes))
        pairs.append({"a": a, "b": b, "episodes": episodes})

    data = {
        "mode": "match",
        "models": models,
        # Intended episodes/pair is run metadata we don't have here; the most
        # represented count is the best available proxy for display.
        "episodes_per_pair": max_eps,
        "max_hands": max_hands,
        "starting_stack": starting_stack,
        "pairs": pairs,
    }
    out = os.path.join(data_dir, "match_data.json")
    json.dump(data, open(out, "w"))
    total_eps = sum(len(p["episodes"]) for p in pairs)
    print(f"Wrote {out}: {len(models)} models, {len(pairs)} pairs, "
          f"{total_eps} episodes (max {max_eps}/pair), max_hands={max_hands}, "
          f"starting_stack={starting_stack}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/match_tournament")
