"""Rebuild a round-robin "pairs" aggregate from per-pair ep*.json files.

Covers the versus-format tournaments whose aggregate is
{"game", "models", "episodes_per_pair", "pairs": [{"a","b","seed","episodes"}]}:
Kuhn (runs/kuhn_poker/kuhn_data.json) and the new-games versus tournaments
(runs/new_games_experiment/<game>/data.json — leduc, blotto, othello).

When new matchups are synced without an updated aggregate, this reconstructs it
from the per-pair dirs (named "<a>__vs__<b>"). Any extra top-level keys in the
existing aggregate (game, structure, settings) are preserved; the stale
leaderboard, if any, is dropped (the analyzer recomputes it). Steps are trimmed
(raw_output/prompt dropped) to keep the file small — the analyzers don't read them.

Usage:
    python3 scripts/aggregate_versus_data.py runs/kuhn_poker kuhn_data.json
    python3 scripts/aggregate_versus_data.py runs/new_games_experiment/leduc_poker
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


def main(run_dir: str, out_name: str = "data.json") -> None:
    pair_dirs = sorted(d for d in glob.glob(os.path.join(run_dir, "*__vs__*"))
                       if os.path.isdir(d))
    if not pair_dirs:
        raise SystemExit(f"no pair directories under {run_dir}")

    dest = os.path.join(run_dir, out_name)
    # Preserve descriptive top-level keys (game/structure/settings); recompute
    # models/episodes_per_pair/pairs; drop any stale leaderboard.
    base = {}
    if os.path.exists(dest):
        old = json.load(open(dest))
        base = {k: old[k] for k in ("game", "structure", "settings") if k in old}

    models: list[str] = []
    pairs = []
    per_pair = 0
    for pd in pair_dirs:
        a, b = os.path.basename(pd).split("__vs__")
        eps = [_trim_episode(json.load(open(p)))
               for p in sorted(glob.glob(os.path.join(pd, "ep*.json")))]
        if not eps:
            continue
        for nm in (a, b):
            if nm not in models:
                models.append(nm)
        per_pair = max(per_pair, len(eps))
        seed = eps[0].get("seed")
        pairs.append({"a": a, "b": b, "seed": seed, "episodes": eps})

    out = {**base, "models": models, "episodes_per_pair": per_pair, "pairs": pairs}
    json.dump(out, open(dest, "w"))
    print(f"Wrote {dest}: {len(models)} models, {len(pairs)} pairs, "
          f"{per_pair} episodes/pair max")
    print(f"models: {models}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: aggregate_versus_data.py <run_dir> [out_name]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data.json")
