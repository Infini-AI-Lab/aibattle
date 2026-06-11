"""Analyze Independent Blackjack runs — player-seat only.

Blackjack is agent-vs-environment: seat ``player_0`` is the model under
evaluation and seat ``player_1`` is the scripted dealer. Plain ``aibattle eval``
would aggregate the dealer as if it were a competitor and pollute standings, so
this dedicated script reports ONLY the player seat (``player_0``) and explicitly
excludes the dealer from any ranking.

For one or more run directories it reports, per player-seat agent name:
  - hands played
  - total profit and mean profit per hand (the key risk-calibration signal)
  - win / loss / push rates
  - bust rate, double rate, natural rate
  - invalid-action rate

Usage:
  python scripts/analyze_blackjack_results.py runs/blackjack_gptoss [more_run_dirs...]
If no directory is given, every immediate subdirectory of runs/ whose game is
independent_blackjack is scanned.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

DEALER_SEAT = "player_1"
PLAYER_SEAT = "player_0"


def _new_stats():
    return {
        "hands": 0, "profit": 0.0, "wins": 0, "losses": 0, "pushes": 0,
        "busts": 0, "doubles": 0, "naturals": 0, "invalid": 0,
    }


def _latest_run(run_root: str):
    """Return the newest run_* subdir of ``run_root`` (or run_root itself)."""
    subs = sorted(glob.glob(os.path.join(run_root, "run_*")))
    return subs[-1] if subs else run_root


def _load_episodes(run_dir: str):
    """Yield episode dicts from trajectories.json or match.jsonl."""
    traj = os.path.join(run_dir, "trajectories.json")
    if os.path.exists(traj):
        with open(traj, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data.get("game"), data.get("episodes", [])
        return None, data
    # Fall back to the JSONL log.
    mj = os.path.join(run_dir, "match.jsonl")
    game = None
    eps = []
    if os.path.exists(mj):
        with open(mj, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("record_type") == "match":
                    game = rec.get("game")
                elif rec.get("record_type") == "episode":
                    eps.append(rec)
    return game, eps


def analyze_run(run_dir: str):
    game, eps = _load_episodes(run_dir)
    if game and game != "independent_blackjack":
        return None
    # Per player-seat agent name, accumulate stats. Only the player seat
    # (player_0) is tracked; the dealer seat (player_1) is never reported.
    stats = defaultdict(_new_stats)
    for ep in eps:
        seats = ep.get("seat_assignment", {})
        name = seats.get(PLAYER_SEAT, PLAYER_SEAT)
        ret = ep.get("returns", {})
        profit = float(ret.get(PLAYER_SEAT, 0.0))
        s = stats[name]
        s["hands"] += 1
        s["profit"] += profit
        if profit > 0:
            s["wins"] += 1
        elif profit < 0:
            s["losses"] += 1
        else:
            s["pushes"] += 1
        if ep.get("player_bust"):
            s["busts"] += 1
        if ep.get("doubled"):
            s["doubles"] += 1
        if ep.get("player_natural"):
            s["naturals"] += 1
        inv = ep.get("invalid_count", {})
        s["invalid"] += int(inv.get(PLAYER_SEAT, 0))
    return stats


def _fmt(stats: dict) -> str:
    lines = []
    header = (f"{'player(model)':<22}{'hands':>7}{'profit':>10}{'mean':>9}"
              f"{'win%':>7}{'loss%':>7}{'push%':>7}{'bust%':>7}{'dbl%':>7}"
              f"{'nat%':>7}{'inval%':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["profit"]):
        h = max(1, s["hands"])
        lines.append(
            f"{name:<22}{s['hands']:>7}{s['profit']:>10.2f}{s['profit']/h:>9.3f}"
            f"{100*s['wins']/h:>7.1f}{100*s['losses']/h:>7.1f}{100*s['pushes']/h:>7.1f}"
            f"{100*s['busts']/h:>7.1f}{100*s['doubles']/h:>7.1f}{100*s['naturals']/h:>7.1f}"
            f"{100*s['invalid']/h:>8.1f}"
        )
    return "\n".join(lines)


def main(argv):
    roots = argv[1:]
    if not roots:
        roots = [d for d in glob.glob("runs/*") if os.path.isdir(d)]
    combined = defaultdict(_new_stats)
    found = False
    for root in roots:
        run_dir = _latest_run(root)
        st = analyze_run(run_dir)
        if st is None:
            continue
        found = True
        print(f"\n== {run_dir} ==")
        print(_fmt(st))
        for name, s in st.items():
            for k, v in s.items():
                combined[name][k] += v
    if not found:
        print("No independent_blackjack runs found.")
        return 1
    if len(roots) > 1:
        print("\n== combined (player seat only; dealer excluded) ==")
        print(_fmt(combined))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
