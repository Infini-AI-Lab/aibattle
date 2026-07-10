"""Extract must-block threat positions from the board-game runs into a probe set.

This is the data side of the threat-axis experiments (see the "miss-rate by
threat axis" section of the board reports): every decision where the acting
model faced exactly ONE blockable immediate-loss threat is dumped as a probe —
board, threat cell, threat axis, the VERBATIM prompt the model saw, and whether
the original model blocked. Replaying these probes under modified prompt
renderings (scripts/threat_probe_replay.py) gives a paired design: identical
positions, only the representation changes.

The threat-finding logic reproduces scripts/threat_axis_probe.py (git 7251020),
the script whose output is pinned in the reports, so the extracted set can be
validated against the published numbers (faced/missed by axis).

Usage:
    python3 scripts/threat_probes_extract.py            # both games
    python3 scripts/threat_probes_extract.py gomoku     # one game

Writes runs/threat_probes/probes_<game>.jsonl and prints the axis table next to
the report's pinned numbers.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

DIRS = {"horizontal": (0, 1), "vertical": (1, 0), "diag_dr": (1, 1), "diag_dl": (1, -1)}
GAMES = {"connect4": ("runs/connect4", 4), "gomoku": ("runs/gomoku", 5)}
OUT_DIR = "runs/threat_probes"
# Pinned report numbers (reports/<game>_report.html) for validation.
REPORT = {
    "connect4": {"horizontal": (825, 78), "vertical": (800, 94),
                 "diag_dr": (308, 36), "diag_dl": (369, 58)},
    "gomoku": {"horizontal": (428, 24), "vertical": (401, 49),
               "diag_dr": (284, 43), "diag_dl": (231, 56)},
}
_GCOLS = "ABCDEFGHI"


def winning_axes(grid, r, c, player, need):
    """Axes on which placing `player` at (r,c) completes a line of >= need."""
    rows, cols = len(grid), len(grid[0])
    g = [list(row) for row in grid]
    g[r][c] = player
    axes = []
    for name, (dr, dc) in DIRS.items():
        count = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < rows and 0 <= cc < cols and g[rr][cc] == player:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= need:
            axes.append(name)
    return axes


def c4_landing(grid, col):
    for r in range(len(grid) - 1, -1, -1):
        if grid[r][col] is None:
            return r
    return None


def _wins_of(grid, player, need, game):
    """All immediate winning placements for `player`: {(r,c): axes}."""
    wins = {}
    if game == "connect4":
        for col in range(len(grid[0])):
            r = c4_landing(grid, col)
            if r is None:
                continue
            ax = winning_axes(grid, r, col, player, need)
            if ax:
                wins[(r, col)] = ax
    else:
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] is None:
                    ax = winning_axes(grid, r, c, player, need)
                    if ax:
                        wins[(r, c)] = ax
    return wins


def extract(game: str):
    gdir, need = GAMES[game]
    probes = []
    faced = defaultdict(int)
    missed = defaultdict(int)
    n_double = 0
    for epf in sorted(glob.glob(os.path.join(gdir, "*", "ep*.json"))):
        ep = json.load(open(epf))
        if "steps" not in ep or "seat_assignment" not in ep:
            continue
        seat = ep["seat_assignment"]
        for si, st in enumerate(ep["steps"]):
            obs = st.get("observation") or {}
            pub = obs.get("public") or {}
            board = pub.get("board")
            if board is None or "player" not in obs:
                continue
            me = obs["player"]
            opp = "player_1" if me == "player_0" else "player_0"

            opp_wins = _wins_of(board, opp, need, game)
            if not opp_wins:
                continue
            if len(opp_wins) > 1:
                n_double += 1
                continue  # double threat: missing is forced, not perception

            (cell, axes), = opp_wins.items()
            axis = sorted(axes)[0]  # original convention (ties: alphabetical)

            move = st["selected_action"]
            if game == "connect4":
                blocked = move is not None and move.isdigit() and int(move) == cell[1]
            else:
                rc = None
                if move and len(move) >= 2 and move[0].upper() in _GCOLS and move[1:].isdigit():
                    r = int(move[1:]) - 1
                    if 0 <= r < 9:
                        rc = (r, _GCOLS.index(move[0].upper()))
                blocked = rc == cell

            # Not part of the original tabulation, but needed for fair scoring
            # downstream: winning yourself instead of blocking is not a miss.
            my_wins = _wins_of(board, me, need, game)

            faced[axis] += 1
            if not blocked:
                missed[axis] += 1

            probes.append({
                "id": f"{game}:{os.path.basename(os.path.dirname(epf))}:{os.path.basename(epf)}:s{si}",
                "game": game,
                "ep_file": epf,
                "step_index": si,
                "actor_player": me,
                "actor_agent": st.get("agent_name"),
                "actor_label": seat[me],
                "board": [["X" if v == "player_0" else "O" if v == "player_1" else None
                           for v in row] for row in board],
                "me_sym": "X" if me == "player_0" else "O",
                "threat_cell": list(cell),
                "axis": axis,
                "axes": sorted(axes),
                "win_available": bool(my_wins),
                "own_win_cells": [list(k) for k in my_wins],
                "original_action": move,
                "original_blocked": blocked,
                "prompt": (st.get("response") or {}).get("prompt"),
            })
    return probes, faced, missed, n_double


def main():
    games = sys.argv[1:] or list(GAMES)
    os.makedirs(OUT_DIR, exist_ok=True)
    for game in games:
        probes, faced, missed, n_double = extract(game)
        out = os.path.join(OUT_DIR, f"probes_{game}.jsonl")
        with open(out, "w") as fh:
            for p in probes:
                fh.write(json.dumps(p) + "\n")
        print(f"\n== {game}: {len(probes)} probes -> {out}  (double threats excluded: {n_double})")
        print(f"{'axis':12s} {'faced':>6s} {'missed':>7s} {'miss%':>6s}   {'report':>13s}  match")
        for ax in ["horizontal", "vertical", "diag_dr", "diag_dl"]:
            f, m = faced[ax], missed[ax]
            rf, rm = REPORT[game][ax]
            ok = "OK" if (f, m) == (rf, rm) else "MISMATCH"
            print(f"{ax:12s} {f:>6d} {m:>7d} {m / f if f else 0:>6.1%}   "
                  f"{rf:>6d}/{rm:<6d}  {ok}")
        n_win = sum(1 for p in probes if p["win_available"])
        n_noprompt = sum(1 for p in probes if not p["prompt"])
        print(f"probes with own immediate win available: {n_win}; without stored prompt: {n_noprompt}")


if __name__ == "__main__":
    main()
