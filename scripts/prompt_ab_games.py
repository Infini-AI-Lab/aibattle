"""Prompt-A/B self-play: does a modified board rendering win real games?

The threat-axis experiments (analysis/threat_axis_*.md) showed miss rates depend
on how the board is serialized. This harness tests whether that converts into
GAME performance: full Gomoku games between two seats of the SAME model, one
seat prompted with the original tournament rendering (`orig`), the other with a
modified rendering. Same seeded random openings as the tournament, seat-balanced
(variant plays player_0 on even seeds), resumable, with per-decision tactical
stats (block rate / win-take) logged per seat.

Settings combine the studied manipulations: flipped row labels (flip), the
45-degree rotated serialization with its own A-Q/1-17 coordinates (serial_dl,
serial_dr), a column-major view (transpose), stone coordinate lists, multi-view
prompts (e.g. orig+serial_dl), and a coaching-only variant (line_coach).

Usage (repo root):
  python3 scripts/prompt_ab_games.py run --models kimi-k2p6 \
      --settings flip,serial_dl --games 100 --concurrency 32
  python3 scripts/prompt_ab_games.py summary

Results: runs/prompt_ab/<model>__<setting>.jsonl (one line per game).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(".fireworks"):
    os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.games.gomoku import Gomoku, rc_to_coord  # noqa: E402
from aibattle.models.registry import make_client  # noqa: E402
from aibattle.types import Move  # noqa: E402
from threat_probe_replay import (  # noqa: E402
    _COORD, _NEW_COORD, _NEWCOLS, _COLS, parse_answer,
    diamond_block, serial_coord_of, serial_rc_of)
from threat_probes_extract import _wins_of  # noqa: E402

OUT_DIR = "runs/prompt_ab"
NEED = 5

# --- prompt pieces (mirroring the coached tournament template) ---------------
RULES_STD = ("You are playing Gomoku-Lite (9x9). Place a stone on any empty cell; "
             "connect five in a row (horizontal, vertical, or diagonal) to win. "
             "Columns are A-I, rows 1-9; center is E5.")
COACH = ("Before you move, check whether you can make five, whether you must block "
         "the opponent's line, and how your own stones connect.")
INSTR_STD = ("Respond with ONLY a coordinate for an empty cell, e.g. E5 (column "
             "letter A-I, row number 1-9). Think privately before you answer.")
RULES_SERIAL = ("You are playing Gomoku-Lite (9x9). Place a stone on any empty cell; "
                "connect five in a row (horizontal, vertical, or diagonal on the "
                "original board) to win. In the rotated view below, columns are A-Q, "
                "rows 1-17; the board center is I9.")
INSTR_SERIAL = ("Respond with ONLY a coordinate for an empty cell in the rotated "
                "view, e.g. I9 (column letter A-Q, row number 1-17). Think privately "
                "before you answer.")
REPAIR_STD = ("Your previous reply was not a valid empty cell. Reply with one "
              "coordinate like E5 that is currently empty.")
REPAIR_SERIAL = ("Your previous reply was not a valid empty cell. Reply with one "
                 "coordinate in the rotated view, like I9 (column letter A-Q, row "
                 "number 1-17), for a cell that is currently empty.")
SERIAL_DL_NOTE = ("This view is rotated 45 degrees: each line is one down-left "
                  "diagonal of the board (row +1, column -1 per step to the right).")
SERIAL_DR_NOTE = ("This view is rotated 45 degrees: each line is one down-right "
                  "diagonal of the board (row +1, column +1 per step to the right).")


# --- board views (grid = 9x9 of 'X'/'O'/None) --------------------------------
def view_orig(g):
    lines = ["   " + " ".join(_COLS)]
    for r in range(9):
        lines.append(f"{r + 1:>2} " + " ".join(g[r][c] or "." for c in range(9)))
    return "\n".join(lines)


def view_flip(g):
    lines = ["   " + " ".join(_COLS)]
    for r in range(9):
        lines.append(f"{9 - r:>2} " + " ".join(g[r][c] or "." for c in range(9)))
    return "\n".join(lines)


def view_transpose(g):
    lines = ["   " + " ".join(str(i + 1) for i in range(9))]
    for c in range(9):
        lines.append(f" {_COLS[c]} " + " ".join(g[r][c] or "." for r in range(9)))
    return "\n".join(lines)


def view_serial_dl(g):
    return diamond_block(g)


def view_serial_dr(g):
    lines = ["    " + " ".join(_NEWCOLS)]
    for k in range(17):           # line k: r-c = k-8, from top-left end down-right
        d = k - 8
        r, c = max(0, d), max(0, -d)
        cells = []
        while r <= 8 and c <= 8:
            cells.append(g[r][c] or ".")
            r += 1
            c += 1
        lines.append(f"{k + 1:>2}  " + "  " * abs(8 - k) + "   ".join(cells))
    return "\n".join(lines)


def view_stones(g):
    xs = [rc_to_coord(r, c) for r in range(9) for c in range(9) if g[r][c] == "X"]
    os_ = [rc_to_coord(r, c) for r in range(9) for c in range(9) if g[r][c] == "O"]
    return (f"X stones: {', '.join(xs) if xs else 'none'}\n"
            f"O stones: {', '.join(os_) if os_ else 'none'}")


VIEW = {"orig": (view_orig, "standard board (row by row, rows 1-9 top to bottom)"),
        "flip": (view_flip, "same board with row labels flipped (9 at top, 1 at bottom)"),
        "transpose": (view_transpose, "same board column by column (each line is one column A-I)"),
        "serial_dl": (view_serial_dl, "same board rotated 45 degrees (each line is one "
                                      "down-left diagonal; reading aid only)"),
        "serial_dr": (view_serial_dr, "same board rotated 45 degrees the other way (each "
                                      "line is one down-right diagonal; reading aid only)"),
        "stones": (view_stones, "the stones as coordinate lists")}

# name -> (views, answer_frame, extra_coaching)
SETTINGS = {
    "orig":                (["orig"], "base", None),
    "flip":                (["flip"], "flip", None),
    "serial_dl":           (["serial_dl"], "serial_dl", None),
    "serial_dr":           (["serial_dr"], "serial_dr", None),
    "transpose":           (["transpose"], "base", None),
    "orig+flip":           (["orig", "flip"], "base", None),
    "orig+serial_dl":      (["orig", "serial_dl"], "base", None),
    "orig+serial_dr":      (["orig", "serial_dr"], "base", None),
    "orig+transpose":      (["orig", "transpose"], "base", None),
    "orig+stones":         (["orig", "stones"], "base", None),
    "orig+flip+serial_dl": (["orig", "flip", "serial_dl"], "base", None),
    "four_views":          (["orig", "transpose", "serial_dl", "serial_dr"], "base", None),
    "flip+serial_dl":      (["flip", "serial_dl"], "flip", None),
    "line_coach":          (["orig"], "base",
                            "Explicitly scan all four directions for lines of four "
                            "— especially the two diagonals (down-right and down-left)."),
}


# --- answer frames ------------------------------------------------------------
def serial_dr_coord_of(r, c):
    return f"{_NEWCOLS[r + c]}{r - c + 9}"


def serial_dr_rc_of(coord):
    k, x = int(coord[1:]) - 1, _NEWCOLS.index(coord[0])
    return (x + k - 8) // 2, (x - k + 8) // 2


FRAMES = {
    "base":      (lambda r, c: f"{_COLS[c]}{r + 1}",
                  lambda coord: (int(coord[1:]) - 1, _COLS.index(coord[0])),
                  _COORD, REPAIR_STD),
    "flip":      (lambda r, c: f"{_COLS[c]}{9 - r}",
                  lambda coord: (9 - int(coord[1:]), _COLS.index(coord[0])),
                  _COORD, REPAIR_STD),
    "serial_dl": (serial_coord_of, serial_rc_of, _NEW_COORD, REPAIR_SERIAL),
    "serial_dr": (serial_dr_coord_of, serial_dr_rc_of, _NEW_COORD, REPAIR_SERIAL),
}


def build_prompt(setting, g, my_sym):
    views, frame, extra = SETTINGS[setting]
    serial_only = frame in ("serial_dl", "serial_dr")
    rules = RULES_SERIAL if serial_only else RULES_STD
    coach = COACH + (" " + extra if extra else "")
    if len(views) == 1:
        note = SERIAL_DL_NOTE if views[0] == "serial_dl" else \
               SERIAL_DR_NOTE if views[0] == "serial_dr" else None
        body = (note + "\n\n" if note and serial_only else "") + VIEW[views[0]][0](g)
    else:
        parts = [f"The current position is shown in {len(views)} views of the SAME board."]
        for i, v in enumerate(views, 1):
            parts.append(f"View {i} — {VIEW[v][1]}:\n{VIEW[v][0](g)}")
        body = "\n\n".join(parts)
    instr = INSTR_SERIAL if serial_only else INSTR_STD
    if len(views) > 1:
        anchor = ("the labels of View 1" if frame == "flip"
                  else "the standard A-I / 1-9 coordinates")
        instr += f" Give your answer using {anchor}."
    return "\n\n".join([rules, coach,
                        f"Match: Hand 1 of 1.\nYou are {my_sym}.\n{body}", instr])


# --- game loop ----------------------------------------------------------------
def _syms(grid):
    return [["X" if v == "player_0" else "O" if v == "player_1" else None
             for v in row] for row in grid]


async def play_game(clients, model, setting, seed, sem, args):
    game = Gomoku(random_open=2)
    s = game.initial_state(random.Random(seed))
    variant_seat = "player_0" if seed % 2 == 0 else "player_1"
    frng = random.Random(seed * 7919 + 13)
    stats = {p: {"blk_faced": 0, "blk_ok": 0, "wt_faced": 0, "wt_ok": 0, "invalid": 0}
             for p in ("player_0", "player_1")}
    moves = []
    while not s.done:
        me = s.to_act
        opp = "player_1" if me == "player_0" else "player_0"
        my_sym = "X" if me == "player_0" else "O"
        g = _syms(s.grid)
        setting_here = setting if me == variant_seat else "orig"
        frame = SETTINGS[setting_here][1]
        coord_of, rc_of, regex, repair = FRAMES[frame]
        legal_rc = [(r, c) for r in range(9) for c in range(9) if g[r][c] is None]
        legal = {coord_of(r, c) for r, c in legal_rc}
        prompt = build_prompt(setting_here, g, my_sym)

        coord = None
        for attempt in range(3):
            async with sem:
                out = await clients[model].generate(
                    prompt if attempt == 0 else f"{prompt}\n\n{repair}",
                    max_tokens=args.max_tokens)
            coord = parse_answer(out.content, legal, regex)
            if coord:
                break
        if coord:
            r, c = rc_of(coord)
        else:
            r, c = frng.choice(legal_rc)
            stats[me]["invalid"] += 1

        opp_wins = _wins_of(s.grid, opp, NEED, "gomoku")
        my_wins = _wins_of(s.grid, me, NEED, "gomoku")
        if my_wins:
            stats[me]["wt_faced"] += 1
            stats[me]["wt_ok"] += (r, c) in my_wins
        elif len(opp_wins) == 1:
            stats[me]["blk_faced"] += 1
            stats[me]["blk_ok"] += (r, c) == list(opp_wins)[0]

        moves.append(rc_to_coord(r, c))
        if args.verbose:
            print(f"    {model}/{setting} seed={seed} move {len(moves)}: "
                  f"{rc_to_coord(r, c)} ({me})", flush=True)
        s = game.step(s, Move(type=rc_to_coord(r, c)))
        if len(moves) > 81:
            break

    winner_seat = s.winner  # player id or None
    outcome = ("draw" if winner_seat is None else
               "variant" if winner_seat == variant_seat else "orig")
    return {"model": model, "setting": setting, "seed": seed,
            "variant_seat": variant_seat, "outcome": outcome,
            "length": len(moves), "moves": moves,
            "variant_stats": stats[variant_seat],
            "orig_stats": stats["player_1" if variant_seat == "player_0" else "player_0"]}


async def run(args):
    models = args.models.split(",")
    settings = args.settings.split(",")
    os.makedirs(OUT_DIR, exist_ok=True)
    clients = {m: make_client({
        "provider": "fireworks", "model_id": f"accounts/fireworks/models/{m}",
        "api_key_env": "FIREWORKS_API_KEY", "temperature": args.temperature,
        "max_tokens": args.max_tokens, "timeout_s": args.timeout}) for m in models}
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    jobs = []
    for m in models:
        for st in settings:
            assert st in SETTINGS and st != "orig", st
            path = os.path.join(OUT_DIR, f"{m}__{st}.jsonl")
            done = set()
            if os.path.exists(path):
                done = {json.loads(l)["seed"] for l in open(path)}
            jobs += [(m, st, seed, path) for seed in range(args.games) if seed not in done]
    print(f"{len(jobs)} games to play")
    t0, n = time.time(), 0

    async def one(m, st, seed, path):
        nonlocal n
        try:
            rec = await play_game(clients, m, st, seed, sem, args)
        except Exception as e:
            print(f"  ERROR {m} {st} seed={seed}: {e!r}")
            return
        async with lock:
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} games ({time.time() - t0:.0f}s)")

    await asyncio.gather(*[one(*j) for j in jobs])


def summary():
    import glob
    print(f"{'model':16s} {'setting':22s} {'n':>4s} {'variant':>8s} {'orig':>6s} "
          f"{'draw':>5s} {'score':>6s}  {'blk% v/o':>10s}  {'inv v/o':>8s}")
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl"))):
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            continue
        m, st = os.path.basename(path)[:-6].split("__")
        w = sum(r["outcome"] == "variant" for r in rows)
        o = sum(r["outcome"] == "orig" for r in rows)
        d = sum(r["outcome"] == "draw" for r in rows)
        score = (w + 0.5 * d) / len(rows)
        bv = sum(r["variant_stats"]["blk_ok"] for r in rows), sum(r["variant_stats"]["blk_faced"] for r in rows)
        bo = sum(r["orig_stats"]["blk_ok"] for r in rows), sum(r["orig_stats"]["blk_faced"] for r in rows)
        iv = sum(r["variant_stats"]["invalid"] for r in rows)
        io = sum(r["orig_stats"]["invalid"] for r in rows)
        blk = (f"{bv[0] / bv[1]:.0%}/{bo[0] / bo[1]:.0%}"
               if bv[1] and bo[1] else "—")
        print(f"{m:16s} {st:22s} {len(rows):>4d} {w:>8d} {o:>6d} {d:>5d} "
              f"{score:>6.1%}  {blk:>10s}  {iv:>3d}/{io:<3d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--models", required=True)
    rp.add_argument("--settings", required=True)
    rp.add_argument("--games", type=int, default=100)
    rp.add_argument("--concurrency", type=int, default=32)
    rp.add_argument("--temperature", type=float, default=0.6)
    rp.add_argument("--max-tokens", type=int, default=131072)
    rp.add_argument("--timeout", type=float, default=300)
    rp.add_argument("--verbose", action="store_true",
                    help="print every completed move (progress visibility)")
    sub.add_parser("summary")
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    else:
        summary()


if __name__ == "__main__":
    main()
