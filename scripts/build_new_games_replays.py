"""Build replay data for the four new games (othello, leduc, blotto, blackjack).

Reads the per-episode ep*.json files under
  runs/new_games_experiment/<game>/<pair-or-model>/ep*.json
and writes, next to each game's run folder,
  runs/new_games_experiment/<game>/replays/<short>/manifest.json
  runs/new_games_experiment/<game>/replays/<short>/<...>.json   (one per pair/model)

in exactly the schema the terminal replay viewers (reports/<short>_replay.html)
consume. This is the new-games counterpart of scripts/build_replays.py.

Othello stores no per-move board, so flips are simulated here with the standard
reversi rules (coord letter = column, digit = 1-based row) and validated against
each episode's recorded final piece_counts; a mismatch is reported, never shipped
silently.
"""

from __future__ import annotations

import glob
import json
import os

EXP = "runs/new_games_experiment"
THINK_MARK = "===== thinking ====="
ANSWER_MARK = "===== answer ====="
TRUNC_MIN_CHARS = 15000
_TERMINAL = '.!?")]'
_PLAYERS = ("player_0", "player_1")


# --- shared step helpers ---------------------------------------------------
def _split_thinking(raw: str) -> str:
    if not raw:
        return ""
    if THINK_MARK in raw:
        return raw.split(THINK_MARK, 1)[1].split(ANSWER_MARK, 1)[0].strip()
    return raw.strip()


# Some models (notably minimax on othello) emit 50k–150k-char reasoning per
# move, which blows a per-pair replay file up past 100 MB — the viewer fetch
# then freezes the browser tab. Cap the stored reasoning so the viewer stays
# usable; the truncation flag already marks runs that ran past the token budget.
THINK_CAP = 6000


def _think(s: dict) -> str:
    t = _split_thinking((s.get("response") or {}).get("raw_output") or "")
    if len(t) > THINK_CAP:
        t = t[:THINK_CAP] + f"\n\n… [reasoning clipped to {THINK_CAP} chars for replay]"
    return t


def _trunc(thinking: str, invalid: bool) -> bool:
    if invalid:
        return True
    t = (thinking or "").rstrip()
    return len(t) >= TRUNC_MIN_CHARS and (not t or t[-1] not in _TERMINAL)


def _meta(s: dict, key: str):
    return (s.get("response") or {}).get("metadata", {}).get(key)


def _eps(pair_dir: str):
    for path in sorted(glob.glob(os.path.join(pair_dir, "ep*.json"))):
        try:
            yield json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def _pairs(game_dir: str):
    return sorted(d for d in glob.glob(os.path.join(game_dir, "*__vs__*"))
                  if os.path.isdir(d))


def _out_dir(game: str, short: str) -> str:
    d = os.path.join(EXP, game, "replays", short)
    os.makedirs(d, exist_ok=True)
    return d


def _write(out_dir: str, name: str, obj: dict):
    json.dump(obj, open(os.path.join(out_dir, name), "w", encoding="utf-8"))


def _report(short: str, out_dir: str, n: int):
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"[{short}] wrote {n} files + manifest to {out_dir} ({total/1e6:.1f} MB)")


# --- othello ---------------------------------------------------------------
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _coord_rc(coord: str, size: int):
    """'C2' -> (row, col): letter = column, 1-based digit = row."""
    s = (coord or "").strip().upper()
    if len(s) < 2 or not s[0].isalpha() or not s[1:].isdigit():
        return None
    col, row = ord(s[0]) - ord("A"), int(s[1:]) - 1
    if 0 <= row < size and 0 <= col < size:
        return (row, col)
    return None


def _flips(board, row, col, me):
    """Cells that flip if `me` plays (row,col); [] if illegal."""
    size = len(board)
    out = []
    for dr, dc in _DIRS:
        line, r, c = [], row + dr, col + dc
        while 0 <= r < size and 0 <= c < size and board[r][c] not in (None, me):
            line.append((r, c)); r += dr; c += dc
        if line and 0 <= r < size and 0 <= c < size and board[r][c] == me:
            out += line
    return out


def build_othello():
    game = "othello_lite_6x6"
    gdir = os.path.join(EXP, game)
    if not os.path.isdir(gdir):
        print(f"skip othello: no {gdir}"); return
    out_dir = _out_dir(game, "othello")
    manifest, size, mismatches = [], 6, 0
    for pd in _pairs(gdir):
        a, b = os.path.basename(pd).split("__vs__")
        episodes = []
        for e in _eps(pd):
            steps = e.get("steps", [])
            if not steps:
                continue
            init = [row[:] for row in steps[0]["observation"]["public"]["board"]]
            size = len(init)
            board = [row[:] for row in init]
            moves = []
            for s in steps:
                me = s["player"]
                act = str(s.get("selected_action") or "").strip()
                is_pass = act.lower() == "pass"
                rc = None if is_pass else _coord_rc(act, size)
                flips = []
                if rc is not None and board[rc[0]][rc[1]] is None:
                    flips = _flips(board, rc[0], rc[1], me)
                    if flips:
                        board[rc[0]][rc[1]] = me
                        for fr, fc in flips:
                            board[fr][fc] = me
                cnt = {p: sum(r.count(p) for r in board) for p in _PLAYERS}
                th = _think(s); inv = bool(s.get("invalid"))
                moves.append({
                    "ply": s["step"], "player": me, "agent": s["agent_name"],
                    "coord": act, "rc": list(rc) if rc else None, "pass": is_pass,
                    "flips": [list(f) for f in flips], "counts": cnt,
                    "invalid": inv, "latency_ms": _meta(s, "latency_ms"),
                    "tokens": _meta(s, "completion_tokens"),
                    "thinking": th, "trunc": _trunc(th, inv),
                })
            final = {p: sum(r.count(p) for r in board) for p in _PLAYERS}
            recorded = e.get("piece_counts") or {}
            if recorded and any(final[p] != recorded.get(p) for p in _PLAYERS):
                mismatches += 1
            episodes.append({
                "episode": e["episode"], "seat_assignment": e["seat_assignment"],
                "winner_name": e.get("winner_name"), "reason": e.get("reason"),
                "length": e.get("length", len(moves)), "init": init, "moves": moves,
            })
        fn = f"othello__{a}__vs__{b}.json"
        _write(out_dir, fn, {"game": "othello", "a": a, "b": b, "size": size,
                             "episodes": episodes})
        manifest.append({"file": fn, "a": a, "b": b})
    _write(out_dir, "manifest.json", {"game": "othello", "size": size, "pairs": manifest})
    if mismatches:
        print(f"  WARNING: {mismatches} othello episodes failed the piece-count check")
    _report("othello", out_dir, len(manifest))


# --- leduc -----------------------------------------------------------------
def build_leduc():
    game = "leduc_poker"
    gdir = os.path.join(EXP, game)
    if not os.path.isdir(gdir):
        print(f"skip leduc: no {gdir}"); return
    out_dir = _out_dir(game, "leduc")
    manifest = []
    for pd in _pairs(gdir):
        a, b = os.path.basename(pd).split("__vs__")
        episodes = []
        for e in _eps(pd):
            moves = []
            for s in e.get("steps", []):
                pub = s["observation"]["public"]
                th = _think(s); inv = bool(s.get("invalid"))
                moves.append({
                    "ply": s["step"], "player": s["player"], "agent": s["agent_name"],
                    "round": pub.get("round"), "action": s.get("selected_action"),
                    "pot": pub.get("pot"), "to_call": pub.get("to_call"),
                    "public_card": pub.get("public_card"),
                    "invalid": inv, "latency_ms": _meta(s, "latency_ms"),
                    "tokens": _meta(s, "completion_tokens"),
                    "thinking": th, "trunc": _trunc(th, inv),
                })
            episodes.append({
                "episode": e["episode"], "seat_assignment": e["seat_assignment"],
                "cards": e.get("cards"), "public_card": e.get("public_card"),
                "pot": e.get("pot"), "returns": e["returns"],
                "winner": e.get("winner"), "winner_name": e.get("winner_name"),
                "reason": e.get("reason"), "length": e.get("length", len(moves)),
                "moves": moves,
            })
        fn = f"leduc__{a}__vs__{b}.json"
        _write(out_dir, fn, {"game": "leduc", "a": a, "b": b, "episodes": episodes})
        manifest.append({"file": fn, "a": a, "b": b})
    _write(out_dir, "manifest.json", {"game": "leduc", "pairs": manifest})
    _report("leduc", out_dir, len(manifest))


# --- blotto ----------------------------------------------------------------
def build_blotto():
    game = "repeated_colonel_blotto"
    gdir = os.path.join(EXP, game)
    if not os.path.isdir(gdir):
        print(f"skip blotto: no {gdir}"); return
    out_dir = _out_dir(game, "blotto")
    manifest = []
    for pd in _pairs(gdir):
        a, b = os.path.basename(pd).split("__vs__")
        episodes = []
        for e in _eps(pd):
            # truncation flag per (round, seat) from that step's reasoning
            tr = {}
            for s in e.get("steps", []):
                rnd = s["observation"]["public"].get("round")
                seat = 0 if s["player"] == "player_0" else 1
                tr[(rnd, seat)] = _trunc(_think(s), bool(s.get("invalid")))
            rounds = []
            for rh in e.get("round_history", []):
                rounds.append({
                    "round": rh["round"], "alloc_0": rh["alloc_0"], "alloc_1": rh["alloc_1"],
                    "battlefields": rh["battlefields"],
                    "points_0": rh["points_0"], "points_1": rh["points_1"],
                    "cumulative": rh["cumulative"],
                    "trunc_0": tr.get((rh["round"], 0), False),
                    "trunc_1": tr.get((rh["round"], 1), False),
                })
            episodes.append({
                "episode": e["episode"], "seat_assignment": e["seat_assignment"],
                "winner_name": e.get("winner_name"), "final_scores": e.get("final_scores"),
                "battlefield_values": e.get("battlefield_values"),
                "rounds_played": e.get("rounds_played", len(rounds)), "rounds": rounds,
            })
        fn = f"blotto__{a}__vs__{b}.json"
        _write(out_dir, fn, {"game": "blotto", "a": a, "b": b, "episodes": episodes})
        manifest.append({"file": fn, "a": a, "b": b})
    _write(out_dir, "manifest.json", {"game": "blotto", "pairs": manifest})
    _report("blotto", out_dir, len(manifest))


# --- blackjack (per model vs dealer) ---------------------------------------
def build_blackjack():
    game = "independent_blackjack"
    gdir = os.path.join(EXP, game)
    if not os.path.isdir(gdir):
        print(f"skip blackjack: no {gdir}"); return
    out_dir = _out_dir(game, "blackjack")
    manifest = []
    for pd in sorted(glob.glob(os.path.join(gdir, "*__vs__dealer"))):
        model = os.path.basename(pd).split("__vs__")[0]
        episodes = []
        for e in _eps(pd):
            moves = []
            for s in e.get("steps", []):
                pub = s["observation"].get("public") or {}
                priv = s["observation"].get("private") or {}
                th = _think(s); inv = bool(s.get("invalid"))
                moves.append({
                    "ply": s["step"],
                    "actor": "player" if s["player"] == "player_0" else "dealer",
                    "action": s.get("selected_action"),
                    "hand": priv.get("your_hand"), "total": priv.get("your_total"),
                    "soft": priv.get("soft"), "upcard": pub.get("dealer_upcard"),
                    "can_double": pub.get("can_double"),
                    "invalid": inv, "latency_ms": _meta(s, "latency_ms"),
                    "tokens": _meta(s, "completion_tokens"),
                    "thinking": th, "trunc": _trunc(th, inv),
                })
            episodes.append({
                "episode": e["episode"], "net": e["returns"].get("player_0"),
                "player_total": e.get("player_total"), "dealer_total": e.get("dealer_total"),
                "player_natural": e.get("player_natural"), "player_bust": e.get("player_bust"),
                "dealer_natural": e.get("dealer_natural"), "dealer_bust": e.get("dealer_bust"),
                "doubled": e.get("doubled"), "moves": moves,
            })
        fn = f"blackjack__{model}.json"
        _write(out_dir, fn, {"game": "blackjack", "model": model, "episodes": episodes})
        manifest.append({"file": fn, "model": model})
    _write(out_dir, "manifest.json", {"game": "blackjack", "models": manifest})
    _report("blackjack", out_dir, len(manifest))


def main():
    build_othello()
    build_leduc()
    build_blotto()
    build_blackjack()


if __name__ == "__main__":
    main()
