"""Build compact per-match replay data for the board-game viewers.

The tournament data JSON is authoritative for structure (episodes, winners,
seat assignment, the move actually played) but does NOT carry the models'
chain-of-thought. The full thinking lives in each match's match.jsonl as
``response.raw_output`` in the form::

    ===== thinking =====
    <reasoning>
    ===== answer =====
    <final answer>

So we drive off the data JSON and splice the thinking in from the jsonl, keyed
by (episode, step). Output goes under runs/board_tournament/replays/<game>/:

  manifest.json                 -- light index for the dropdowns
  <a>__vs__<b>.json             -- one file per pairing, loaded on demand

The 58MB tournament JSON would be miserable to load whole; per-pairing files
keep each fetch to a few MB. Board state is reconstructed in the browser from
the move list, so we only store moves + thinking here:

  - Connect Four: each move is a column id (``col``); the viewer applies gravity.
  - Gomoku: each move is a coordinate like "E5" -> (row, col) (``rc``); the
    viewer places the stone directly (no gravity).

Both variants seed the board with opening pieces (``random_open=2``), so step
0's observation is NOT an empty grid; we store it as ``init`` and the viewer
reconstructs from there, or the winning line won't line up.
"""

from __future__ import annotations

import json
import os

DATA_DIR = "runs/board_tournament"
GAMES = {"connect4": {"need": 4}, "gomoku": {"need": 5}}
THINK_MARK = "===== thinking ====="
ANSWER_MARK = "===== answer ====="
_GOMOKU_COLS = "ABCDEFGHI"

# Truncation detection. These are reasoning models on a 16,384-token budget; a
# generation that exhausts it stops mid-thought. Two fingerprints:
#   - separate-reasoning models (gpt-oss): the answer comes back empty, the move
#     is logged `invalid`, and the runner plays a fallback.
#   - inline-reasoning models (minimax, kimi, ...): the reasoning IS the output,
#     so it's cut off mid-sentence and the move parser scrapes whatever token
#     sits in the truncated last line — a "valid" but essentially random move.
#     The invalid flag never trips, so length is the only tell.
# This content is coordinate-heavy (~1 char/token), so the cap bites near 16k
# chars; empirically 92% of >=15k-char traces end with no terminal punctuation.
TRUNC_MIN_CHARS = 15000
_TERMINAL = '.!?")]'


def _truncated(thinking: str, invalid: bool) -> bool:
    if invalid:
        return True
    t = (thinking or "").rstrip()
    return len(t) >= TRUNC_MIN_CHARS and (not t or t[-1] not in _TERMINAL)


def _coord_to_rc(coord):
    """'E5' -> [row, col] for the 9x9 Gomoku board, or None if malformed."""
    if not isinstance(coord, str):
        return None
    s = coord.strip().upper().replace("-", "")
    if len(s) < 2 or s[0] not in _GOMOKU_COLS or not s[1:].isdigit():
        return None
    r, c = int(s[1:]) - 1, _GOMOKU_COLS.index(s[0])
    return [r, c] if 0 <= r < 9 and 0 <= c < 9 else None


def _split_thinking(raw: str) -> str:
    """Pull the reasoning out of a raw_output blob.

    Reasoning models that expose a dedicated CoT field get the marker block;
    we return the text between the markers. Models that inline reasoning in the
    answer have no markers — we return the whole blob (it's all reasoning + the
    answer), which is still what a reader wants to see.
    """
    if not raw:
        return ""
    if THINK_MARK in raw:
        body = raw.split(THINK_MARK, 1)[1]
        return body.split(ANSWER_MARK, 1)[0].strip()
    return raw.strip()


def _thinking_lookup(pair_dir: str) -> dict:
    """{(episode, step): thinking} for one match.jsonl."""
    path = os.path.join(pair_dir, "match.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("record_type") != "step":
                continue
            raw = (o.get("response") or {}).get("raw_output") or ""
            out[(o["episode"], o["step"])] = _split_thinking(raw)
    return out


def _encode_move(game: str, s: dict, thinking: str) -> dict:
    invalid = bool(s.get("invalid"))
    mv = {
        "ply": s["step"],
        "player": s["player"],
        "agent": s["agent_name"],
        "invalid": invalid,
        "latency_ms": (s.get("response") or {}).get("metadata", {}).get("latency_ms"),
        "thinking": thinking,
        "trunc": _truncated(thinking, invalid),
    }
    if game == "connect4":
        try:
            mv["col"] = int(s["selected_action"])
        except (TypeError, ValueError):
            mv["col"] = None
    else:  # gomoku
        mv["coord"] = s["selected_action"]
        mv["rc"] = _coord_to_rc(s["selected_action"])
    return mv


def build_game(game: str, need: int):
    data = json.load(open(os.path.join(DATA_DIR, f"{game}_data.json")))
    sample = data["games"][0]["episodes"][0]["steps"][0]["observation"]["public"]["board"]
    rows, cols = len(sample), len(sample[0])

    out_dir = os.path.join(DATA_DIR, "replays", game)
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for g in data["games"]:
        a, b = g["a"], g["b"]
        think = _thinking_lookup(os.path.join(DATA_DIR, f"{game}__{a}__vs__{b}"))

        episodes, man_eps = [], []
        for e in g["episodes"]:
            init_board = (e["steps"][0]["observation"]["public"]["board"]
                          if e["steps"] else [[None] * cols for _ in range(rows)])
            moves = [_encode_move(game, s, think.get((e["episode"], s["step"]), ""))
                     for s in e["steps"]]
            episodes.append({
                "episode": e["episode"],
                "winner_name": e.get("winner_name"),
                "reason": e.get("reason"),
                "length": e["length"],
                "seat_assignment": e["seat_assignment"],
                "returns": e["returns"],
                "init": init_board,
                "moves": moves,
            })
            man_eps.append({
                "i": e["episode"], "winner": e.get("winner_name"),
                "length": e["length"],
                "first": e["steps"][0]["agent_name"] if e["steps"] else None,
            })

        fname = f"{game}__{a}__vs__{b}.json"
        json.dump({"game": game, "a": a, "b": b, "rows": rows, "cols": cols,
                   "need": need, "episodes": episodes},
                  open(os.path.join(out_dir, fname), "w", encoding="utf-8"))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "episodes": man_eps})

    manifest = {"game": game, "rows": rows, "cols": cols, "need": need,
                "pairs": manifest_pairs}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))

    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    largest = max(os.path.getsize(os.path.join(out_dir, p["file"])) for p in manifest_pairs)
    print(f"[{game}] wrote {len(manifest_pairs)} pairings + manifest to {out_dir} "
          f"({total/1e6:.1f} MB total · largest pair {largest/1e6:.1f} MB)")


def main():
    for game, cfg in GAMES.items():
        path = os.path.join(DATA_DIR, f"{game}_data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        build_game(game, cfg["need"])


if __name__ == "__main__":
    main()
