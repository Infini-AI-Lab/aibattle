"""Build compact per-match replay data for the board-game viewer.

The tournament data JSON is authoritative for structure (episodes, winners,
seat assignment, the column actually played) but does NOT carry the models'
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
the column sequence (gravity), so we only store the move list + thinking here.

Currently Connect Four only — Gomoku uses (row,col) coordinates and is left for
a follow-up.
"""

from __future__ import annotations

import json
import os

DATA_DIR = "runs/board_tournament"
GAME = "connect4"
THINK_MARK = "===== thinking ====="
ANSWER_MARK = "===== answer ====="

# Truncation detection. These are reasoning models on a 16,384-token budget; a
# generation that exhausts it stops mid-thought. Two fingerprints:
#   - separate-reasoning models (gpt-oss): the answer comes back empty, the move
#     is logged `invalid`, and the runner plays a fallback.
#   - inline-reasoning models (minimax, kimi, ...): the reasoning IS the output,
#     so it's simply cut off mid-sentence and the column parser scrapes whatever
#     digit sits in the truncated last line — a "valid" but essentially random
#     move. The invalid flag never trips, so length is the only tell.
# This content is coordinate-heavy (~1 char/token), so the cap bites near 16k
# chars; empirically 92% of >=15k-char traces end with no terminal punctuation.
TRUNC_MIN_CHARS = 15000
_TERMINAL = '.!?")]'


def _truncated(thinking: str, invalid: bool) -> bool:
    if invalid:
        return True
    t = (thinking or "").rstrip()
    return len(t) >= TRUNC_MIN_CHARS and (not t or t[-1] not in _TERMINAL)


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


def build():
    data = json.load(open(os.path.join(DATA_DIR, f"{GAME}_data.json")))
    sample = data["games"][0]["episodes"][0]["steps"][0]["observation"]["public"]["board"]
    rows, cols = len(sample), len(sample[0])

    out_dir = os.path.join(DATA_DIR, "replays", GAME)
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for g in data["games"]:
        a, b = g["a"], g["b"]
        pair_dir = os.path.join(DATA_DIR, f"{GAME}__{a}__vs__{b}")
        think = _thinking_lookup(pair_dir)

        episodes = []
        man_eps = []
        for e in g["episodes"]:
            # IMPORTANT: this Connect Four variant seeds the board with opening
            # pieces (one per player) before play, so step 0's observation is
            # NOT an empty grid. The viewer must reconstruct from this initial
            # board, not from scratch, or the position (and the winning line)
            # won't match.
            init_board = (e["steps"][0]["observation"]["public"]["board"]
                          if e["steps"] else [[None] * cols for _ in range(rows)])
            moves = []
            for s in e["steps"]:
                try:
                    col = int(s["selected_action"])
                except (TypeError, ValueError):
                    col = None
                moves.append({
                    "ply": s["step"],
                    "player": s["player"],
                    "agent": s["agent_name"],
                    "col": col,
                    "invalid": bool(s.get("invalid")),
                    "latency_ms": (s.get("response") or {}).get("metadata", {}).get("latency_ms"),
                    "thinking": think.get((e["episode"], s["step"]), ""),
                })
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
            first = e["steps"][0]["agent_name"] if e["steps"] else None
            man_eps.append({
                "i": e["episode"], "winner": e.get("winner_name"),
                "length": e["length"], "first": first,
            })

        fname = f"{GAME}__{a}__vs__{b}.json"
        json.dump({"game": GAME, "a": a, "b": b, "rows": rows, "cols": cols,
                   "episodes": episodes},
                  open(os.path.join(out_dir, fname), "w", encoding="utf-8"))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "episodes": man_eps})

    manifest = {"game": GAME, "rows": rows, "cols": cols, "pairs": manifest_pairs}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))

    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"Wrote {len(manifest_pairs)} pairings + manifest to {out_dir}")
    print(f"  total {total/1e6:.1f} MB · "
          f"largest pair {max(os.path.getsize(os.path.join(out_dir, p['file'])) for p in manifest_pairs)/1e6:.1f} MB")


if __name__ == "__main__":
    build()
