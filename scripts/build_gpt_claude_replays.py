"""Build replay data for the GPT-vs-Claude coached run.

The base build_replays.py targets the canonical per-game run dirs (runs/connect4,
runs/gomoku, runs/holdem_1hand, ...) whose layout differs from this run:
``runs/gpt_vs_claude`` keeps each game in its own subdir with the
aggregate in ``<label>/<label>_data.json`` (keyed by ``pairs``, not ``games``)
and per-episode logs in ``<label>/<a>__vs__<b>/ep*.json``.

So we reuse the pure encoders from build_replays (move/thinking/hole helpers) but
drive them off this run's layout. Output mirrors the base format exactly so the
existing viewers work unchanged (only their fetch BASE differs):

  runs/gpt_vs_claude/replays/<label>/
    manifest.json
    <pair>.json

Run from the repo root:  PYTHONPATH=src python scripts/build_gpt_claude_replays.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_replays as br  # noqa: E402  (pure encoder helpers)

RUN = "runs/gpt_vs_claude"
BOARD = {"connect4": 4, "gomoku": 5}


def _load(label: str) -> dict:
    return json.load(open(os.path.join(RUN, label, f"{label}_data.json")))


def _dump(obj, path: str):
    json.dump(obj, open(path, "w", encoding="utf-8"))


def _report(tag: str, out_dir: str, n: int):
    files = os.listdir(out_dir)
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
    print(f"[{tag}] wrote {n} pairings + manifest to {out_dir} ({total/1e6:.1f} MB)")


def build_board(game: str, need: int):
    data = _load(game)
    sample = data["pairs"][0]["episodes"][0]["steps"][0]["observation"]["public"]["board"]
    rows, cols = len(sample), len(sample[0])
    out_dir = os.path.join(RUN, "replays", game)
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for pr in data["pairs"]:
        a, b = pr["a"], pr["b"]
        think = br._thinking_lookup(os.path.join(RUN, game, f"{a}__vs__{b}"))
        episodes, man_eps = [], []
        for e in pr["episodes"]:
            init_board = (e["steps"][0]["observation"]["public"]["board"]
                          if e["steps"] else [[None] * cols for _ in range(rows)])
            moves = [br._encode_move(game, s, think.get((e["episode"], s["step"]), ""))
                     for s in e["steps"]]
            episodes.append({
                "episode": e["episode"], "winner_name": e.get("winner_name"),
                "reason": e.get("reason"), "length": e["length"],
                "seat_assignment": e["seat_assignment"], "returns": e["returns"],
                "init": init_board, "moves": moves,
            })
            man_eps.append({"i": e["episode"], "winner": e.get("winner_name"),
                            "length": e["length"],
                            "first": e["steps"][0]["agent_name"] if e["steps"] else None})
        fname = f"{game}__{a}__vs__{b}.json"
        _dump({"game": game, "a": a, "b": b, "rows": rows, "cols": cols,
               "need": need, "episodes": episodes}, os.path.join(out_dir, fname))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "episodes": man_eps})

    _dump({"game": game, "rows": rows, "cols": cols, "need": need,
           "pairs": manifest_pairs}, os.path.join(out_dir, "manifest.json"))
    _report(game, out_dir, len(manifest_pairs))


def build_holdem_1hand():
    label = "holdem_1hand"
    data = _load(label)
    out_dir = os.path.join(RUN, "replays", label)
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for pr in data["pairs"]:
        a, b = pr["a"], pr["b"]
        pair_dir = os.path.join(RUN, label, f"{a}__vs__{b}")
        think = br._thinking_lookup(pair_dir)
        prompts = br._prompt_lookup(pair_dir)

        hands, man_hands = [], []
        for e in pr["episodes"]:
            holes = br._dealt_holes(e.get("game", "holdem"), e.get("seed"))
            moves = []
            for s in e["steps"]:
                hole = s["observation"]["private"].get("hole")
                if hole and s["player"] not in holes:
                    holes[s["player"]] = hole
                key = (e["episode"], s["step"])
                moves.append(br._holdem_move(s, think.get(key, ""), prompts.get(key, "")))
            hands.append({
                "episode": e["episode"], "seat_assignment": e["seat_assignment"],
                "holes": holes, "big_blind": e.get("big_blind"),
                "final_board": e.get("final_board", []),
                "hand_categories": e.get("hand_categories"),
                "winner": e.get("winner"), "winner_name": e.get("winner_name"),
                "returns": e["returns"], "reason": e.get("reason"),
                "length": e["length"], "moves": moves,
            })
            man_hands.append({"i": e["episode"], "winner": e.get("winner_name"),
                              "reason": e.get("reason"), "length": e["length"],
                              "returns": e["returns"]})
        # The base holdem viewer labels pairings "… · rep N"; this run has no
        # seat-swap reps (both seatings live in one 100-hand pairing), so pin rep=1.
        fname = f"{a}__vs__{b}.json"
        _dump({"game": "holdem", "a": a, "b": b, "rep": 1, "episodes": hands},
              os.path.join(out_dir, fname))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "rep": 1,
                               "episodes": man_hands})

    _dump({"game": "holdem", "pairs": manifest_pairs},
          os.path.join(out_dir, "manifest.json"))
    _report(label, out_dir, len(manifest_pairs))


def build_holdem_match():
    label = "holdem_match"
    out_dir = os.path.join(RUN, "replays", label)
    os.makedirs(out_dir, exist_ok=True)
    pair_dirs = sorted(d for d in glob.glob(os.path.join(RUN, label, "*__vs__*"))
                       if os.path.isdir(d))

    manifest_pairs = []
    for pd in pair_dirs:
        a, b = os.path.basename(pd).split("__vs__")
        ep_files = sorted(glob.glob(os.path.join(pd, "ep*.json")))
        matches, man_matches = [], []
        for path in ep_files:
            o = json.load(open(path, encoding="utf-8"))
            hsmap = {hs["hand"]: hs for hs in o.get("hand_summaries", [])}
            hands = br._group_by_hand(o["steps"], "match_hand")
            hand_objs = []
            for hno in sorted(hands, key=lambda x: (x is None, x)):
                steps_h = hands[hno]
                moves = [br._match_move(s) for s in steps_h]
                pub0 = steps_h[0]["observation"]["public"]
                p0 = steps_h[0]["player"]
                hs = hsmap.get(hno, {})
                hand_objs.append({
                    "hand": hno, "button": hs.get("button"),
                    "winner": hs.get("winner"), "reason": hs.get("reason"),
                    "deltas": hs.get("deltas"), "stacks_after": hs.get("stacks_after"),
                    "chips_before": {p0: pub0.get("match_your_chips"),
                                     br._other(p0): pub0.get("match_opp_chips")},
                    "big_blind": o.get("big_blind"),
                    "holes": br._holes_from_steps(steps_h),
                    "final_board": max((m["board"] for m in moves), key=len, default=[]),
                    "moves": moves,
                })
            matches.append({
                "episode": o["episode"], "seat_assignment": o["seat_assignment"],
                "winner": o.get("winner"), "winner_name": o.get("winner_name"),
                "returns": o.get("returns"), "final_stacks": o.get("final_stacks"),
                "hands_played": o.get("hands_played"), "reason": o.get("reason"),
                "max_hands": o.get("max_hands"), "big_blind": o.get("big_blind"),
                "hands": hand_objs,
            })
            man_matches.append({"i": o["episode"], "winner": o.get("winner_name"),
                                "hands": o.get("hands_played"), "reason": o.get("reason")})
        fname = f"match__{a}__vs__{b}.json"
        _dump({"game": "match", "a": a, "b": b, "episodes": matches},
              os.path.join(out_dir, fname))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "episodes": man_matches})

    _dump({"game": "match", "pairs": manifest_pairs},
          os.path.join(out_dir, "manifest.json"))
    _report(label, out_dir, len(manifest_pairs))


def main():
    for game, need in BOARD.items():
        build_board(game, need)
    build_holdem_1hand()
    build_holdem_match()


if __name__ == "__main__":
    main()
