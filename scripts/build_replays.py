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
by (episode, step). Output goes under runs/<game>/replays/<game>/:

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

import glob
import json
import os
import random
import sys

# Hole cards are only logged in observations of players who actually acted, so a
# hand that ends on an immediate fold leaves the non-acting winner face-down.
# The deal is fully deterministic from the per-hand seed, though, so we re-deal
# from the game's own logic to recover BOTH players' cards for the viewer. This
# couples build_holdem to the aibattle package; if it isn't importable we fall
# back to the observation-derived holes (acting players only).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from aibattle.games.registry import make_game
except Exception:  # pragma: no cover - degrade gracefully
    make_game = None

# Coached per-game folders. Board games each have their own folder, so the
# board data dir is derived per-game (runs/<game>) inside build_game().
HOLDEM_DIR = "runs/holdem_1hand"
GAMES = {"connect4": {"need": 4}, "gomoku": {"need": 5}}
THINK_MARK = "===== thinking ====="
ANSWER_MARK = "===== answer ====="
_GOMOKU_COLS = "ABCDEFGHI"
_PLAYERS = ("player_0", "player_1")

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
    """{(episode, step): thinking} for one match.

    The chain-of-thought lives in ``response.raw_output`` and is dropped from the
    aggregate tournament JSON, so we read it back from the per-match logs. New
    runs write one ``ep<NNN>.json`` file per episode; older runs wrote a single
    ``match.jsonl``. Support both, preferring the per-episode files.
    """
    out = {}
    ep_files = sorted(glob.glob(os.path.join(pair_dir, "ep*.json")))
    if ep_files:
        for path in ep_files:
            try:
                o = json.load(open(path, encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            ep = o.get("episode")
            for s in o.get("steps", []):
                raw = (s.get("response") or {}).get("raw_output") or ""
                out[(ep, s["step"])] = _split_thinking(raw)
        return out
    path = os.path.join(pair_dir, "match.jsonl")
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


def _prompt_lookup(pair_dir: str) -> dict:
    """{(episode, step): prompt} for one match.

    Like the chain-of-thought, the exact input prompt (``response.prompt``,
    recorded only by v2 runs) is trimmed from the aggregate tournament JSON, so
    read it back from the per-episode ``ep<NNN>.json`` files. Returns {} for v1
    runs (no prompt field) or when the files are missing.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(pair_dir, "ep*.json"))):
        try:
            o = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ep = o.get("episode")
        for s in o.get("steps", []):
            pr = (s.get("response") or {}).get("prompt")
            if pr:
                out[(ep, s["step"])] = pr
    return out


def _encode_move(game: str, s: dict, thinking: str) -> dict:
    invalid = bool(s.get("invalid"))
    mv = {
        "ply": s["step"],
        "player": s["player"],
        "agent": s["agent_name"],
        "invalid": invalid,
        "latency_ms": (s.get("response") or {}).get("metadata", {}).get("latency_ms"),
        "tokens": (s.get("response") or {}).get("metadata", {}).get("completion_tokens"),
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
    data_dir = os.path.join("runs", game)
    data = json.load(open(os.path.join(data_dir, f"{game}_data.json")))
    sample = data["games"][0]["episodes"][0]["steps"][0]["observation"]["public"]["board"]
    rows, cols = len(sample), len(sample[0])

    out_dir = os.path.join(data_dir, "replays", game)
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for g in data["games"]:
        a, b = g["a"], g["b"]
        think = _thinking_lookup(os.path.join(data_dir, f"{game}__{a}__vs__{b}"))

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


def _other(p):
    return _PLAYERS[1 - _PLAYERS.index(p)]


def _dealt_holes(game: str, seed) -> dict:
    """Re-deal a hand from its seed to recover both players' hole cards.

    The deal is deterministic (``initial_state`` shuffles a full deck with
    ``random.Random(seed)``), so this reproduces the exact hole cards every
    player was dealt — including one who folded without acting and was never
    captured in an observation. Returns {player: [card, card]} or {} if the
    game package isn't importable or the seed is missing.
    """
    if make_game is None or seed is None:
        return {}
    try:
        st = make_game(game).initial_state(random.Random(seed))
        return {p: list(c) for p, c in st.hole.items()}
    except Exception:
        return {}


def _holdem_move(s: dict, thinking: str, prompt: str = "") -> dict:
    """One betting action with the table state the player saw when deciding."""
    pub = s["observation"]["public"]
    me = s["player"]
    invalid = bool(s.get("invalid"))
    return {
        "ply": s["step"],
        "player": me,
        "agent": s["agent_name"],
        "action": s["selected_action"],
        "amount": s.get("selected_amount"),
        "pos": pub.get("position"),
        "street": pub.get("street"),
        "board": pub.get("board", []),
        "pot": pub.get("pot"),
        "to_call": pub.get("to_call"),
        # observation is from the actor's POV; normalise stacks to seat ids
        "stacks": {me: pub.get("your_stack"), _other(me): pub.get("opp_stack")},
        "invalid": invalid,
        "latency_ms": (s.get("response") or {}).get("metadata", {}).get("latency_ms"),
        "tokens": (s.get("response") or {}).get("metadata", {}).get("completion_tokens"),
        "thinking": thinking,
        # v2 runs record the exact prompt string sent to the model; v1 has none.
        # The aggregate JSON trims it, so it is recovered from the ep files and
        # passed in (see _prompt_lookup).
        "prompt": prompt or None,
        "trunc": _truncated(thinking, invalid),
    }


def build_holdem():
    """Heads-up Hold'em: one file per pairing+rep; each hand is an episode.

    The reasoning lives in match.jsonl (the tournament JSON drops it); hole cards
    are gathered per player from their own observations (an actor sees only its
    own hole), so a player who folds preflop without acting stays face-down.
    """
    path = os.path.join(HOLDEM_DIR, "tournament_data.json")
    if not os.path.exists(path):
        print(f"skip holdem: no data at {path}")
        return
    data = json.load(open(path))
    out_dir = os.path.join(HOLDEM_DIR, "replays", "holdem")
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for g in data["games"]:
        a, b, rep = g["a"], g["b"], g["rep"]
        pair_dir = os.path.join(HOLDEM_DIR, f"{a}__vs__{b}__r{rep}")
        think = _thinking_lookup(pair_dir)
        prompts = _prompt_lookup(pair_dir)

        hands, man_hands = [], []
        for e in g["episodes"]:
            # Authoritative full deal from the seed (both players, even a folder
            # who never acted); fall back to observation holes if unavailable.
            holes = _dealt_holes(e.get("game", "holdem"), e.get("seed"))
            moves = []
            for s in e["steps"]:
                hole = s["observation"]["private"].get("hole")
                if hole and s["player"] not in holes:
                    holes[s["player"]] = hole
                key = (e["episode"], s["step"])
                moves.append(_holdem_move(s, think.get(key, ""), prompts.get(key, "")))
            hands.append({
                "episode": e["episode"],
                "seat_assignment": e["seat_assignment"],
                "holes": holes,
                "big_blind": e.get("big_blind"),
                "final_board": e.get("final_board", []),
                "hand_categories": e.get("hand_categories"),
                "winner": e.get("winner"),
                "winner_name": e.get("winner_name"),
                "returns": e["returns"],
                "reason": e.get("reason"),
                "length": e["length"],
                "moves": moves,
            })
            man_hands.append({
                "i": e["episode"], "winner": e.get("winner_name"),
                "reason": e.get("reason"), "length": e["length"],
                "returns": e["returns"],
            })

        fname = f"{a}__vs__{b}__r{rep}.json"
        json.dump({"game": "holdem", "a": a, "b": b, "rep": rep, "episodes": hands},
                  open(os.path.join(out_dir, fname), "w", encoding="utf-8"))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "rep": rep,
                               "episodes": man_hands})

    manifest = {"game": "holdem", "pairs": manifest_pairs}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    largest = max(os.path.getsize(os.path.join(out_dir, p["file"])) for p in manifest_pairs)
    print(f"[holdem] wrote {len(manifest_pairs)} pairings + manifest to {out_dir} "
          f"({total/1e6:.1f} MB total · largest {largest/1e6:.1f} MB)")


MATCH_DIR = "runs/holdem_match"
TABLE_DIR = "runs/holdem_table"


def _think_of(s: dict) -> str:
    return _split_thinking((s.get("response") or {}).get("raw_output") or "")


def _meta(s: dict, key: str):
    return (s.get("response") or {}).get("metadata", {}).get(key)


def _group_by_hand(steps: list, key: str) -> "dict":
    """Steps in play order grouped by their hand index, preserving order."""
    hands = {}
    for s in steps:
        h = s["observation"]["public"].get(key)
        hands.setdefault(h, []).append(s)
    return hands


def _holes_from_steps(steps: list) -> dict:
    """{player: [card, card]} gathered from each actor's own observation.

    Heads-up/ring hands only reveal an actor's own hole, so a player who never
    acts (e.g. folds out of the blinds without a decision) stays face-down.
    """
    holes = {}
    for s in steps:
        hole = s["observation"]["private"].get("hole")
        if hole and s["player"] not in holes:
            holes[s["player"]] = hole
    return holes


def _match_move(s: dict) -> dict:
    """One betting action in a heads-up match hand (carried stacks)."""
    pub = s["observation"]["public"]
    me = s["player"]
    invalid = bool(s.get("invalid"))
    thinking = _think_of(s)
    return {
        "ply": s["step"], "player": me, "agent": s["agent_name"],
        "action": s["selected_action"], "amount": s.get("selected_amount"),
        "pos": pub.get("position"), "street": pub.get("street"),
        "board": pub.get("board", []), "pot": pub.get("pot"),
        "to_call": pub.get("to_call"),
        # in-hand chip stacks (carry across hands in a match)
        "stacks": {me: pub.get("your_stack"), _other(me): pub.get("opp_stack")},
        # match-level chip totals at the start of this hand
        "chips": {me: pub.get("match_your_chips"), _other(me): pub.get("match_opp_chips")},
        "invalid": invalid, "latency_ms": _meta(s, "latency_ms"),
        "tokens": _meta(s, "completion_tokens"),
        "thinking": thinking, "trunc": _truncated(thinking, invalid),
    }


def build_match():
    """Heads-up MATCH mode: one file per pairing; a match is many hands with
    carried stacks. The aggregate match_data.json drops steps, so we read the
    per-episode ep*.json files (thinking lives inline in response.raw_output)."""
    pair_dirs = sorted(d for d in glob.glob(os.path.join(MATCH_DIR, "*__vs__*"))
                       if os.path.isdir(d))
    if not pair_dirs:
        print(f"skip match: no pair dirs under {MATCH_DIR}")
        return
    out_dir = os.path.join(MATCH_DIR, "replays", "match")
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for pd in pair_dirs:
        a, b = os.path.basename(pd).split("__vs__")
        ep_files = sorted(glob.glob(os.path.join(pd, "ep*.json")))
        matches, man_matches = [], []
        for path in ep_files:
            o = json.load(open(path, encoding="utf-8"))
            hsmap = {hs["hand"]: hs for hs in o.get("hand_summaries", [])}
            hands = _group_by_hand(o["steps"], "match_hand")
            hand_objs = []
            for hno in sorted(hands, key=lambda x: (x is None, x)):
                steps_h = hands[hno]
                moves = [_match_move(s) for s in steps_h]
                pub0 = steps_h[0]["observation"]["public"]
                p0 = steps_h[0]["player"]
                hs = hsmap.get(hno, {})
                hand_objs.append({
                    "hand": hno, "button": hs.get("button"),
                    "winner": hs.get("winner"), "reason": hs.get("reason"),
                    "deltas": hs.get("deltas"), "stacks_after": hs.get("stacks_after"),
                    "chips_before": {p0: pub0.get("match_your_chips"),
                                     _other(p0): pub0.get("match_opp_chips")},
                    "big_blind": o.get("big_blind"),
                    "holes": _holes_from_steps(steps_h),
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
        json.dump({"game": "match", "a": a, "b": b, "episodes": matches},
                  open(os.path.join(out_dir, fname), "w", encoding="utf-8"))
        manifest_pairs.append({"file": fname, "a": a, "b": b, "episodes": man_matches})

    json.dump({"game": "match", "pairs": manifest_pairs},
              open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"[match] wrote {len(manifest_pairs)} pairings + manifest to {out_dir} "
          f"({total/1e6:.1f} MB total)")


def _table_move(s: dict) -> dict:
    """One betting action at a 5-handed table. The per-step `seats` snapshot
    carries every player's live stack/status/committed, so the viewer can draw
    the whole table from each move alone."""
    pub = s["observation"]["public"]
    me = s["player"]
    invalid = bool(s.get("invalid"))
    thinking = _think_of(s)
    return {
        "ply": s["step"], "player": me, "agent": s["agent_name"],
        "action": s["selected_action"], "amount": s.get("selected_amount"),
        "street": pub.get("street"), "board": pub.get("board", []),
        "pot": pub.get("pot"), "to_call": pub.get("to_call"),
        "button": pub.get("button"), "seats": pub.get("seats"),
        "invalid": invalid, "latency_ms": _meta(s, "latency_ms"),
        "tokens": _meta(s, "completion_tokens"),
        "thinking": thinking, "trunc": _truncated(thinking, invalid),
    }


def build_table():
    """Multi-agent TABLE mode: one file per session (a session = up to N hands
    among all 5 models, scored by finishing rank). Steps come from the
    per-episode ep*.json files under runs/holdem_table/table/."""
    ep_files = sorted(glob.glob(os.path.join(TABLE_DIR, "table", "ep*.json")))
    if not ep_files:
        print(f"skip table: no ep files under {TABLE_DIR}/table")
        return
    out_dir = os.path.join(TABLE_DIR, "replays", "table")
    os.makedirs(out_dir, exist_ok=True)

    man_sessions = []
    for path in ep_files:
        o = json.load(open(path, encoding="utf-8"))
        seat = o["seat_assignment"]
        hsmap = {hs["hand"]: hs for hs in o.get("hand_summaries", [])}
        hands = _group_by_hand(o["steps"], "table_hand")
        hand_objs = []
        for hno in sorted(hands, key=lambda x: (x is None, x)):
            steps_h = hands[hno]
            moves = [_table_move(s) for s in steps_h]
            hs = hsmap.get(hno, {})
            hand_objs.append({
                "hand": hno,
                "button": hs.get("button") or (moves[0]["button"] if moves else None),
                "winner": hs.get("winner"), "reason": hs.get("reason"),
                "deltas": hs.get("deltas"), "stacks_after": hs.get("stacks_after"),
                "holes": _holes_from_steps(steps_h),
                "final_board": max((m["board"] for m in moves), key=len, default=[]),
                "moves": moves,
            })
        session = {
            "episode": o["episode"], "seat_assignment": seat,
            "num_players": o.get("num_players"), "ranking": o.get("ranking"),
            "rank_of": o.get("rank_of"), "final_stacks": o.get("final_stacks"),
            "bust_order": o.get("bust_order"), "hands_played": o.get("hands_played"),
            "reason": o.get("reason"), "hands": hand_objs,
        }
        fname = f"session{o['episode']:03d}.json"
        json.dump(session, open(os.path.join(out_dir, fname), "w", encoding="utf-8"))
        man_sessions.append({
            "file": fname, "i": o["episode"], "hands": o.get("hands_played"),
            "ranking": [seat.get(p, p) for p in (o.get("ranking") or [])],
        })

    json.dump({"game": "table", "num_players": man_sessions and 5,
               "sessions": man_sessions},
              open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"[table] wrote {len(man_sessions)} sessions + manifest to {out_dir} "
          f"({total/1e6:.1f} MB total)")


def _bare(name):
    """Drop the '-coached' agent suffix; the site shows bare model names."""
    if isinstance(name, str) and name.endswith("-coached"):
        return name[: -len("-coached")]
    return name


def build_kuhn():
    """Heads-up Kuhn poker: one file per pairing, one hand per episode. Steps
    (with reasoning) live inline in kuhn_data.json, so we build straight from it.
    Both private cards are recoverable because each player acts at least once."""
    path = "runs/kuhn_poker/kuhn_data.json"
    if not os.path.exists(path):
        print(f"skip kuhn: no data at {path}")
        return
    data = json.load(open(path))
    out_dir = os.path.join("runs/kuhn_poker", "replays", "kuhn")
    os.makedirs(out_dir, exist_ok=True)

    manifest_pairs = []
    for g in data["pairs"]:
        a, b = _bare(g["a"]), _bare(g["b"])
        episodes = []
        for e in g["episodes"]:
            seat = {k: _bare(v) for k, v in e["seat_assignment"].items()}
            cards, moves = {}, []
            for s in e.get("steps", []):
                me = s["player"]
                card = (s["observation"].get("private") or {}).get("card")
                if card and me not in cards:
                    cards[me] = card
                pub = s["observation"].get("public") or {}
                th = _think_of(s); inv = bool(s.get("invalid"))
                moves.append({
                    "ply": s["step"], "player": me, "agent": _bare(s["agent_name"]),
                    "action": s.get("selected_action"), "pot": pub.get("pot"),
                    "invalid": inv, "latency_ms": _meta(s, "latency_ms"),
                    "tokens": _meta(s, "completion_tokens"),
                    "thinking": th, "trunc": _truncated(th, inv),
                })
            # pot starts at the 2-chip ante and grows 1 per bet and per call
            final_pot = 2 + sum(1 for m in moves if m["action"] in ("bet", "call"))
            reason = "fold" if moves and moves[-1]["action"] == "fold" else "showdown"
            episodes.append({
                "episode": e["episode"], "seat_assignment": seat, "cards": cards,
                "pot": final_pot, "returns": e["returns"], "winner": e.get("winner"),
                "winner_name": _bare(e.get("winner_name")), "reason": reason,
                "length": e.get("length", len(moves)), "moves": moves,
            })
        fname = f"kuhn__{a}__vs__{b}.json"
        json.dump({"game": "kuhn", "a": a, "b": b, "episodes": episodes},
                  open(os.path.join(out_dir, fname), "w", encoding="utf-8"))
        manifest_pairs.append({"file": fname, "a": a, "b": b})

    json.dump({"game": "kuhn", "pairs": manifest_pairs},
              open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"))
    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"[kuhn] wrote {len(manifest_pairs)} pairings + manifest to {out_dir} "
          f"({total/1e6:.1f} MB)")


def main():
    for game, cfg in GAMES.items():
        path = os.path.join("runs", game, f"{game}_data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        build_game(game, cfg["need"])
    build_holdem()
    build_match()
    build_table()
    build_kuhn()


if __name__ == "__main__":
    main()
