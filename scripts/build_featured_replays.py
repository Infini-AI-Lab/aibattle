"""Curate a small set of representative games for the replay viewers.

The per-game replay viewer can step through *every* pairing, which is noisy.
This picks ~10-20 representative games (data-driven from the raw runs/ logs, so
it stays reproducible) and writes a small committed JSON the viewer loads into a
"Featured game" dropdown. The viewer still keeps its full matchup/match
selectors below, so "browse everything" stays available.

v1 covers Hold'em Match -> reports/match_featured.json.

    PYTHONPATH=src:scripts python scripts/build_featured_replays.py
"""
from __future__ import annotations

import glob
import json
import os

from analyze_match_tournament import _replay_file_for_pair_dir, EXCLUDE_HOLDEM
from model_names import strip_coached, display_name

REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# Built replay tree per game (relative to REPORT_DIR, the served root). The
# manifest here is authored by build_replays.py and is the source of truth for
# what the viewer can actually load — so we validate every featured pick against
# it. NOTE: build_featured_replays.py must run AFTER build_replays.py.
GAME_REPLAY_DIR = {
    "match": "runs/holdem_match/replays/match",
    "holdem": "runs/holdem_1hand/replays/holdem",
    "connect4": "runs/connect4/replays/connect4",
    "gomoku": "runs/gomoku/replays/gomoku",
    "othello": "runs/new_games_experiment/othello_lite_6x6/replays/othello",
    "kuhn": "runs/kuhn_poker/replays/kuhn",
    "leduc": "runs/new_games_experiment/leduc_poker/replays/leduc",
    "blotto": "runs/new_games_experiment/repeated_colonel_blotto/replays/blotto",
    "blackjack": "runs/new_games_experiment/independent_blackjack/replays/blackjack",
}


def _make_resolver(game):
    """Return resolve(raw_a, raw_b, orig_pair, ep) -> the manifest's exact pair
    `file` for that game/episode, or None if it was never built.

    Curation reads the rich raw runs/ logs, but build_replays.py may name pairs
    differently (it keeps the ``-coached`` suffix for some games and strips it
    for others) and may build only a subset of pairs/episodes. Matching against
    the manifest makes the featured ``pair`` field always equal what the viewer
    looks up, and silently drops picks for games that weren't built.
    """
    rd = os.path.join(REPORT_DIR, GAME_REPLAY_DIR[game])
    mp = os.path.join(rd, "manifest.json")
    if not os.path.exists(mp):
        print(f"  WARN: manifest missing for {game} ({mp}); "
              "featured links left UNVALIDATED — run build_replays.py first")
        return None
    man = json.load(open(mp))
    files = {p["file"] for p in man["pairs"]}
    by_raw, by_stripped = {}, {}
    for p in man["pairs"]:
        by_raw[frozenset((p["a"], p["b"]))] = p["file"]
        by_stripped.setdefault(
            frozenset((strip_coached(p["a"]), strip_coached(p["b"]))), []).append(p["file"])
    cache = {}

    def eps_of(fname):
        if fname not in cache:
            ids = set()
            try:
                d = json.load(open(os.path.join(rd, fname)))
                for e in d.get("episodes") or []:
                    v = e.get("episode", e.get("i")) if isinstance(e, dict) else e
                    if v is not None:
                        ids.add(str(v))
            except (OSError, json.JSONDecodeError):
                pass
            cache[fname] = ids
        return cache[fname]

    def resolve(raw_a, raw_b, orig_pair, ep):
        # Candidate files, most-specific first: the original name (holdem reps,
        # board), the exact raw model pair (keeps the coached variant), then the
        # stripped model pair (manifests that drop -coached, e.g. kuhn/leduc).
        cand = []
        if orig_pair in files:
            cand.append(orig_pair)
        f = by_raw.get(frozenset((raw_a, raw_b)))
        if f and f not in cand:
            cand.append(f)
        for f in by_stripped.get(frozenset((strip_coached(raw_a), strip_coached(raw_b))), []):
            if f not in cand:
                cand.append(f)
        if not cand:
            return None
        known = False
        for f in cand:                       # prefer the variant holding this episode
            e = eps_of(f)
            if e:
                known = True
                if str(ep) in e:
                    return f
        # The episode wasn't among the built ones. If we *know* the built set
        # (non-empty), the pick is unviewable -> drop so curation backfills.
        # Only fail open (return the pair) when we couldn't read any episode list.
        return None if known else cand[0]

    return resolve


def _validate(cs, game, ep_key):
    """Drop candidates whose pair/episode was not built, and rewrite each
    surviving candidate's ``pair`` to the manifest's exact file name."""
    resolve = _make_resolver(game)
    if resolve is None:
        return cs
    out = []
    for c in cs:
        f = resolve(c.get("raw_a", ""), c.get("raw_b", ""), c["pair"], c[ep_key])
        if f:
            out.append(dict(c, pair=f))
    if len(out) != len(cs):
        print(f"  [{game}] {len(out)}/{len(cs)} candidates resolve to built replays")
    return out


def _match_candidates():
    cands = []
    for pd in sorted(glob.glob("runs/holdem_match/*__vs__*")):
        if not os.path.isdir(pd):
            continue
        a_raw, b_raw = os.path.basename(pd).split("__vs__")
        a, b = strip_coached(a_raw), strip_coached(b_raw)
        if {a, b} & EXCLUDE_HOLDEM:
            continue
        pair_file = _replay_file_for_pair_dir(pd, a, b)
        for f in glob.glob(os.path.join(pd, "ep*.json")):
            try:
                e = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            hs = e.get("hand_summaries") or []
            winner = strip_coached(e.get("winner_name") or "")
            if not hs or not winner:
                continue
            sa = {k: strip_coached(v) for k, v in e["seat_assignment"].items()}
            wseat = next((k for k, v in sa.items() if v == winner), None)
            if wseat is None:
                continue
            oseat = "player_1" if wseat == "player_0" else "player_0"
            margins = [(h["stacks_after"].get(wseat, 0) - h["stacks_after"].get(oseat, 0))
                       for h in hs if h.get("stacks_after")]
            if not margins:
                continue
            cands.append({
                "pair": pair_file, "match": e["episode"],
                "raw_a": a_raw, "raw_b": b_raw,
                "a": a, "b": b, "winner": winner, "loser": sa[oseat],
                "reason": e.get("reason"), "hands": e.get("hands_played") or len(hs),
                "min_margin": min(margins), "final_margin": margins[-1],
                "ahead_all": all(m > 0 for m in margins),
            })
    return cands


def _curate_match():
    cs = _validate(_match_candidates(), "match", "match")
    if not cs:
        return []
    picks, used, won = [], set(), set()

    def take(pool, key, title, why_fn):
        cand = max((c for c in pool if (c["pair"], c["match"]) not in used),
                   key=key, default=None)
        if cand is None:
            return
        used.add((cand["pair"], cand["match"])); won.add(cand["winner"])
        picks.append({"pair": cand["pair"], "match": cand["match"],
                      "title": title,
                      "label": f"{title} — {display_name(cand['a'])} vs {display_name(cand['b'])}",
                      "why": why_fn(cand)})

    cap = max(c["hands"] for c in cs)
    take([c for c in cs if c["min_margin"] < 0], lambda c: -c["min_margin"],
         "Biggest comeback",
         lambda c: f"{display_name(c['winner'])} falls {-c['min_margin']} chips behind, then wins.")
    take([c for c in cs if {c["a"], c["b"]} == {"gpt-5.5", "gpt-5.4"}], lambda c: c["hands"],
         "Clash of the top two",
         lambda c: f"The two strongest models, {c['hands']} hands; {display_name(c['winner'])} wins.")
    take([c for c in cs if c["reason"] == "bust"], lambda c: -c["hands"],
         "Quick cooler",
         lambda c: f"{display_name(c['winner'])} busts {display_name(c['loser'])} in {c['hands']} hands.")
    take([c for c in cs if c["ahead_all"]], lambda c: c["min_margin"],
         "Wire-to-wire",
         lambda c: f"{display_name(c['winner'])} leads from hand 1 to last (+{c['final_margin']}).")
    take([c for c in cs if c["hands"] >= cap and c["final_margin"] > 0],
         lambda c: -c["final_margin"],
         "Down to the wire",
         lambda c: f"A full {c['hands']}-hand grind decided by {c['final_margin']} chips.")

    # Model diversity: feature each not-yet-shown model's most watchable win.
    # Restricted to multi-hand games (>=2) so the set isn't padded with one-hand
    # busts — the single short game is already covered by "Quick cooler" above.
    by_winner = {}
    for c in cs:
        if c["hands"] >= 2:
            by_winner.setdefault(c["winner"], []).append(c)
    MAX = 15
    for m in sorted(by_winner, key=lambda m: -len(by_winner[m])):
        if len(picks) >= MAX:
            break
        if m in won:
            continue
        pool = by_winner[m]
        comebacks = [c for c in pool if c["min_margin"] < 0]
        if comebacks:
            take(comebacks, lambda c: -c["min_margin"], f"{display_name(m)} fights back",
                 lambda c: f"{display_name(c['winner'])} claws back from {-c['min_margin']} chips "
                           f"behind to beat {display_name(c['loser'])}.")
        else:
            take(pool, lambda c: c["final_margin"], f"{display_name(m)} in control",
                 lambda c: f"{display_name(c['winner'])} out-grinds {display_name(c['loser'])}, "
                           f"winning by {c['final_margin']} chips.")
    return picks


def _curate_holdem():
    """Hold'em 1-Hand: each hand is its own game. Pair files carry a __r{rep}
    suffix, so we deep-link by the exact file name + the episode id."""
    cs = []
    for pd in sorted(glob.glob("runs/holdem_1hand/*__vs__*")):
        if not os.path.isdir(pd):
            continue
        base = os.path.basename(pd)                 # a__vs__b__r{rep}
        core = base.rsplit("__r", 1)[0]
        a_raw, b_raw = core.split("__vs__")
        a, b = strip_coached(a_raw), strip_coached(b_raw)
        if {a, b} & EXCLUDE_HOLDEM:
            continue
        for f in glob.glob(os.path.join(pd, "ep*.json")):
            try:
                e = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            winner = strip_coached(e.get("winner_name") or "")
            ret = e.get("returns") or {}
            if not winner:
                continue
            sa = {k: strip_coached(v) for k, v in e["seat_assignment"].items()}
            wseat = next((k for k, v in sa.items() if v == winner), None)
            if wseat is None:
                continue
            cs.append({"pair": base + ".json", "ep": e["episode"],
                       "raw_a": a_raw, "raw_b": b_raw,
                       "a": a, "b": b, "winner": winner,
                       "loser": sa["player_1" if wseat == "player_0" else "player_0"],
                       "pot": abs(ret.get(wseat, 0)), "reason": e.get("reason")})
    cs = _validate(cs, "holdem", "ep")
    if not cs:
        return []
    picks, used, won = [], set(), set()

    def take(pool, key, title, why_fn):
        c = max((x for x in pool if (x["pair"], x["ep"]) not in used), key=key, default=None)
        if c is None:
            return
        used.add((c["pair"], c["ep"])); won.add(c["winner"])
        picks.append({"pair": c["pair"], "ep": c["ep"], "title": title,
                      "label": f"{title} — {display_name(c['a'])} vs {display_name(c['b'])}",
                      "why": why_fn(c)})

    take(cs, lambda c: c["pot"], "Biggest pot",
         lambda c: f"{display_name(c['winner'])} wins a {c['pot']:.0f}-chip pot off {display_name(c['loser'])}.")
    take([c for c in cs if c["reason"] == "fold"], lambda c: c["pot"], "Big fold forced",
         lambda c: f"{display_name(c['winner'])} bets {display_name(c['loser'])} off a {c['pot']:.0f}-chip pot.")
    by_winner = {}
    for c in cs:
        by_winner.setdefault(c["winner"], []).append(c)
    for m in sorted(by_winner, key=lambda m: -len(by_winner[m])):
        if len(picks) >= 13:
            break
        if m in won:
            continue
        take(by_winner[m], lambda c: c["pot"], f"{display_name(m)}'s biggest win",
             lambda c: f"{display_name(c['winner'])} stacks {display_name(c['loser'])} for {c['pot']:.0f} chips.")
    return picks


def _curate_generic(game, glob_pat, prefix, kind):
    """Curate board / kuhn / leduc / blotto games. Pairs are validated and
    rewritten to the manifest's exact file name via _validate(game)."""
    cs = []
    for pd in sorted(glob.glob(glob_pat)):
        if not os.path.isdir(pd):
            continue
        base = os.path.basename(pd)
        core = base[len(prefix):] if prefix else base   # strip "game__" for board
        if "__vs__" not in core:
            continue
        a_raw, b_raw = core.split("__vs__")
        a, b = strip_coached(a_raw), strip_coached(b_raw)
        for f in glob.glob(os.path.join(pd, "ep*.json")):
            try:
                e = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            winner = strip_coached(e.get("winner_name") or "")
            if not winner:                # skip draws / ties
                continue
            sa = {k: strip_coached(v) for k, v in e["seat_assignment"].items()}
            wseat = next((k for k, v in sa.items() if v == winner), None)
            if wseat is None:
                continue
            ret = e.get("returns") or {}
            cs.append({"pair": core, "ep": e["episode"], "raw_a": a_raw, "raw_b": b_raw,
                       "a": a, "b": b, "winner": winner,
                       "loser": sa["player_1" if wseat == "player_0" else "player_0"],
                       "length": e.get("length") or 0, "pot": abs(ret.get(wseat, 0) or 0)})
    cs = _validate(cs, game, "ep")
    if not cs:
        return []
    picks, used, won = [], set(), set()

    def take(pool, key, title, why_fn):
        c = max((x for x in pool if (x["pair"], x["ep"]) not in used), key=key, default=None)
        if c is None:
            return
        used.add((c["pair"], c["ep"])); won.add(c["winner"])
        picks.append({"pair": c["pair"], "ep": c["ep"], "title": title,
                      "label": f"{title} — {display_name(c['a'])} vs {display_name(c['b'])}",
                      "why": why_fn(c)})

    if kind == "poker":
        take(cs, lambda c: c["pot"], "Biggest pot",
             lambda c: f"{display_name(c['winner'])} wins a {c['pot']:.0f}-chip pot off {display_name(c['loser'])}.")
    elif kind == "board":
        take(cs, lambda c: c["length"], "Longest battle",
             lambda c: f"A {c['length']}-move war; {display_name(c['winner'])} edges {display_name(c['loser'])}.")

    by = {}
    for c in cs:
        by.setdefault(c["winner"], []).append(c)
    for m in sorted(by, key=lambda m: -len(by[m])):
        if len(picks) >= 13:
            break
        if m in won:
            continue
        if kind == "poker":
            take(by[m], lambda c: c["pot"], f"{display_name(m)}'s biggest win",
                 lambda c: f"{display_name(c['winner'])} stacks {display_name(c['loser'])} for {c['pot']:.0f} chips.")
        else:
            take(by[m], lambda c: c["length"], f"{display_name(m)}'s win",
                 lambda c: f"{display_name(c['winner'])} beats {display_name(c['loser'])} ({c['length']} moves).")
    return picks


def _curate_blackjack():
    """Blackjack is solitaire vs the dealer, so the manifest is keyed by model
    (no opponent). Curated from the *built* per-model files, so pair/episode
    always resolve. ~10 watchable hands across models."""
    src = os.path.join(REPORT_DIR, GAME_REPLAY_DIR["blackjack"])
    man_path = os.path.join(src, "manifest.json")
    if not os.path.exists(man_path):
        print(f"  WARN: no built blackjack manifest at {man_path}")
        return []
    man = json.load(open(man_path))
    cs = []
    for entry in man.get("models", []):
        model = strip_coached(entry["model"])
        try:
            d = json.load(open(os.path.join(src, entry["file"])))
        except (OSError, json.JSONDecodeError):
            continue
        for e in d.get("episodes", []):
            def flag(k):
                return str(e.get(k)).lower() == "true"
            cs.append({"pair": entry["file"], "ep": str(e.get("episode")),
                       "model": model, "net": float(e.get("net") or 0),
                       "doubled": flag("doubled"), "p_nat": flag("player_natural"),
                       "p_bust": flag("player_bust"), "d_bust": flag("dealer_bust"),
                       "dt": e.get("dealer_total"), "pt": e.get("player_total")})
    if not cs:
        return []
    picks, used, who = [], set(), set()

    def take(pool, key, title, why_fn, label=None):
        c = max((x for x in pool if (x["pair"], x["ep"]) not in used), key=key, default=None)
        if c is None:
            return
        used.add((c["pair"], c["ep"])); who.add(c["model"])
        picks.append({"pair": c["pair"], "ep": c["ep"], "title": title,
                      "label": label or f"{title} — {display_name(c['model'])}", "why": why_fn(c)})

    take([c for c in cs if c["doubled"] and c["net"] > 0], lambda c: c["net"],
         "Biggest doubled win",
         lambda c: f"{display_name(c['model'])} doubles down and wins {c['net']:+.1f}.")
    take([c for c in cs if c["p_nat"] and c["net"] > 0], lambda c: c["net"],
         "Natural blackjack",
         lambda c: f"{display_name(c['model'])} is dealt a natural 21 for {c['net']:+.1f}.")
    take([c for c in cs if c["d_bust"] and c["net"] > 0], lambda c: c["net"],
         "Dealer busts",
         lambda c: f"The dealer busts and {display_name(c['model'])} collects {c['net']:+.1f}.")
    take([c for c in cs if c["doubled"] and c["p_bust"]], lambda c: -c["net"],
         "Double gone wrong",
         lambda c: f"{display_name(c['model'])} doubles, busts, and loses {c['net']:+.1f}.")

    by = {}
    for c in cs:
        by.setdefault(c["model"], []).append(c)
    for m in sorted(by, key=lambda m: -len(by[m])):
        if len(picks) >= 10:
            break
        if m in who:
            continue
        take(by[m], lambda c: c["net"], display_name(m),
             lambda c: f"{display_name(c['model'])} beats the dealer for {c['net']:+.1f}.",
             label=display_name(m))
    return picks


GENERIC = [
    ("connect4_featured.json", "connect4", "runs/connect4/connect4__*__vs__*", "connect4__", "board"),
    ("gomoku_featured.json", "gomoku", "runs/gomoku/gomoku__*__vs__*", "gomoku__", "board"),
    ("othello_featured.json", "othello", "runs/new_games_experiment/othello_lite_6x6/*__vs__*", None, "board"),
    ("kuhn_featured.json", "kuhn", "runs/kuhn_poker/*__vs__*", None, "poker"),
    ("leduc_featured.json", "leduc", "runs/new_games_experiment/leduc_poker/*__vs__*", None, "poker"),
    ("blotto_featured.json", "blotto", "runs/new_games_experiment/repeated_colonel_blotto/*__vs__*", None, "blotto"),
]


def _write(name, game, featured):
    out = os.path.join(REPORT_DIR, name)
    json.dump({"game": game, "featured": featured}, open(out, "w"), indent=2)
    print(f"Wrote {out}: {len(featured)} featured games")
    for p in featured:
        print(f"  {p['label']}")


# Committed, self-contained replay data for the public site. The full
# runs/<game>/replays tree is ~3.4 GB (gitignored); here we extract ONLY the
# featured episodes out of their (large) per-pair files, so the viewers work
# straight from the repo without the full data or the reports/runs symlink.
SLIM_DIR = os.path.join(REPORT_DIR, "replays")


def _emit_slim(game, featured):
    """Extract the featured episodes from the built per-pair files into small
    standalone copies under reports/replays/<game>/, with a manifest trimmed to
    only the featured pairs."""
    if not featured:
        return
    src = os.path.join(REPORT_DIR, GAME_REPLAY_DIR[game])
    man_path = os.path.join(src, "manifest.json")
    if not os.path.exists(man_path):
        print(f"  WARN: cannot emit slim {game}; no built manifest at {man_path}")
        return
    man = json.load(open(man_path))
    want = {}                                   # pair file -> {episode ids to keep}
    for f in featured:
        want.setdefault(f["pair"], set()).add(str(f.get("ep", f.get("match"))))

    def _ep_id(e):
        return str(e.get("episode", e.get("i"))) if isinstance(e, dict) else str(e)

    # Heads-up games key the manifest by "pairs"; blackjack (solitaire) by "models".
    entry_key = "pairs" if "pairs" in man else "models"
    out_dir = os.path.join(SLIM_DIR, game)
    os.makedirs(out_dir, exist_ok=True)
    slim_entries, total = [], 0
    for p in man[entry_key]:
        if p["file"] not in want:
            continue
        keep = want[p["file"]]
        d = json.load(open(os.path.join(src, p["file"])))
        d["episodes"] = [e for e in d.get("episodes", []) if _ep_id(e) in keep]
        dst = os.path.join(out_dir, p["file"])
        json.dump(d, open(dst, "w"))
        total += os.path.getsize(dst)
        pp = dict(p)                            # trim the manifest summary too
        if isinstance(p.get("episodes"), list):
            pp["episodes"] = [e for e in p["episodes"] if _ep_id(e) in keep]
        slim_entries.append(pp)
    slim_man = dict(man)
    slim_man[entry_key] = slim_entries
    json.dump(slim_man, open(os.path.join(out_dir, "manifest.json"), "w"))
    print(f"  [{game}] slim replays -> {out_dir} "
          f"({len(slim_entries)} {entry_key}, {total / 1024:.0f} KB)")


def _build(name, game, featured):
    _write(name, game, featured)
    _emit_slim(game, featured)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    _build("match_featured.json", "match", _curate_match())
    _build("holdem_featured.json", "holdem", _curate_holdem())
    _build("blackjack_featured.json", "blackjack", _curate_blackjack())
    for name, game, glob_pat, prefix, kind in GENERIC:
        _build(name, game, _curate_generic(game, glob_pat, prefix, kind))


if __name__ == "__main__":
    main()
