"""Compare base vs coached behavior, pairing each model with its -coached self.

Reads reports/*_analysis.json (base) and reports/coached/*_analysis.json
(coached) and prints per-model deltas for STYLE / ERROR metrics, which are the
ones comparable across the two separate tournament pools. Win-rate and Elo are
pool-relative (coached models only played each other) and are flagged as such.
"""

from __future__ import annotations

import json
import os

BASE = "reports"
COA = "reports/coached"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pm(rep):
    """game-report -> {model: stats} for the per_model-style reports."""
    return rep["per_model"]


def lb(rep):
    """leaderboard-style reports -> {model: row}."""
    return {r["model"]: r for r in rep["leaderboard"]}


def base_name(m):
    return m[:-len("-coached")] if m.endswith("-coached") else m


def pct(x):
    return "  —  " if x is None else f"{x*100:5.1f}%"


def row(label, base_v, coa_v, fmt="pct"):
    if base_v is None or coa_v is None:
        return f"  {label:<16} {'—':>8} {'—':>8} {'':>8}"
    if fmt == "pct":
        b, c = base_v * 100, coa_v * 100
        d = c - b
        return f"  {label:<16} {b:7.1f}% {c:7.1f}% {d:+7.1f}"
    if fmt == "num":
        d = coa_v - base_v
        return f"  {label:<16} {base_v:8.2f} {coa_v:8.2f} {d:+8.2f}"
    if fmt == "int":
        d = coa_v - base_v
        return f"  {label:<16} {base_v:8.0f} {coa_v:8.0f} {d:+8.0f}"


def header(model):
    print(f"\n  {model}")
    print(f"  {'metric':<16} {'base':>8} {'coached':>8} {'Δ':>8}")


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def compare_holdem():
    b, c = load(f"{BASE}/holdem_tournament_analysis.json"), load(f"{COA}/holdem_tournament_analysis.json")
    if not (b and c):
        return
    section("HOLD'EM 1-HAND — playing style (cross-pool comparable)")
    bp, cpm = pm(b), pm(c)
    cmap = {base_name(m): m for m in cpm}
    METR = [("VPIP (looseness)", "vpip"), ("PFR", "pfr"), ("agg freq", "agg_freq"),
            ("all-in freq", "allin_freq"), ("fold-to-bet", "fold_to_bet"),
            ("c-bet rate", "cbet_rate"), ("fold-to-cbet", "fold_to_cbet"),
            ("WTSD", "wtsd"), ("W$SD", "wsd")]
    for m in bp:
        if base_name(m) not in cmap:
            continue
        cm = cmap[base_name(m)]
        header(m)
        print(f"  style: {bp[m]['style']['label']:<18} -> {cpm[cm]['style']['label']}")
        for lab, k in METR:
            print(row(lab, bp[m].get(k), cpm[cm].get(k)))
        print(row("avg bet (xPot)", bp[m].get("avg_bet_xpot"), cpm[cm].get("avg_bet_xpot"), "num"))
        binv = (bp[m]["invalid_actions"] + bp[m]["invalid_amounts"]) / max(1, bp[m]["decisions"])
        cinv = (cpm[cm]["invalid_actions"] + cpm[cm]["invalid_amounts"]) / max(1, cpm[cm]["decisions"])
        print(row("invalid rate", binv, cinv))
        print(row("tokens/dec", bp[m].get("avg_comp_tokens"), cpm[cm].get("avg_comp_tokens"), "int"))


def compare_kuhn():
    b, c = load(f"{BASE}/kuhn_tournament_analysis.json"), load(f"{COA}/kuhn_tournament_analysis.json")
    if not (b and c):
        return
    section("KUHN POKER — error vs GTO (blunder_rate is absolute, GTO-scored)")
    bl, cl = lb(b), lb(c)
    cmap = {base_name(m): m for m in cl}
    for m in bl:
        if base_name(m) not in cmap:
            continue
        cm = cmap[base_name(m)]
        br, cr = bl[m], cl[cm]
        header(m)
        print(row("blunder rate", br.get("blunder_rate"), cr.get("blunder_rate")))
        print(row("bet K (value)", br.get("bet_K"), cr.get("bet_K")))
        print(row("bet Q", br.get("bet_Q"), cr.get("bet_Q")))
        print(row("bet J (bluff)", br.get("bet_J"), cr.get("bet_J")))
        print(row("net/hand", br.get("net_per_hand"), cr.get("net_per_hand"), "num"))
        print(row("invalid rate", br.get("invalid_rate"), cr.get("invalid_rate")))
        print(row("tokens/dec", br.get("avg_tokens"), cr.get("avg_tokens"), "int"))


def compare_board(game):
    b, c = load(f"{BASE}/{game}_analysis.json"), load(f"{COA}/{game}_analysis.json")
    if not (b and c):
        return
    section(f"{game.upper()} — tactics & errors (block/invalid/blunder comparable)")
    bp, cp = pm(b), pm(c)
    cmap = {base_name(m): m for m in cp}
    for m in bp:
        if base_name(m) not in cmap:
            continue
        cm = cmap[base_name(m)]
        header(m)
        print(row("invalid rate", bp[m].get("invalid_rate"), cp[cm].get("invalid_rate")))
        print(row("block rate", bp[m].get("block_rate"), cp[cm].get("block_rate")))
        print(row("win-take rate", bp[m].get("win_take_rate"), cp[cm].get("win_take_rate")))
        print(row("missed wins", bp[m].get("missed_wins"), cp[cm].get("missed_wins"), "int"))
        print(row("allowed losses", bp[m].get("allowed_losses"), cp[cm].get("allowed_losses"), "int"))
        print(row("avg game len", bp[m].get("avg_len"), cp[cm].get("avg_len"), "num"))
        print(row("latency (s)", bp[m].get("avg_latency_s"), cp[cm].get("avg_latency_s"), "num"))


def compare_match():
    b, c = load(f"{BASE}/match_tournament_analysis.json"), load(f"{COA}/match_tournament_analysis.json")
    if not (b and c):
        return
    section("HOLD'EM MATCH — behavior (⚠ coached run is sparse; treat as indicative)")
    bb, cb = b.get("behavior", {}), c.get("behavior", {})
    cmap = {base_name(m): m for m in cb}
    METR = [("VPIP", "vpip"), ("PFR", "pfr"), ("agg freq", "aggression"),
            ("all-in freq", "allin_rate"), ("fold-to-bet", "fold_to_bet"),
            ("WTSD", "wtsd"), ("W$SD", "wsd")]
    for m in bb:
        if base_name(m) not in cmap:
            continue
        cm = cmap[base_name(m)]
        header(m)
        for lab, k in METR:
            print(row(lab, bb[m].get(k), cb[cm].get(k)))


if __name__ == "__main__":
    compare_holdem()
    compare_kuhn()
    compare_board("connect4")
    compare_board("gomoku")
    compare_match()
    print("\nNote: win-rate / Elo / net-per-hand are POOL-RELATIVE (coached models")
    print("only played coached opponents), so cross-pool gaps reflect the pool, not")
    print("a head-to-head improvement. Style & error rates above are comparable.")
