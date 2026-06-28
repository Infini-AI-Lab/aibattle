"""Lightweight (no-equity) decision factors for the Hold'em 1-Hand report.

Writes reports/holdem_1hand_factors.json with two analyses that are based purely
on observed actions (no opponent-range assumptions):

  fold_induced : per model, per street, the rate at which the opponent folds to
                 this model's bet/raise/all-in — i.e. how often its pressure
                 actually forces a fold (credible aggression / barreling).
  positional   : per model, VPIP and aggression split by position (button/SB vs
                 BB) — does it open up in position and tighten out of position
                 (a classic skill marker)?

analyze_tournament.py reads this file (graceful fallback if absent).
"""
from __future__ import annotations
import json, glob, collections

DATA_GLOB = "runs/holdem_1hand/*__vs__*/ep*.json"
OUT = "reports/holdem_1hand_factors.json"
STREETS = ("preflop", "flop", "turn", "river")
AGG = ("bet", "raise", "all_in")

def norm(m): return m.replace("-coached", "")

def main():
    # fold-induced: model -> street -> [agg_count, folded_count]
    fi = collections.defaultdict(lambda: {s: [0, 0] for s in STREETS})
    # positional: model -> pos -> {hands, vpip, aggr, passive}
    pos = collections.defaultdict(lambda: {"button/SB": collections.Counter(),
                                           "BB": collections.Counter()})
    # net chips by the street the hand ENDED on: model -> street -> total chips
    nbs = collections.defaultdict(lambda: {s: 0.0 for s in STREETS})
    nhands = collections.Counter()
    for f in glob.glob(DATA_GLOB):
        e = json.load(open(f))
        steps = e["steps"]
        sa = {k: norm(v) for k, v in e["seat_assignment"].items()}
        # --- net chips by ending street (street of the last decision in the hand) ---
        ending = None
        for s in reversed(steps):
            st = s.get("observation", {}).get("public", {}).get("street")
            if st in STREETS:
                ending = st; break
        if ending:
            returns = e.get("returns", {})
            for seat, m in sa.items():
                nbs[m][ending] += returns.get(seat, 0) or 0
                nhands[m] += 1
        # --- fold-induced: pair each aggressive action with opponent's next action ---
        for i, s in enumerate(steps):
            m = norm(s.get("agent_name", "")); act = s.get("selected_action")
            if act not in AGG:
                continue
            st = s.get("observation", {}).get("public", {}).get("street", "?")
            if st not in STREETS:
                continue
            resp = None
            for j in range(i + 1, len(steps)):
                if norm(steps[j].get("agent_name", "")) != m:
                    resp = steps[j].get("selected_action"); break
            fi[m][st][0] += 1
            if resp == "fold":
                fi[m][st][1] += 1
        # --- positional: per hand VPIP + per decision aggression, by position ---
        # determine each model's position + whether it VPIP'd this hand
        first_pos = {}; vpipd = {}
        for s in steps:
            m = norm(s.get("agent_name", "")); pub = s.get("observation", {}).get("public", {})
            p = pub.get("position")
            if p not in ("button/SB", "BB"):
                continue
            if m not in first_pos:
                first_pos[m] = p
            act = s.get("selected_action")
            if pub.get("street") == "preflop" and act in ("call",) + AGG:
                vpipd[m] = True
            # aggression frequency (all streets)
            if act in AGG:
                pos[m][p]["aggr"] += 1
            elif act in ("call", "check"):
                pos[m][p]["passive"] += 1
        for m, p in first_pos.items():
            pos[m][p]["hands"] += 1
            if vpipd.get(m):
                pos[m][p]["vpip"] += 1

    out_fi = {m: {s: round(v[s][1] / v[s][0], 3) if v[s][0] else None for s in STREETS}
              for m, v in fi.items()}
    out_pos = {}
    for m, d in pos.items():
        out_pos[m] = {}
        for p, key in (("button/SB", "button"), ("BB", "bb")):
            c = d[p]; h = c["hands"] or 1; acts = (c["aggr"] + c["passive"]) or 1
            out_pos[m][key] = {"vpip": round(c["vpip"] / h, 3),
                               "aggr": round(c["aggr"] / acts, 3), "hands": c["hands"]}
    out_nbs = {m: {s: round(nbs[m][s] / max(nhands[m], 1), 3) for s in STREETS}
               for m in nbs}
    json.dump({"fold_induced": out_fi, "positional": out_pos, "net_by_street": out_nbs},
              open(OUT, "w"), indent=2)
    print(f"WROTE {OUT}: {len(out_fi)} models")

if __name__ == "__main__":
    main()
