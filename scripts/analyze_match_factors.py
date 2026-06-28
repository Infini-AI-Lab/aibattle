"""Match-specific factors for the Hold'em Match report (no equity needed).

Writes reports/match_factors.json:
  win_type  : per model, how its matches resolve — won by busting the opponent
              vs won at the hand cap (leading), and lost by being busted vs lost
              at the cap. The "two ways to win a match" split.
  by_depth  : per model, aggression by effective stack depth (deep / mid / short
              = push-fold) — does it shift gears correctly as stacks get shallow?
  by_lead   : per model, aggression when ahead vs behind on chips (front-running
              vs comeback behaviour).
"""
from __future__ import annotations
import json, glob, collections

DATA_GLOB = "runs/holdem_match/*__vs__*/ep*.json"
OUT = "reports/match_factors.json"
BB = 2
CAP = 30   # hands per match (max_hands); lead trajectory is tracked over these
EXCLUDE = {"gpt-oss-120b"}   # dropped for an incomplete schedule (see the match report)
AGG = ("bet", "raise", "all_in")
PASSIVE = ("call", "check")

def norm(m): return m.replace("-coached", "")

def depth_bucket(eff_bb):
    if eff_bb >= 40: return "deep"
    if eff_bb >= 15: return "mid"
    return "short"

def main():
    win = collections.defaultdict(lambda: collections.Counter())
    by_depth = collections.defaultdict(lambda: {b: [0, 0] for b in ("deep", "mid", "short")})
    by_lead = collections.defaultdict(lambda: {b: collections.Counter() for b in ("ahead", "behind", "even")})
    # lead trajectory: per model, per hand index 0..CAP-1 -> [ahead_weight, matches].
    # A match that ends early (a bust) carries its final stacks forward to hand CAP,
    # so being ahead at the last hand equals winning the match.
    lead_traj = collections.defaultdict(lambda: [[0.0, 0] for _ in range(CAP)])
    # match-length distribution: per ending hand 1..CAP, split by how it ended.
    length = {"bust": [0] * CAP, "cap": [0] * CAP, "total": 0}
    files = glob.glob(DATA_GLOB)
    for fi, f in enumerate(files):
        e = json.load(open(f))
        sa = {k: norm(v) for k, v in e["seat_assignment"].items()}
        if EXCLUDE & set(sa.values()):
            continue
        models = list(sa.values())
        winner = e.get("winner_name")
        winner = norm(winner) if winner else None
        reason = e.get("reason")          # 'bust' or 'max_hands'
        # --- win_type: classify the match outcome for each model ---
        for m in models:
            win[m]["matches"] += 1
            if winner is None:
                win[m]["draw"] += 1
            elif m == winner:
                win[m]["bust_win" if reason == "bust" else "cap_win"] += 1
            else:
                win[m]["lost_bust" if reason == "bust" else "lost_cap"] += 1
        # --- match length: which hand the match ended on, by ending reason ---
        hp = e.get("hands_played") or len(e.get("hand_summaries") or [])
        if 1 <= hp <= CAP:
            length["total"] += 1
            length["bust" if reason == "bust" else "cap"][hp - 1] += 1
        # --- lead trajectory: ahead-on-chips share after each hand (carry final fwd) ---
        seats = list(sa.keys())
        if len(seats) == 2:
            s0, s1 = seats
            summaries = e.get("hand_summaries") or []
            last = None
            for h in range(CAP):
                if h < len(summaries):
                    sat = summaries[h].get("stacks_after") or last
                    if sat:
                        last = sat
                sat = last
                if not sat:
                    continue
                c0, c1 = sat.get(s0, 0), sat.get(s1, 0)
                for seat, opp in ((s0, c1), (s1, c0)):
                    me = sat.get(seat, 0)
                    cell = lead_traj[sa[seat]][h]
                    cell[0] += 1.0 if me > opp else (0.5 if me == opp else 0.0)
                    cell[1] += 1
        # --- by depth + by lead: per decision ---
        for s in e["steps"]:
            m = norm(s.get("agent_name", "")); act = s.get("selected_action")
            if not act:
                continue
            pub = s.get("observation", {}).get("public", {})
            lead = pub.get("match_lead")
            # by lead: aggression (over bet/raise/call/check) + fold-to-bet (any
            # action while facing a bet) — how it changes gears ahead vs behind.
            if lead is not None:
                lb = "ahead" if lead > 0 else ("behind" if lead < 0 else "even")
                c = by_lead[m][lb]
                if act in AGG + PASSIVE:
                    c["agg"] += int(act in AGG); c["n"] += 1
                if pub.get("to_call", 0) > 0:
                    c["faced"] += 1
                    if act == "fold":
                        c["fold"] += 1
            # by depth: aggression bucketed by effective stack (bet/raise/call/check)
            if act in AGG + PASSIVE:
                yc = pub.get("match_your_chips"); oc = pub.get("match_opp_chips")
                if yc is not None and oc is not None:
                    b = depth_bucket(min(yc, oc) / BB)
                    d = by_depth[m][b]; d[0] += int(act in AGG); d[1] += 1
        if fi % 1000 == 0:
            print(f"  {fi}/{len(files)}", flush=True)

    out_win = {m: dict(c) for m, c in win.items()}
    out_depth = {m: {b: {"agg": round(v[0] / v[1], 3) if v[1] else None, "n": v[1]}
                     for b, v in d.items()} for m, d in by_depth.items()}
    out_lead = {m: {b: {"agg": round(c["agg"] / c["n"], 3) if c["n"] else None, "n": c["n"],
                        "fold": round(c["fold"] / c["faced"], 3) if c["faced"] else None,
                        "faced": c["faced"]}
                    for b, c in d.items()} for m, d in by_lead.items()}
    out_traj = {m: [round(cell[0] / cell[1] * 100, 1) if cell[1] else None for cell in cells]
                for m, cells in lead_traj.items()}
    json.dump({"win_type": out_win, "by_depth": out_depth, "by_lead": out_lead,
               "lead_trajectory": out_traj, "match_length": length},
              open(OUT, "w"), indent=2)
    print(f"WROTE {OUT}: {len(out_win)} models")

if __name__ == "__main__":
    main()
