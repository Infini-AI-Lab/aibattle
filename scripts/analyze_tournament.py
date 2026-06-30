"""Analyze the tournament data and emit a self-contained interactive HTML report.

Reads runs/holdem_1hand/tournament_data.json (games -> episodes -> steps) and
computes per-model poker behavior + results, then writes runs/holdem_1hand/report.html.

Behavior metrics (the "player personality"):
  - VPIP   : % of hands the model voluntarily put chips in preflop (looseness)
  - PFR    : % of hands it raised preflop (preflop aggression)
  - AggFreq: (bet+raise+all_in) / (bet+raise+all_in+call+check)  (aggression)
  - fold-to-bet, all-in freq, action mix, avg bet size (xPot), avg latency
Results: chip delta, bb/100, win rate, showdown vs fold wins, invalid rates,
plus a head-to-head chip matrix.
"""

from __future__ import annotations

import json
import math
import os
import html as html_lib
from collections import defaultdict

from model_names import strip_coached, display_name, model_cell, output_price
from elo_util import bradley_terry, elo_key, bootstrap_elo, gross_from_records
from report_theme import BASE_CSS, CHART_SETUP
from report_legends import legend as _legend

# Coached is now the canonical (and only) run set; data lives in per-game folders.
DATA = "runs/holdem_1hand/tournament_data.json"
OUT = "runs/holdem_1hand/report.html"
# Tracked copy committed to the repo (runs/ is gitignored).
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")
BB = 2


STREETS = ("preflop", "flop", "turn", "river")
HAND_BUCKETS = ("premium", "strong", "playable", "marginal", "trash")

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# pages include those two files in <head> via NAV_HEAD and the bar is injected
# by JS, so the nav markup lives in one place.
NAV_HEAD = '<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="nav.css?v=7"><script defer src="nav.js?v=32"></script>'
BETSIZE_BUCKETS = ("small", "medium", "pot", "over")
SHOWDOWN_CATS = ("high card", "pair", "two pair", "trips", "straight", "flush",
                 "full house")


def _blank():
    return {
        "hands": 0, "decisions": 0, "chips": 0.0,
        # chips won/lost split by how the hand ended: "red" = at showdown
        # (hand strength), "blue" = without showdown (pressure / fold equity).
        "red_chips": 0.0, "blue_chips": 0.0,
        "wins": 0, "losses": 0, "ties": 0,
        "showdown_wins": 0, "fold_wins": 0,
        "acts": defaultdict(int),
        "facing_bet": 0, "fold_facing_bet": 0,
        "vpip_hands": 0, "pfr_hands": 0,
        "betsize_ratios": [], "latencies": [], "comp_tokens": [],
        "invalid_actions": 0, "invalid_amounts": 0,
        # street-level decisions: street -> {action: count}
        "by_street": {s: defaultdict(int) for s in STREETS},
        # c-bet tracking: as preflop aggressor reaching flop
        "cbet_opps": 0, "cbet_attempts": 0,
        # facing c-bet: as caller-of-preflop-aggressor on flop
        "facing_cbet": 0, "fold_to_cbet": 0,
        # showdown participation
        "saw_showdown": 0, "won_showdown": 0,
        "showdown_hand_cats": defaultdict(int),
        # preflop hand-bucket behavior: bucket -> {open/call/fold counts}
        "pf_bucket": {b: {"open": 0, "call_only": 0, "fold": 0, "all": 0}
                      for b in HAND_BUCKETS},
        # bet sizing histogram (share of bet/raise actions by size bucket)
        "betsize_bins": {b: 0 for b in BETSIZE_BUCKETS},
        # latency samples for scatter (kept in ms then converted)
        "lat_samples_ms": [],
    }


_AMOUNT_REASONS = {"missing_amount", "non_integer_amount", "below_minimum",
                   "above_stack", "unexpected_amount"}

# Chen-formula hand strength for heads-up preflop bucketing.
_RANKS = "23456789TJQKA"
_RANK_VAL = {r: i + 2 for i, r in enumerate(_RANKS)}  # 2..14
_CHEN_HIGH = {"A": 10, "K": 8, "Q": 7, "J": 6, "T": 5, "9": 4.5, "8": 4,
              "7": 3.5, "6": 3, "5": 2.5, "4": 2, "3": 1.5, "2": 1}


def _chen_score(card1: str, card2: str) -> float:
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    if _RANK_VAL[r1] < _RANK_VAL[r2]:
        r1, r2 = r2, r1
    high = _CHEN_HIGH[r1]
    suited = s1 == s2
    pair = r1 == r2
    if pair:
        score = max(high * 2, 5)
    else:
        score = high
        if suited:
            score += 2
        gap = _RANK_VAL[r1] - _RANK_VAL[r2] - 1
        penalty = {0: 0, 1: -1, 2: -2, 3: -4}.get(gap, -5)
        score += penalty
        # straight bonus: connector/1-gap and both lower than Q
        if gap <= 1 and _RANK_VAL[r1] < _RANK_VAL["Q"]:
            score += 1
    # round up to integer per Chen convention
    return int(score + 0.999) if score > 0 else int(score)


def _hand_bucket(card1: str, card2: str) -> str:
    s = _chen_score(card1, card2)
    if s >= 9:
        return "premium"     # AA,KK,QQ,JJ,TT,AKs,AKo,AQs,AJs,KQs
    if s >= 7:
        return "strong"      # 99,AQo,AJo,KQo,ATs,KJs,QJs,strong suited
    if s >= 5:
        return "playable"    # small pairs, suited connectors, suited Ax
    if s >= 3:
        return "marginal"
    return "trash"


def _betsize_bucket(ratio: float) -> str:
    if ratio < 0.5:
        return "small"
    if ratio < 1.0:
        return "medium"
    if ratio < 1.5:
        return "pot"
    return "over"


def analyze(data: dict) -> dict:
    models = data["models"]
    stats = {m: _blank() for m in models}
    h2h = defaultdict(lambda: defaultdict(float))  # h2h[a][b] = net chips a won vs b
    # gross chips each direction per pair; fed to the Bradley-Terry fit as a
    # chip-weighted "win" total so the Elo rewards how *much* you win, not just
    # how often (winning big pots > winning many tiny ones), while staying on the
    # 1500 scale and adjusting for who each model actually played.
    h2h_gross = defaultdict(lambda: defaultdict(float))
    h2h_hands = defaultdict(lambda: defaultdict(int))  # hands played each direction
    elo_records = []  # per-hand (a, b, chips_a, chips_b) for the Elo bootstrap
    pair_games = defaultdict(int)

    for g in data["games"]:
        a, b = g["a"], g["b"]
        for e in g["episodes"]:
            seat_name = e["seat_assignment"]            # seat -> model name
            name_seat = {v: k for k, v in seat_name.items()}
            returns = e["returns"]
            reason = e.get("reason")
            winner_name = e.get("winner_name")

            # episode-level: chips, win/loss, head-to-head
            for seat, nm in seat_name.items():
                st = stats[nm]
                st["hands"] += 1
                payoff = returns[seat]
                st["chips"] += payoff
                if reason == "showdown":
                    st["red_chips"] += payoff
                else:
                    st["blue_chips"] += payoff
                if payoff > 0:
                    st["wins"] += 1
                elif payoff < 0:
                    st["losses"] += 1
                else:
                    st["ties"] += 1
            # head-to-head (a's chips vs b)
            ra, rb = returns[name_seat[a]], returns[name_seat[b]]
            h2h[a][b] += ra
            h2h[b][a] += rb
            h2h_hands[a][b] += 1
            h2h_hands[b][a] += 1
            # gross chips each side extracted this hand (the chip-weighted "score")
            h2h_gross[a][b] += max(ra, 0.0)
            h2h_gross[b][a] += max(rb, 0.0)
            elo_records.append((a, b, ra, rb))
            if winner_name and reason == "showdown":
                stats[winner_name]["showdown_wins"] += 1
            elif winner_name and reason == "fold":
                stats[winner_name]["fold_wins"] += 1

            # step-level behavior; track per-(player) preflop voluntary action
            vpip_seen = {}      # name -> bool voluntarily put money in preflop
            pfr_seen = {}       # name -> bool raised preflop
            # preflop-aggressor tracking (last preflop bet/raise/all_in)
            last_pf_aggressor = None
            # per-seat first action on each post-flop street
            first_act_on_street = {}    # (seat, street) -> step
            # per-seat preflop summary for hand-bucket bookkeeping
            seat_pf_actions = {nm: [] for nm in seat_name.values()}
            seat_hole = {}      # nm -> (c1, c2) from first observation

            for s in e["steps"]:
                nm = s["agent_name"]
                seat = s["player"]
                st = stats[nm]
                st["decisions"] += 1
                act = s["selected_action"]
                st["acts"][act] += 1
                pub = s["observation"]["public"]
                priv = s["observation"].get("private") or {}
                to_call = pub.get("to_call", 0)
                street = pub.get("street")
                pot = pub.get("pot", 0) or 0
                sc = pub.get("your_street_commit", 0) or 0

                # per-street action counts
                if street in STREETS:
                    st["by_street"][street][act] += 1

                # remember hole cards (set from first decision)
                if nm not in seat_hole and priv.get("hole"):
                    seat_hole[nm] = tuple(priv["hole"])

                # first action per (seat, street) — used for c-bet detection
                key = (seat, street)
                if key not in first_act_on_street:
                    first_act_on_street[key] = s

                if to_call > 0:
                    st["facing_bet"] += 1
                    if act == "fold":
                        st["fold_facing_bet"] += 1

                if street == "preflop":
                    seat_pf_actions[nm].append(act)
                    if act in ("call", "bet", "raise", "all_in"):
                        vpip_seen[nm] = True
                    if act in ("raise", "all_in", "bet"):
                        pfr_seen[nm] = True
                        last_pf_aggressor = nm

                if act in ("bet", "raise", "all_in"):
                    amt = s.get("selected_amount")
                    invested = (amt - sc) if amt is not None else None
                    if invested is None and act == "all_in":
                        invested = pub.get("your_stack", 0)
                    if invested and pot > 0:
                        ratio = invested / pot
                        st["betsize_ratios"].append(ratio)
                        st["betsize_bins"][_betsize_bucket(ratio)] += 1

                meta = (s.get("response") or {}).get("metadata", {})
                lat = meta.get("latency_ms")
                if lat:
                    st["latencies"].append(lat)
                ctok = meta.get("completion_tokens")
                if isinstance(ctok, (int, float)):
                    st["comp_tokens"].append(ctok)

                if s.get("invalid"):
                    reason_i = (s.get("invalid_info") or {}).get("reason")
                    if reason_i in _AMOUNT_REASONS:
                        st["invalid_amounts"] += 1
                    else:
                        st["invalid_actions"] += 1

            for nm in seat_name.values():
                if vpip_seen.get(nm):
                    stats[nm]["vpip_hands"] += 1
                if pfr_seen.get(nm):
                    stats[nm]["pfr_hands"] += 1

            # ---- preflop hand-bucket bookkeeping ----
            for nm, hole in seat_hole.items():
                if len(hole) != 2:
                    continue
                bucket = _hand_bucket(hole[0], hole[1])
                bk = stats[nm]["pf_bucket"][bucket]
                bk["all"] += 1
                acts_pf = seat_pf_actions.get(nm, [])
                if any(a in ("bet", "raise", "all_in") for a in acts_pf):
                    bk["open"] += 1
                elif "call" in acts_pf:
                    bk["call_only"] += 1
                elif "fold" in acts_pf:
                    bk["fold"] += 1

            # ---- c-bet and fold-to-c-bet (flop only) ----
            if last_pf_aggressor is not None:
                agg_seat = name_seat[last_pf_aggressor]
                opp_seat = "player_1" if agg_seat == "player_0" else "player_0"
                opp_name = seat_name[opp_seat]
                agg_flop = first_act_on_street.get((agg_seat, "flop"))
                if agg_flop is not None:
                    stats[last_pf_aggressor]["cbet_opps"] += 1
                    if agg_flop["selected_action"] in ("bet", "raise", "all_in"):
                        stats[last_pf_aggressor]["cbet_attempts"] += 1
                        # opponent's first response on the flop
                        opp_flop = first_act_on_street.get((opp_seat, "flop"))
                        if opp_flop is not None:
                            stats[opp_name]["facing_cbet"] += 1
                            if opp_flop["selected_action"] == "fold":
                                stats[opp_name]["fold_to_cbet"] += 1

            # ---- showdown participation / W$SD / made-hand categories ----
            if reason == "showdown":
                cats = e.get("hand_categories") or {}
                for seat, nm in seat_name.items():
                    stats[nm]["saw_showdown"] += 1
                    cat = cats.get(seat)
                    if cat:
                        stats[nm]["showdown_hand_cats"][cat] += 1
                if winner_name:
                    stats[winner_name]["won_showdown"] += 1

        pair_games[frozenset((a, b))] += 1

    # finalize per-model derived metrics
    out_models = {}
    for m in models:
        st = stats[m]
        n = max(st["hands"], 1)
        d = max(st["decisions"], 1)
        aggr_acts = st["acts"]["bet"] + st["acts"]["raise"] + st["acts"]["all_in"]
        passive_acts = st["acts"]["call"] + st["acts"]["check"]
        agg_freq = aggr_acts / max(aggr_acts + passive_acts, 1)
        calls = max(st["acts"]["call"], 1)

        # per-street action share (% of that street's actions that were aggressive)
        by_street_out = {}
        for street in STREETS:
            acts = st["by_street"][street]
            total = sum(acts.values())
            aggr = acts["bet"] + acts["raise"] + acts["all_in"]
            passive = acts["call"] + acts["check"]
            by_street_out[street] = {
                "decisions": total,
                "agg_freq": round(aggr / max(aggr + passive, 1), 4),
                "fold_share": round(acts["fold"] / max(total, 1), 4),
                "mix": {k: acts[k] for k in
                        ("fold", "check", "call", "bet", "raise", "all_in")},
            }

        # preflop hand-bucket open rates
        pf_bucket_out = {}
        for b in HAND_BUCKETS:
            bk = st["pf_bucket"][b]
            total = max(bk["all"], 1)
            pf_bucket_out[b] = {
                "hands": bk["all"],
                "open_rate": round(bk["open"] / total, 4),
                "call_rate": round(bk["call_only"] / total, 4),
                "fold_rate": round(bk["fold"] / total, 4),
            }

        # bet-size distribution (share of bet/raise/all_in actions)
        bs_total = max(sum(st["betsize_bins"].values()), 1)
        betsize_dist = {b: round(st["betsize_bins"][b] / bs_total, 4)
                        for b in BETSIZE_BUCKETS}

        # showdown made-hand share
        sd_total = max(sum(st["showdown_hand_cats"].values()), 1)
        showdown_cat_dist = {c: round(st["showdown_hand_cats"][c] / sd_total, 4)
                             for c in SHOWDOWN_CATS}

        out_models[m] = {
            "hands": st["hands"], "decisions": st["decisions"],
            "chips": round(st["chips"], 1),
            "chips_per_hand": round(st["chips"] / n, 3),
            # "where the money comes from" split (per hand): pressure vs showdown.
            "blue_per_hand": round(st["blue_chips"] / n, 3),
            "red_per_hand": round(st["red_chips"] / n, 3),
            "bb_per_100": round((st["chips"] / n) / BB * 100, 2),
            "win_rate": round(st["wins"] / n, 4),
            "wins": st["wins"], "losses": st["losses"], "ties": st["ties"],
            "showdown_wins": st["showdown_wins"], "fold_wins": st["fold_wins"],
            "vpip": round(st["vpip_hands"] / n, 4),
            "pfr": round(st["pfr_hands"] / n, 4),
            "agg_freq": round(agg_freq, 4),
            "agg_factor": round((st["acts"]["bet"] + st["acts"]["raise"]) / calls, 3),
            "fold_to_bet": round(st["fold_facing_bet"] / max(st["facing_bet"], 1), 4),
            "allin_freq": round(st["acts"]["all_in"] / d, 4),
            "avg_bet_xpot": round(sum(st["betsize_ratios"]) / len(st["betsize_ratios"]), 3)
                            if st["betsize_ratios"] else 0.0,
            "avg_latency_s": round(sum(st["latencies"]) / len(st["latencies"]) / 1000, 1)
                             if st["latencies"] else 0.0,
            "avg_comp_tokens": round(sum(st["comp_tokens"]) / len(st["comp_tokens"]))
                               if st["comp_tokens"] else 0,
            "invalid_actions": st["invalid_actions"],
            "invalid_amounts": st["invalid_amounts"],
            "action_mix": {k: st["acts"][k] for k in
                           ("fold", "check", "call", "bet", "raise", "all_in")},
            "style": _style(st["vpip_hands"] / n, agg_freq, st["acts"]["all_in"] / d),
            # new
            "by_street": by_street_out,
            "cbet_rate": round(st["cbet_attempts"] / max(st["cbet_opps"], 1), 4),
            "cbet_opps": st["cbet_opps"],
            "fold_to_cbet": round(st["fold_to_cbet"] / max(st["facing_cbet"], 1), 4),
            "facing_cbet": st["facing_cbet"],
            "wtsd": round(st["saw_showdown"] / n, 4),
            "wsd": round(st["won_showdown"] / max(st["saw_showdown"], 1), 4),
            "saw_showdown": st["saw_showdown"],
            "won_showdown": st["won_showdown"],
            "pf_bucket": pf_bucket_out,
            "betsize_dist": betsize_dist,
            "showdown_cats": showdown_cat_dist,
        }
    # Per-hand net chips (hand counts differ across pairs once the field is
    # assembled in waves, so totals aren't comparable — normalize by hands played).
    # per-hand net chips; None for pairs that never played (so the report shows
    # a dash instead of a misleading +0.00 "draw").
    h2h_out = {a: {b: (round(h2h[a][b] / h2h_hands[a][b], 3) if h2h_hands[a][b] else None)
                   for b in models if b != a} for a in models}
    # Chip-weighted Elo: a Bradley-Terry fit fed each pair's gross chips won in
    # each direction (instead of hand counts). Magnitude counts — a model that
    # wins big pots outrates one that wins more small ones — and the fit adjusts
    # for opponent strength, so it stays fair when the round-robin is incomplete.
    chip_wld = {a: {b: (h2h_gross[a][b], h2h_gross[b][a], 0.0)
                    for b in models if b != a} for a in models}
    _, elo = bradley_terry(models, chip_wld)
    elo_ci = bootstrap_elo(models, elo_records, lambda s: gross_from_records(models, s))
    for m in models:
        out_models[m]["elo"] = elo[m]
        out_models[m]["elo_sd"] = elo_ci[m]["sd"]
    return {"models": models, "per_model": out_models, "h2h": h2h_out,
            "elo": elo, "elo_ci": elo_ci,
            "hands_per_game": data.get("hands"), "num_games": len(data["games"])}


def _style(vpip: float, agg: float, allin: float) -> dict:
    loose = vpip >= 0.5
    aggressive = agg >= 0.45
    label = (("Loose" if loose else "Tight") + "-" +
             ("Aggressive" if aggressive else "Passive"))
    tags = []
    if allin >= 0.15:
        tags.append("shove-happy")
    if not loose and not aggressive:
        tags.append("nit")
    if loose and aggressive:
        tags.append("LAG")
    if not loose and aggressive:
        tags.append("TAG")
    if loose and not aggressive:
        tags.append("calling station")
    return {"label": label, "tags": tags}


# ---------------------------------------------------------------------------
def _percentile(values, q):
    vals = sorted(values)
    if not vals:
        return 0
    k = (len(vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def _bucket_vpip(s, bucket):
    r = s["pf_bucket"][bucket]
    return r["open_rate"] + r["call_rate"]


def _street_aggr(s, street):
    return s["by_street"][street]["agg_freq"]


def _strategy_analysis(report: dict, ranked: list) -> tuple[str, str]:
    """Signal-count model cards for the Analysis section.

    Scores are diagnostic leak evidence: each triggered signal adds one point.
    Higher values mean "more to review", not "better strategic skill".
    """
    pm = report["per_model"]
    # Six axes chosen so the shape itself explains winning/losing: two style axes
    # (loose, aggressive), two "where the money comes from" axes (pressure vs
    # showdown chips), and two showdown-quality axes (win rate, restraint).
    dims = [
        ("vpip", "Loose", "Share of hands it voluntarily plays (VPIP). Further out = looser; centre = "
         "tight. Loose only profits when backed by aggression."),
        ("agg_freq", "Aggressive", "How often it bets/raises instead of calling or checking. Further out "
         "= takes the initiative; centre = passive, drifting to showdown as the underdog."),
        ("blue_per_hand", "Pressure $", "Chips won per hand WITHOUT showdown — by folding opponents out. "
         "Further out = its bets really make opponents fold (how GPT 5.5 earns)."),
        ("red_per_hand", "Showdown $", "Chips won per hand AT showdown. Further out = shows down stronger "
         "hands; deep inside = pays off second-best hands — the top reason models lose."),
        ("wsd", "Showdown win%", "Of hands that reach showdown, the share it wins (W$SD). Further out = "
         "shows down strong; centre = drags weak hands to showdown and loses them."),
        ("restraint", "Restraint", "How often it avoids showdown (1 − WTSD). Further out = ends hands "
         "early via pressure or folding; centre = can't lay hands down."),
    ]

    def mval(s, key):
        if key == "restraint":
            return 1.0 - float(s.get("wtsd") or 0)
        return float(s.get(key) or 0)

    def pct(x, digits=0):
        return f"{x * 100:.{digits}f}%"

    def percentile(values):
        vals = sorted(values)
        if len(vals) <= 1:
            return {vals[0]: 50.0} if vals else {}
        return {
            v: 100.0 * sum(1 for x in vals if x <= v) / len(vals)
            for v in vals
        }

    pct_by_metric = {}
    for key, _label, _help in dims:
        values = [mval(pm[m], key) for m in ranked]
        pct_by_metric[key] = percentile(values)

    # scale for the pressure/showdown bars: largest |contribution| in the field.
    bar_scale = max([abs(mval(pm[m], "blue_per_hand")) for m in ranked]
                    + [abs(mval(pm[m], "red_per_hand")) for m in ranked] + [0.1])

    def verdict(s):
        vpip = s.get("vpip", 0); agg = s.get("agg_freq", 0)
        blue = s.get("blue_per_hand", 0); red = s.get("red_per_hand", 0); total = blue + red
        loose = "Loose" if vpip >= 0.62 else ("Tight" if vpip <= 0.50 else "Balanced")
        aggr = "aggressive" if agg >= 0.40 else ("passive" if agg <= 0.30 else "moderate")
        style = f"{loose} &amp; {aggr}"
        if total > 0.05:
            if blue >= red and blue > 0.15:
                tag, why = "Pressure winner", f"wins pots without showdown (pressure {blue:+.2f}/hand)"
            elif red > blue and red > 0.15:
                tag, why = "Value winner", f"shows down stronger hands (showdown {red:+.2f}/hand)"
            else:
                tag, why = "All-round winner", "small edge from both pressure and showdown"
        elif total < -0.05:
            if red <= -0.3:
                tag, why = "Pays off at showdown", f"bleeds at showdown ({red:+.2f}/hand) with second-best hands"
            elif blue <= -0.3:
                tag, why = "Pressure backfires", f"loses without showdown ({blue:+.2f}/hand), out-played pre-showdown"
            else:
                tag, why = "Slightly losing", "just below break-even across the board"
        else:
            tag, why = "Break-even", "roughly neutral"
        return style, tag, why

    def attr_bar(s):
        def row(label, v):
            w = min(abs(v) / bar_scale * 50.0, 50.0)
            cls = "pos" if v >= 0 else "neg"
            side = f"left:50%;width:{w:.1f}%" if v >= 0 else f"right:50%;width:{w:.1f}%"
            return (f"<div class='pl-row'><span class='pl-lbl'>{label}</span>"
                    f"<div class='pl-track'><i class='plf {cls}' style='{side}'></i></div>"
                    f"<span class='pl-val {cls}'>{v:+.2f}</span></div>")
        return ("<div class='pl-attr'>"
                + row("Pressure $/h", s.get("blue_per_hand", 0))
                + row("Showdown $/h", s.get("red_per_hand", 0))
                + "</div>")

    def summarize_profile(s):
        vpip = s.get("vpip", 0)
        pfr = s.get("pfr", 0)
        agg = s.get("agg_freq", 0)
        cbet = s.get("cbet_rate", 0)
        wsd = s.get("wsd", 0)
        allin = s.get("allin_freq", 0)

        if vpip >= 0.68:
            entry = "very loose preflop"
        elif vpip >= 0.56:
            entry = "loose preflop"
        elif vpip <= 0.44:
            entry = "selective preflop"
        else:
            entry = "moderately selective preflop"

        if pfr >= 0.36:
            initiative = "takes initiative early"
        elif pfr <= 0.22:
            initiative = "enters more passively"
        else:
            initiative = "has moderate preflop initiative"

        if agg >= 0.42:
            pressure = "keeps pressure high after the flop"
        elif agg <= 0.30:
            pressure = "often shifts into a passive postflop line"
        else:
            pressure = "applies postflop pressure at a balanced rate"

        if cbet >= 0.70:
            continuation = "c-bets frequently"
        elif cbet <= 0.55:
            continuation = "does not always follow through with c-bets"
        else:
            continuation = "has a middle-of-field c-bet pattern"

        if wsd >= 0.52:
            showdown = "converts showdowns well"
        elif wsd <= 0.40:
            showdown = "struggles to convert showdowns"
        else:
            showdown = "has middling showdown conversion"

        if allin >= 0.004:
            risk = "with noticeable all-in risk exposure"
        elif allin <= 0.0015:
            risk = "while avoiding many all-in spots"
        else:
            risk = "with controlled all-in exposure"

        return [entry, initiative, pressure, continuation, showdown, risk]

    chart_payload = []
    cards = []
    glossary = ""
    for key, label, help_text in dims:
        glossary += f"""
      <div class="strategy-gloss">
        <b>{html_lib.escape(label)}</b>
        <span>{html_lib.escape(help_text)}</span>
        <em>Metric: <code>{html_lib.escape(key)}</code></em>
      </div>"""

    def raw_label(key, s):
        if key in ("blue_per_hand", "red_per_hand"):
            return f"{mval(s, key):+.2f}/h"
        return pct(mval(s, key))

    for rank, m in enumerate(ranked, 1):
        s = pm[m]
        raw = {key: mval(s, key) for key, _label, _help in dims}
        scores = [pct_by_metric[key].get(raw[key], 0) for key, _label, _help in dims]
        raw_labels = {label: raw_label(key, s) for key, label, _help in dims}
        chart_payload.append({
            "id": f"strategyRadar{rank}",
            "model": m,
            "label": display_name(m),
            "scores": scores,
            "raw": raw_labels,
        })
        style, tag, why = verdict(s)
        cards.append(f"""
    <article class="strategy-card metric-card">
      <div class="strategy-head">
        <div><h3>{rank}. {model_cell(m)}</h3></div>
        <div class="strategy-kpi {'pos' if s['bb_per_100'] >= 0 else 'neg'}">{s['bb_per_100']:+.1f}<span>bb/100</span></div>
      </div>
      <div class="pl-verdict"><b>{style}</b> · <span class="pl-tag">{tag}</span> — {why}.</div>
      {attr_bar(s)}
      <div class="metric-profile-layout">
        <canvas id="strategyRadar{rank}"></canvas>
      </div>
    </article>""")

    html = f"""
  <div class="strategy-intro">
    <b>Why each player wins or loses.</b> Each card opens with a one-line verdict,
    then the <b>Pressure vs Showdown bar</b> shows where its chips come from —
    <span class="pos">Pressure&nbsp;$</span> is won without showdown (folding
    opponents out), <span class="pos">Showdown&nbsp;$</span> is won at showdown
    (hand strength); a bar pointing left (red) means that line loses money. The
    radar is the player's fingerprint across six axes (radius = field percentile);
    the dashed ring is the field median, so anything bulging past it is a strength
    and anything inside it a weakness.
    <div class="strategy-glossary">{glossary}
    </div>
  </div>
  <div class="strategy-grid">
    {''.join(cards)}
  </div>"""
    js = f"""
const STRATEGY = {json.dumps(chart_payload)};
const STRATEGY_LABELS = {json.dumps([label for _key, label, _help in dims])};
STRATEGY.forEach((card, i) => {{
  const el = document.getElementById(card.id);
  if (!el) return;
  new Chart(el, {{
    type:'radar',
    data:{{ labels:STRATEGY_LABELS,
      datasets:[
        {{ label:'field median', data:[50,50,50,50,50,50], borderColor:'#9aa0a6',
           borderWidth:1, borderDash:[4,3], backgroundColor:'transparent', pointRadius:0 }},
        {{ label:card.label, data:card.scores,
           borderColor:mcol(card.model), backgroundColor:mcol(card.model) + '33',
           pointBackgroundColor:mcol(card.model) }}
      ] }},
    options:{{ scales:{{ r:{{ min:0, max:100, ticks:{{ display:false, stepSize:25 }} }} }},
      plugins:{{ legend:{{ display:false }},
        tooltip:{{ filter:(item) => item.datasetIndex === 1, callbacks:{{ label(ctx) {{
          const raw = card.raw[ctx.label] || '';
          return `percentile ${{ctx.formattedValue}} / raw ${{raw}}`;
        }} }} }} }} }} }});
}});
"""
    return html, js


# ---------------------------------------------------------------------------
def render_html(report: dict) -> str:
    models = report["models"]
    pm = report["per_model"]
    payload = json.dumps(report)
    replay_btn = ('<a class="replaybtn" href="holdem_replay.html?cacheBust=19">'
                  '🎬 Watch featured replays →</a>')

    # ranked leaderboard by chip-weighted Elo; raw metrics kept for reference.
    elo = report.get("elo", {})
    ranked = sorted(models, key=lambda m: (elo_key(elo, m), pm[m]["bb_per_100"]),
                    reverse=True)
    strategy_html, strategy_js = _strategy_analysis(report, ranked)
    # "Where the chips come from" map: pressure (blue) vs showdown (red), per hand.
    quad_pts = [{"m": m, "label": display_name(m), "t": round(pm[m]["chips_per_hand"], 2),
                 "x": pm[m]["blue_per_hand"], "y": pm[m]["red_per_hand"]} for m in ranked]

    # Precomputed action-based factors (fold-induced by street, positional play).
    try:
        _factors = json.load(open(os.path.join(REPORT_DIR, "holdem_1hand_factors.json")))
    except (OSError, json.JSONDecodeError):
        _factors = {}
    fi_data = _factors.get("fold_induced", {})
    pos_data = _factors.get("positional", {})
    nbs_data = _factors.get("net_by_street", {})
    # GPT models first (so the headline contrast reads top-down), then by rank.
    fi_order = ([m for m in ranked if m.startswith("gpt-5")]
                + [m for m in ranked if not m.startswith("gpt-5")])
    # merged by-street table: per street, "fire" (own aggression, amber) next to
    # "fold" (opponent folds to that bet, green) so the pair reads together.
    bystreet_rows = ""
    for m in fi_order:
        if m not in fi_data:
            continue
        bs = pm[m].get("by_street", {})
        cells = ""
        for s in ("preflop", "flop", "turn", "river"):
            agg = bs.get(s, {}).get("agg_freq")
            if agg is None:
                cells += "<td>—</td>"
            else:
                a = min(0.55, max(0.0, agg * 0.55))
                cells += f"<td style='background:rgba(180,83,9,{a:.2f})'>{agg*100:.0f}%</td>"
            fold = fi_data[m].get(s)
            if fold is None:
                cells += "<td class='strk'>—</td>"
            else:
                a = min(0.6, max(0.0, (fold * 100 - 20) / 45 * 0.6))
                cells += f"<td class='strk' style='background:rgba(26,127,55,{a:.2f})'>{fold*100:.0f}%</td>"
        bystreet_rows += f"<tr><td class='model'>{model_cell(m)}</td>{cells}</tr>"
    # net chips per hand attributed to the street the hand ended on (sums to chips/hand)
    nbs_rows = ""
    for m in ranked:
        nb = nbs_data.get(m)
        if not nb:
            continue
        cells = ""
        for s in ("preflop", "flop", "turn", "river"):
            v = nb.get(s, 0)
            alpha = min(0.6, abs(v) / 1.0 * 0.6)
            rgb = "26,127,55" if v >= 0 else "185,28,28"
            cells += f"<td style='background:rgba({rgb},{alpha:.2f})'>{v:+.2f}</td>"
        tot = sum(nb.values())
        nbs_rows += (f"<tr><td class='model'>{model_cell(m)}</td>{cells}"
                     f"<td class='{'pos' if tot>=0 else 'neg'}'><b>{tot:+.2f}</b></td></tr>")
    pos_rows = ""
    for m in ranked:
        p = pos_data.get(m)
        if not p:
            continue
        b, bb = p["button"], p["bb"]
        gap = (b["vpip"] - bb["vpip"]) * 100
        pos_rows += (f"<tr><td class='model'>{model_cell(m)}</td>"
                     f"<td>{b['vpip']*100:.0f}%</td><td>{bb['vpip']*100:.0f}%</td>"
                     f"<td class='{'pos' if gap>0 else ''}'>{gap:+.0f}pp</td>"
                     f"<td>{b['aggr']*100:.0f}%</td><td>{bb['aggr']*100:.0f}%</td></tr>")

    fold_pos_html = ""
    if bystreet_rows:
        fold_pos_html += f"""
  <h2>🔨 Aggression &amp; fold-induced by street</h2>
  <div class="note">Per street, two paired numbers: <b style="color:#b45309">fire</b> = aggression =
    (bet+raise+all-in) ÷ (bet+raise+all-in+call+check), folds excluded (its own aggression, amber);
    <b style="color:#1a7f37">fold</b> = how often the opponent
    then folds to that bet (whether the pressure works, green). Read them together — a model that
    <b>keeps firing AND forces folds into the turn/river</b> (GPT 5.5) is the credible aggressor; one that
    <b>fires but gets called</b> (GPT 5.4 on the river) is bluffing into showdown. GPT models listed first.</div>
  <table class="bystreet">
    <tr><th rowspan="2" class='model'>model</th><th colspan="2">preflop</th><th colspan="2">flop</th>
        <th colspan="2">turn</th><th colspan="2">river</th></tr>
    <tr><th>fire</th><th>fold</th><th>fire</th><th>fold</th><th>fire</th><th>fold</th><th>fire</th><th>fold</th></tr>
    {bystreet_rows}
  </table>
"""
    if nbs_rows:
        fold_pos_html += f"""
  <h2>💰 Net chips by street <span class="note">(where the money is made / lost)</span></h2>
  <div class="note">Each hand's net result is attributed to the street it <b>ended</b> on, then averaged per
    hand — so the four columns add up to the model's <b>chips/hand total</b>. Green = won there, red = lost.
    It shows exactly where a model earns or bleeds: GPT 5.5 books most of its profit on the <b>turn</b>
    (pressure), Kimi on the <b>river</b> (showdown value), GPT 5.4 <b>loses on the river</b> (bluffs get
    called), and Claude Opus bleeds on every street.</div>
  <table>
    <tr><th class='model'>model</th><th>preflop</th><th>flop</th><th>turn</th><th>river</th><th>total</th></tr>
    {nbs_rows}
  </table>
"""
    if pos_rows:
        fold_pos_html += f"""
  <h2>🎯 Positional play <span class="note">(in position vs out of position)</span></h2>
  <div class="note">Heads-up, the <b>button/SB</b> acts last postflop (in position) — strong players open
    up there and tighten in the <b>BB</b> (out of position). A large <b>VPIP gap</b> (button − BB) plus
    more button aggression means it understands position. <b>aggr</b> = (bet+raise+all-in) ÷
    (bet+raise+all-in+call+check), folds excluded.</div>
  <table>
    <tr><th class='model'>model</th><th>button VPIP</th><th>BB VPIP</th><th>VPIP gap</th>
        <th>button aggr</th><th>BB aggr</th></tr>
    {pos_rows}
  </table>
"""

    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        chip_cls = "pos" if s["chips"] > 0 else ("neg" if s["chips"] < 0 else "")
        if s.get("elo") is None:
            elo_disp = "—"
        elif s.get("elo_sd") is not None:
            elo_disp = f"{s['elo']}<div class='small'>±{s['elo_sd']:.0f}</div>"
        else:
            elo_disp = str(s["elo"])
        tags = " ".join(f"<span class='tag'>{t}</span>" for t in s["style"]["tags"])
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{model_cell(m)}</td>
          <td class='stylecell'><div class='slabel'><b>{s['style']['label']}</b></div><div class='stags'>{tags or '&nbsp;'}</div></td>
          <td><b>{elo_disp}</b></td>
          <td class='{chip_cls}'>{s['chips_per_hand']:+.2f}</td>
          <td class='{chip_cls}'>{s['bb_per_100']:+.1f}</td>
          <td>{s['win_rate']*100:.0f}%</td>
          <td>{s['hands']}</td>
          <td>{'—' if (m.startswith('claude') or m.startswith('gpt-5')) else format(s['avg_comp_tokens'], ',')}</td>
          <td>{('—' if (output_price(m) is None or m.startswith('claude') or m.startswith('gpt-5')) else f"${s['avg_comp_tokens']*output_price(m)/1e6*1000:.2f}")}</td>
        </tr>"""

    # head-to-head matrix, ordered by leaderboard rank (strongest first)
    h2h = report["h2h"]
    hh = "<tr><th></th>" + "".join(f"<th>{display_name(m)}</th>" for m in ranked) + "</tr>"
    for a in ranked:
        hh += f"<tr><th class='model'>{model_cell(a)}</th>"
        for b in ranked:
            if a == b:
                hh += "<td class='diag'>—</td>"
            else:
                v = h2h[a].get(b)
                if v is None:
                    hh += "<td class='na'>—</td>"
                else:
                    cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                    hh += f"<td class='{cls}' style='--v:{v}'>{v:+.2f}</td>"
        hh += "</tr>"

    # postflop / showdown table
    pf_rows = ""
    for m in ranked:
        s = pm[m]
        pf_rows += (
            f"<tr><td class='model'>{model_cell(m)}</td>"
            f"<td>{s['cbet_rate']*100:.0f}%</td>"
            f"<td class='small'>{s['cbet_opps']}</td>"
            f"<td>{s['fold_to_cbet']*100:.0f}%</td>"
            f"<td class='small'>{s['facing_cbet']}</td>"
            f"<td>{s['wtsd']*100:.0f}%</td>"
            f"<td>{s['wsd']*100:.0f}%</td>"
            f"<td class='small'>{s['saw_showdown']}/{s['won_showdown']}</td>"
            f"</tr>"
        )

    # preflop hand-bucket table (open-rate matrix: rows = buckets, cols = models)
    bucket_rows = ""
    for b in HAND_BUCKETS:
        bucket_rows += f"<tr><th class='model'>{b}</th>"
        for m in models:
            r = pm[m]["pf_bucket"][b]
            rate = r["open_rate"]
            open_pct = rate * 100
            # red (accent) heatmap: a single hue scaled by open-rate on the paper
            # background. Flip to white text once the red fill is dark enough.
            bg = f"rgba(143,29,29,{rate:.2f})"
            txt = "#fbfbf8" if rate > 0.5 else "#1c1c1c"
            sub = "#f1d9d9" if rate > 0.5 else "#6b6b6b"
            bucket_rows += (
                f"<td style='background:{bg};color:{txt}'>{open_pct:.0f}%"
                f"<div class='small' style='color:{sub}'>n={r['hands']}</div></td>"
            )
        bucket_rows += "</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Tournament</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🃏</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>{BASE_CSS}
  td.bucket {{ font-weight: 600; }}
  /* Prominent dividers for the three top-level sections (Results / Why / More). */
  h2.section {{ font-size:23px; margin:56px 0 18px; padding-top:16px;
    border-top:3px solid var(--red); color:var(--red); letter-spacing:.01em; }}
  h2.section:first-of-type {{ margin-top:32px; }}
  /* Style cell: label on its own line, behaviour tag(s) on a second line, always
     two lines so the column never wraps mid-phrase. */
  td.stylecell {{ white-space:nowrap; line-height:1.5; }}
  td.stylecell .stags {{ margin-top:2px; }}
  .strategy-intro {{ background:var(--faint); border:1px solid var(--line);
    border-left:3px solid var(--red); padding:12px 14px; margin:12px 0 18px;
    font-size:13px; line-height:1.55; }}
  .strategy-glossary {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
    gap:8px 14px; margin-top:12px; }}
  .winloss-keys {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
    gap:10px; margin-top:12px; }}
  .winloss-keys > div {{ border:1px solid var(--line); border-radius:4px;
    padding:8px 10px; background:var(--panel); }}
  .winloss-keys b {{ display:block; font-size:12px; }}
  .winloss-keys span {{ display:block; color:var(--dim); font-size:11px; margin-top:2px; }}
  .wk-win {{ border-left:3px solid var(--pos) !important; }}
  .wk-lose {{ border-left:3px solid var(--neg) !important; }}
  /* merged by-street table: divider before each street's (fire,fold) pair */
  table.bystreet td:nth-child(even), table.bystreet tr:nth-child(2) th:nth-child(odd) {{
    border-left:2px solid var(--line); }}
  table.bystreet th[colspan] {{ text-align:center; }}
  .strategy-gloss {{ border-top:1px solid var(--line); padding-top:7px; }}
  .strategy-gloss b {{ display:block; color:var(--fg); }}
  .strategy-gloss span {{ display:block; color:var(--fg); }}
  .strategy-gloss em {{ display:block; color:var(--dim); font-style:normal; font-size:11px; }}
  .strategy-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
    gap:18px; margin-top:14px; }}
  .strategy-card {{ border:1px solid var(--line); background:var(--panel);
    padding:14px; }}
  .strategy-head {{ display:flex; justify-content:space-between; gap:12px;
    align-items:flex-start; }}
  .strategy-head h3 {{ margin-bottom:4px; }}
  .strategy-kpi {{ text-align:right; font-weight:700; font-size:18px; white-space:nowrap; }}
  .strategy-kpi span {{ display:block; color:var(--dim); font-size:10px; font-weight:400; }}
  .metric-profile-layout {{ display:block; margin-top:4px; }}
  .metric-profile-layout canvas {{ width:100% !important; max-height:230px; margin:0 0 10px; }}
  .profile-summary {{ margin:0; padding:10px 12px 10px 24px; background:var(--faint);
    border:1px solid var(--line); color:var(--fg); font-size:12px; line-height:1.42; }}
  .profile-summary li {{ margin:5px 0; }}
  /* one-line verdict + where-the-money-comes-from bar (pressure vs showdown) */
  .pl-verdict {{ margin:2px 0 8px; font-size:12.5px; line-height:1.45; color:var(--fg); }}
  .pl-verdict .pl-tag {{ font-weight:700; }}
  .pl-attr {{ margin:0 0 10px; font-size:11px; }}
  .pl-row {{ display:flex; align-items:center; gap:7px; margin:4px 0; }}
  .pl-lbl {{ width:74px; color:var(--dim); white-space:nowrap; }}
  .pl-track {{ position:relative; flex:1; height:13px; background:var(--faint);
    border:1px solid var(--line); border-radius:2px; }}
  .pl-track::before {{ content:''; position:absolute; left:50%; top:-1px; bottom:-1px;
    width:1px; background:var(--dim); opacity:.5; }}
  .plf {{ position:absolute; top:1px; bottom:1px; }}
  .plf.pos {{ background:var(--pos); }}
  .plf.neg {{ background:var(--neg); }}
  .pl-val {{ width:52px; text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }}
  .pl-val.pos {{ color:var(--pos); }} .pl-val.neg {{ color:var(--neg); }}
  .strategy-cases {{ margin:6px 0 0; padding-left:22px; font-size:12px; line-height:1.45; }}
  .strategy-cases li {{ margin:8px 0; }}
  .case-title {{ font-weight:700; }}
  .case-signal {{ color:var(--fg); margin-top:2px; }}
  .case-meta {{ color:var(--dim); font-size:11px; margin-top:2px; }}
  .case-link {{ display:inline-block; margin-top:5px; border:1px solid var(--line);
    background:var(--faint); color:#4338ca; text-decoration:none; padding:3px 8px; font-size:11px; }}
  .case-link:hover {{ border-color:var(--red); color:var(--fg); }}
  @media (max-width:860px) {{ .strategy-grid, .strategy-glossary {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/holdem/1hand<span class="cursor"></span></h1>
  <div class="sub">🃏 Hold'em 1-Hand · heads-up · each hand scored independently (bb/100) · {report['num_games']} games · {report['hands_per_game']} hands each · {len(models)} models · round-robin</div>
  {replay_btn}

  <div class="rules">
    <h3>Setup — Hold'em 1-Hand</h3>
    Standard heads-up No-Limit
    <a href="https://en.wikipedia.org/wiki/Texas_hold_%27em" target="_blank" rel="noopener">Texas Hold'em</a>
    (full rules on Wikipedia); what defines this arena is how the hands are dealt and
    scored:
    <ul>
      <li><b>Heads-up round-robin:</b> every pair of models plays, with seats/button
        swapped so neither sits in a fixed position.</li>
      <li><b>{report['hands_per_game']} hands per matchup, each scored on its own</b> —
        stacks <b>reset to 200 chips</b> every hand (blinds <b>1 / 2</b>), so a single
        cooler can't snowball and nobody busts out.</li>
      <li>Win rate is reported as <b>bb/100</b> (big blinds won per 100 hands), the
        standard yardstick that stays comparable across different hand counts.</li>
    </ul>
    <div class="seq">Imperfect information with chance, so beyond results we profile each
    model's <b>playing style</b> (VPIP / aggression) and rate skill with a
    <b>chip-weighted, opponent-adjusted Elo</b> — rewarding how much you win, not just
    how often.</div>
    <div class="seq"><b>What the model sees each turn:</b> its own two hole cards, the community
    board, the pot and both stacks, its position, the bet it faces, the legal actions, and the
    action history this hand — never the opponent's cards.</div>
  </div>

  <h2 class="section">1 · Results — who won</h2>
  <h3>Leaderboard</h3>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>style</th><th>Elo</th><th>chips/hand</th><th>bb/100</th>
        <th>win%</th><th>hands</th><th>tokens/dec</th><th>$/1K dec</th></tr>
    {rows}
  </table>
  {_legend('holdem')}
  <div class="note"><b>Elo</b> = chip-weighted Bradley-Terry rating (field mean 1500): a standard Elo
    fit, but fed the chips won in each matchup rather than hand counts, so it rewards <i>how much</i> you
    win and adjusts for opponent strength — the fair comparison when models faced different opponents.
    ± is one bootstrap SD (resampling hands 300×); ratings within ±1 of each other are a statistical tie.
    chips / bb/100 / win% are raw, unadjusted metrics.
    tokens/dec = avg completion (reasoning) tokens generated per decision.
    <b>$/1K dec</b> = estimated cost per 1,000 decisions = tokens/dec × the model's Fireworks
    serverless decode price (output $/1M tokens). Both are <b>—</b> for Claude and GPT-5.x, which
    hide their chain-of-thought, so their token count (and cost) is not observable here.</div>
  <div class="note"><b>style</b> = play-style archetype, from VPIP (how loose) × aggression (how aggressive):
    <b>LAG</b> = loose-aggressive — plays many hands and bets/raises a lot (high-variance, wins if skilled);
    <b>TAG</b> = tight-aggressive — selective but aggressive (the classic solid winner);
    <b>calling station</b> = loose-passive — plays many hands but mostly calls, rarely folds or raises
    (pays off with second-best hands — the classic losing type);
    <b>nit</b> = tight-passive — folds almost everything, only plays the strongest hands;
    <b>shove-happy</b> = an unusually high all-in rate.</div>

  <h2>⚔️ Head-to-head chip results</h2>
  <div class="sub">Net chips <b>per hand</b> the row model won against the column model
    (normalized by hands played, since pairs played different counts; sums to zero per pair).</div>
  <table class="h2h">{hh}</table>

  <h2 class="section">2 · Why — what makes a model win or lose</h2>
  <div class="strategy-intro">
    <b>Two ways to win, one way to lose.</b> Chips come from two places:
    <span class="pos">Pressure</span> — winning pots <i>without</i> showdown by folding
    opponents out — and <span class="pos">Showdown</span> — winning <i>at</i> showdown with the
    stronger hand. Winners do at least one of these well; losers mostly bleed at showdown. The map
    below places every model on those two axes; the cards under it break down each one.
    <div class="winloss-keys">
      <div class="wk-win"><b>↑ tends to win</b><span>aggression &amp; pressure that forces folds; winning the pots you contest</span></div>
      <div class="wk-lose"><b>↓ tends to lose</b><span>over-sized bets; drifting to showdown with second-best hands</span></div>
    </div>
  </div>
  <h3>The map — where each model's chips come from</h3>
  <canvas id="quadrant" style="max-height:520px"></canvas>
  <div class="note">x = chips won by <b>pressure</b> (no showdown) · y = chips won at <b>showdown</b>.
    The <b>green/red number</b> next to each model is its <b>avg chips/hand</b> (pressure + showdown =
    total). Above the dashed line = net winner, below = net loser. Far right = wins by forcing folds
    (e.g. GPT 5.5); high up = wins with strong hands (e.g. Kimi); bottom = showdown bleed
    (e.g. Claude Opus).</div>

  {strategy_html}

  <h2 class="section">3 · Analysis</h2>
  {fold_pos_html}
  <h2>♟️ Action tendencies</h2>
  <canvas id="actions"></canvas>
  <div class="note">Share of each action across all the model's decisions.</div>

  <h2>🃏 Preflop open-rate by hand strength</h2>
  <div class="sub">% of hands the model voluntarily put chips in with each Chen-formula
    bucket. A model that reads its cards opens premium ≫ trash.</div>
  <table>
    <tr><th class='model'>bucket</th>{''.join(f"<th>{display_name(m)}</th>" for m in models)}</tr>
    {bucket_rows}
  </table>

  <h2>🎯 Postflop &amp; showdown stats</h2>
  <table>
    <tr><th class='model'>model</th><th>c-bet%</th><th>n</th><th>fold→c-bet%</th><th>n</th>
        <th>WTSD%</th><th>W$SD%</th><th>SD seen / won</th></tr>
    {pf_rows}
  </table>
  <div class="note">
    <b>c-bet</b> = flop bet by the preflop aggressor. <b>fold→c-bet</b> = how often the
    caller folds when facing a flop c-bet. <b>WTSD</b> = went to showdown%.
    <b>W$SD</b> = won at showdown (of showdowns seen).
  </div>

  <h2>🏅 Made-hand mix at showdown</h2>
  <canvas id="madeHand"></canvas>
  <div class="note">Share of the model's showdown hands that finished as each category
    (high card → full house). Tight selectors reach showdown with stronger made hands.</div>

  <h2>💸 Bet-sizing distribution</h2>
  <canvas id="betsize"></canvas>
  <div class="note">Of all aggressive actions, what fraction were small (&lt;½ pot),
    medium (½–1 pot), pot-sized (1–1.5 pot), or over-pot (≥1.5 pot).</div>

<script>
const R = {payload};
const MODELS = R.models, PM = R.per_model;
// Official display names; PM stays keyed by the slug, so dn() is display-only.
const DN = {json.dumps({m: display_name(m) for m in models})};
const dn = m => DN[m] || m;
const COLORS = ['#60a5fa','#f472b6','#4ade80','#fbbf24','#a78bfa','#22d3ee'];
const col = i => COLORS[i % COLORS.length];
// stable per-model color, used everywhere a chart is keyed by model
const MODEL_COL = Object.fromEntries(MODELS.map((m,i)=>[m, col(i)]));
const mcol = m => MODEL_COL[m];
const cssText = getComputedStyle(document.body).color;
{CHART_SETUP}
{strategy_js}

// action distribution (stacked %) — GPT models pulled to the front
const ACTS=['fold','check','call','bet','raise','all_in'];
const ACOL={{fold:'#6b7280',check:'#94a3b8',call:'#38bdf8',bet:'#fbbf24',raise:'#fb923c',all_in:'#f87171'}};
const ACT_ORDER=[...MODELS.filter(m=>m.startsWith('gpt-5')).sort().reverse(),
                 ...MODELS.filter(m=>!m.startsWith('gpt-5'))];
new Chart(actions, {{ type:'bar',
  data:{{ labels:ACT_ORDER.map(dn), datasets:ACTS.map(a=>({{ label:a, backgroundColor:ACOL[a],
      data:ACT_ORDER.map(m=>{{const mix=PM[m].action_mix;const tot=Object.values(mix).reduce((x,y)=>x+y,0)||1;
        return 100*mix[a]/tot;}}) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100,title:{{display:true,text:'% of actions'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// bet-sizing distribution (stacked %)
const BS=['small','medium','pot','over'];
const BSC={{small:'#94a3b8',medium:'#38bdf8',pot:'#fbbf24',over:'#f87171'}};
new Chart(betsize, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:BS.map(b=>({{ label:b, backgroundColor:BSC[b],
      data:MODELS.map(m=>PM[m].betsize_dist[b]*100) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100,
      title:{{display:true,text:'% of bets'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});


// made-hand mix at showdown (stacked %)
const CATS=['high card','pair','two pair','trips','straight','flush','full house'];
const CC=['#6b7280','#94a3b8','#38bdf8','#4ade80','#fbbf24','#a78bfa','#f472b6'];
new Chart(madeHand, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:CATS.map((c,j)=>({{ label:c, backgroundColor:CC[j],
      data:MODELS.map(m=>(PM[m].showdown_cats[c]||0)*100) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100,
      title:{{display:true,text:'% of showdowns'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// "where the chips come from" map: pressure (x) vs showdown (y), per hand
const QUAD = {json.dumps(quad_pts)};
(function(){{
  const el=document.getElementById('quadrant'); if(!el||!window.Chart) return;
  const pts=QUAD.map(p=>({{x:p.x,y:p.y,label:p.label,model:p.m,t:p.t}}));
  const fmt=v=>(v>=0?'+':'')+v.toFixed(2);
  const guides={{ id:'qguides', beforeDatasetsDraw(c){{
    const a=c.chartArea, x=c.scales.x, y=c.scales.y, ctx=c.ctx;
    ctx.save(); ctx.beginPath(); ctx.rect(a.left,a.top,a.right-a.left,a.bottom-a.top); ctx.clip();
    // solid zero axes (--line); shade the winning half (total > 0) very faintly green
    const x0=x.getPixelForValue(0), y0=y.getPixelForValue(0);
    ctx.setLineDash([4,4]); ctx.strokeStyle='#c7b9b9';
    ctx.beginPath();
    ctx.moveTo(x.getPixelForValue(x.min), y.getPixelForValue(-x.min));
    ctx.lineTo(x.getPixelForValue(x.max), y.getPixelForValue(-x.max));
    ctx.stroke();
    ctx.setLineDash([]); ctx.strokeStyle='#ddd8cf'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x0,a.top); ctx.lineTo(x0,a.bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(a.left,y0); ctx.lineTo(a.right,y0); ctx.stroke();
    ctx.restore();
  }} }};
  const labels={{ id:'qlabels', afterDatasetsDraw(c){{
    const ctx=c.ctx, meta=c.getDatasetMeta(0), a=c.chartArea, base=getComputedStyle(document.body).color;
    ctx.save(); ctx.textBaseline='middle';
    const placed=[], H=12;
    const hit=(b1,b2)=>!(b1.x+b1.w<b2.x||b2.x+b2.w<b1.x||b1.y+b1.h<b2.y||b2.y+b2.h<b1.y);
    // place tighter clusters first so dense spots get the simple right-side slot
    const order=[...meta.data.keys()];
    order.forEach((idx)=>{{
      const pt=meta.data[idx], p=pts[idx], name=p.label+' ';
      ctx.font='10px ui-monospace,monospace'; const nw=ctx.measureText(name).width;
      ctx.font='bold 10px ui-monospace,monospace'; const w=nw+ctx.measureText(fmt(p.t)).width;
      const cands=[[10,0],[10,-13],[10,13],[-w-9,0],[10,-26],[10,26],[-w-9,-13],[-w-9,13]];
      let bx=pt.x+10, by=pt.y;
      for(const d of cands){{
        const cx=pt.x+d[0], cy=pt.y+d[1], box={{x:cx,y:cy-H/2,w:w,h:H}};
        if(cx<a.left||cx+w>a.right||cy-H/2<a.top||cy+H/2>a.bottom) continue;
        if(placed.some(pb=>hit(box,pb))) continue;
        bx=cx; by=cy; break;
      }}
      placed.push({{x:bx,y:by-H/2,w:w,h:H}});
      ctx.font='10px ui-monospace,monospace'; ctx.fillStyle=base; ctx.fillText(name, bx, by);
      ctx.font='bold 10px ui-monospace,monospace';
      ctx.fillStyle=p.t>=0?'#1a7f37':'#b91c1c'; ctx.fillText(fmt(p.t), bx+nw, by);
    }});
    ctx.restore();
  }} }};
  new Chart(el,{{ type:'scatter',
    data:{{ datasets:[{{ data:pts, pointRadius:5, pointHoverRadius:8,
      backgroundColor:pts.map(p=>mcol(p.model)), borderColor:'#0003', borderWidth:1 }}] }},
    options:{{ layout:{{padding:{{right:96}}}}, plugins:{{ legend:{{display:false}},
        tooltip:{{ callbacks:{{ label:(ctx)=>`${{ctx.raw.label}} — total ${{fmt(ctx.raw.t)}}/h (pressure ${{ctx.raw.x.toFixed(2)}}, showdown ${{ctx.raw.y.toFixed(2)}})` }} }} }},
      scales:{{ x:{{ title:{{display:true,text:'Pressure $ / hand  (win without showdown →)'}} }},
                y:{{ title:{{display:true,text:'Showdown $ / hand  (win at showdown ↑)'}} }} }} }},
    plugins:[guides,labels] }});
}})();
</script>
</div></body></html>"""


EXCLUDE_HOLDEM = {"gpt-oss-120b"}   # dropped for an incomplete schedule (no GPT 5.5 / 5.4 games)


def main():
    data = strip_coached(json.load(open(DATA)))
    # Remove GPT-OSS and every game it played, then re-aggregate so the remaining
    # models' chips / win rate / Elo / h2h are computed gpt-oss-free.
    data["models"] = [m for m in data["models"] if m not in EXCLUDE_HOLDEM]
    data["games"] = [g for g in data["games"]
                     if g["a"] not in EXCLUDE_HOLDEM and g["b"] not in EXCLUDE_HOLDEM]
    report = analyze(data)
    html = render_html(report)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    # also dump computed stats as json for inspection
    json.dump(report, open(os.path.join(os.path.dirname(OUT), "analysis.json"), "w"), indent=2)
    # tracked copy for the repo (runs/ is gitignored)
    os.makedirs(REPORT_DIR, exist_ok=True)
    repo_html = os.path.join(REPORT_DIR, "holdem_tournament_report.html")
    with open(repo_html, "w", encoding="utf-8") as f:
        f.write(html)
    json.dump(report, open(os.path.join(REPORT_DIR, "holdem_tournament_analysis.json"), "w"),
              indent=2)
    print(f"Wrote {OUT}")
    print(f"Wrote {repo_html} (tracked)")
    print(f"Models analyzed: {report['num_games']} games, "
          f"{report['hands_per_game']} hands each")
    for m in sorted(report["per_model"],
                    key=lambda x: report["per_model"][x]["bb_per_100"], reverse=True):
        s = report["per_model"][m]
        print(f"  {m:<16} {s['style']['label']:<18} bb/100={s['bb_per_100']:+6.1f} "
              f"VPIP={s['vpip']*100:.0f}% aggr={s['agg_freq']*100:.0f}% "
              f"allin={s['allin_freq']*100:.0f}%")


if __name__ == "__main__":
    main()
