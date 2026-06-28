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

from model_names import strip_coached, display_name, model_cell
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
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=29"></script>'
BETSIZE_BUCKETS = ("small", "medium", "pot", "over")
SHOWDOWN_CATS = ("high card", "pair", "two pair", "trips", "straight", "flush",
                 "full house")


def _blank():
    return {
        "hands": 0, "decisions": 0, "chips": 0.0,
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
    h2h_out = {a: {b: round(h2h[a][b] / max(h2h_hands[a][b], 1), 3)
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
    """Signal-count model cards for the Additional analysis section.

    Scores are diagnostic leak evidence: each triggered signal adds one point.
    Higher values mean "more to review", not "better strategic skill".
    """
    pm = report["per_model"]
    dims = [
        ("vpip", "VPIP", "Voluntarily put chips in preflop; higher means looser preflop entry."),
        ("pfr", "PFR", "Preflop raise frequency; higher means more preflop initiative."),
        ("agg_freq", "AFq", "Aggressive action frequency after opportunities to bet/raise."),
        ("cbet_rate", "C-bet", "Flop continuation bet rate after being the preflop aggressor."),
        ("wsd", "W$SD", "Won money at showdown; higher means better showdown conversion."),
        ("allin_freq", "All-in", "All-in action frequency; higher means more stack-risk exposure."),
    ]

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
        values = [float(pm[m].get(key) or 0) for m in ranked]
        pct_by_metric[key] = percentile(values)

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

    for rank, m in enumerate(ranked, 1):
        s = pm[m]
        raw = {key: float(s.get(key) or 0) for key, _label, _help in dims}
        scores = [pct_by_metric[key].get(raw[key], 0) for key, _label, _help in dims]
        raw_labels = {
            label: pct(raw[key], 1 if key == "allin_freq" else 0)
            for key, label, _help in dims
        }
        chart_payload.append({
            "id": f"strategyRadar{rank}",
            "model": m,
            "label": display_name(m),
            "scores": scores,
            "raw": raw_labels,
        })
        summary_items = "".join(f"<li>{html_lib.escape(item)}</li>" for item in summarize_profile(s))
        cards.append(f"""
    <article class="strategy-card metric-card">
      <div class="strategy-head">
        <div><h3>{rank}. {model_cell(m)}</h3></div>
        <div class="strategy-kpi {'pos' if s['bb_per_100'] >= 0 else 'neg'}">{s['bb_per_100']:+.1f}<span>bb/100</span></div>
      </div>
      <div class="metric-profile-layout">
        <canvas id="strategyRadar{rank}"></canvas>
        <ul class="profile-summary">{summary_items}</ul>
      </div>
    </article>""")

    html = f"""
  <div class="strategy-intro">
    <b>Player metric radar:</b> each axis is exactly one observed metric, not a
    combined leak score. Radar radius is field percentile so small-scale metrics
    such as all-in frequency remain visible; the bullets under each chart
    summarize the player's style from those same metrics.
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
      datasets:[{{ label:card.label, data:card.scores,
        borderColor:mcol(card.model), backgroundColor:mcol(card.model) + '33',
        pointBackgroundColor:mcol(card.model) }}] }},
    options:{{ scales:{{ r:{{ min:0, max:100, ticks:{{ display:false, stepSize:25 }} }} }},
      plugins:{{ legend:{{ display:false }},
        tooltip:{{ callbacks:{{ label(ctx) {{
          const raw = card.raw[ctx.label] || '';
          return `percentile ${{ctx.formattedValue}} / raw ${{raw}}`;
        }} }} }} }} }} }});
}});
"""
    return html, js


def _handeq_pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def _handeq_pp(x):
    return "—" if x is None else f"{x * 100:+.1f}pp"


def _handeq_heat_color(v, center=0.0, span=0.45):
    if v is None:
        return "background:var(--faint);color:var(--dim)"
    delta = v - center
    alpha = min(0.72, abs(delta) / span * 0.72)
    rgb = "26,127,55" if delta >= 0 else "185,28,28"
    return f"background:rgba({rgb},{alpha:.3f})"


def _handeq_centered_color(v):
    if v is None:
        return "background:var(--faint);color:var(--dim)"
    delta = v - 0.5
    if abs(delta) < 0.005:
        return "background:#fff"
    alpha = min(0.72, abs(delta) / 0.5 * 0.72)
    rgb = "26,127,55" if delta > 0 else "185,28,28"
    return f"background:rgba({rgb},{alpha:.3f})"


def _handeq_positive_actions(action_quality_matrix):
    positives = {}
    for row in action_quality_matrix:
        sid = row["id"]
        positives[sid] = {
            action["action"]
            for action in row.get("actions", [])
            if (action.get("quality_lift") or 0) > 0
        }
    return positives


def _render_handeq_action_quality(matrix):
    rows = []
    for row in matrix:
        cells = []
        for cell in row.get("actions", []):
            lift = cell.get("equity_lift")
            if lift is None:
                lift = cell.get("quality_lift")
            cells.append(
                f"<div class='handeq-action-cell' style='{_handeq_heat_color(lift)}'>"
                f"<div class='handeq-action-name'>{html_lib.escape(cell['label'])}</div>"
                f"<div class='handeq-action-lift'>{_handeq_pp(lift)}</div>"
                f"<div class='handeq-action-meta'>EQ {_handeq_pct(cell.get('action_avg_equity'))} · "
                f"base {_handeq_pct(cell.get('baseline_avg_equity'))}</div>"
                f"<div class='handeq-sample-track'><i style='width:{(cell.get('sample_rate') or 0) * 100:.1f}%'></i></div>"
                f"<div class='handeq-action-meta'>sample {_handeq_pct(cell.get('sample_rate'))} · "
                f"n={cell.get('count')}</div>"
                f"</div>"
            )
        rows.append(f"""
    <section class="handeq-action-row">
      <div class="handeq-situation-label">
        <b>{html_lib.escape(row['label'])}</b>
        <span>avg EQ {_handeq_pct(row.get('avg_equity'))} · n={row.get('opportunities')}</span>
      </div>
      <div class="handeq-action-cells">{''.join(cells)}</div>
    </section>""")
    return f"""
  <h2>Action quality by situation</h2>
  <div class="note">Each row is a strategic situation found during the hand. Each cell is an action observed in that situation. Quality change = that action's average hand equity minus the average hand equity of the other observed actions in the same situation.</div>
  <div class="handeq-action-chart">{''.join(rows)}</div>
"""


def _render_handeq_model_matrix(data, model_order):
    matrix = data.get("model_action_quality_matrix", {})
    columns = matrix.get("columns", [])
    rows = matrix.get("rows", [])
    positives = _handeq_positive_actions(data.get("action_quality_matrix", []))

    situations = []
    for col in columns:
        if col["situation_id"] not in [s["id"] for s in situations]:
            situations.append({"id": col["situation_id"], "label": col["situation"]})
    cols_by_situation = {
        situation["id"]: [col for col in columns if col["situation_id"] == situation["id"]]
        for situation in situations
    }
    row_by_model = {row["model"]: row for row in rows}
    model_rows = []
    for model in model_order:
        row = row_by_model.get(model)
        if not row:
            continue
        cell_by_col = {cell["column_id"]: cell for cell in row["cells"]}
        cells = []
        model_total_opps = 0
        for situation in situations:
            sid = situation["id"]
            opps = 0
            aligned = 0
            top_actions = []
            for col in cols_by_situation[sid]:
                cell = cell_by_col.get(col["id"], {})
                if cell.get("opportunities"):
                    opps = max(opps, cell["opportunities"])
                count = cell.get("count") or 0
                if col["action"] in positives.get(sid, set()):
                    aligned += count
                if count:
                    top_actions.append((cell.get("sample_rate") or 0, col["label"]))
            model_total_opps += opps
            top_actions.sort(reverse=True)
            cells.append({
                "label": situation["label"],
                "opportunities": opps,
                "aligned_rate": aligned / opps if opps else None,
                "top_actions": top_actions[:2],
            })
        for cell in cells:
            cell["situation_share"] = cell["opportunities"] / model_total_opps if model_total_opps else None
        model_rows.append({"model": model, "cells": cells})

    head = "".join(f"<th><span>{html_lib.escape(s['label'])}</span></th>" for s in situations)
    body = []
    for item in model_rows:
        tds = []
        for cell in item["cells"]:
            aligned = cell["aligned_rate"]
            share = cell["situation_share"]
            top_actions = "".join(
                f"<span>{html_lib.escape(label)} {_handeq_pct(rate)}</span>"
                for rate, label in cell["top_actions"]
            )
            tds.append(f"""
        <td class="handeq-model-cell" style="{_handeq_centered_color(aligned)}">
          <div class="handeq-cell-main">{_handeq_pct(aligned)}</div>
          <div class="handeq-cell-label">quality-aligned</div>
          <div class="handeq-cell-meta">situation share {_handeq_pct(share)} · n={cell['opportunities']}</div>
          <div class="handeq-cell-actions">{top_actions}</div>
        </td>""")
        body.append(f"<tr><td class='model'>{model_cell(item['model'])}</td>{''.join(tds)}</tr>")
    return f"""
  <h2>Model situation exposure × quality-aligned choice rate</h2>
  <div class="note">Y-axis is model; X-axis is situation. The large number is the share of choices assigned to actions whose hand-equity lift is positive within that situation. Cell color uses 50% as the white midpoint: greener above, redder below.</div>
  <div class="wide-scroll"><table class="handeq-model-table">
    <tr><th class="model">model</th>{head}</tr>
    {''.join(body)}
  </table></div>
"""


def _hand_equity_analysis(ranked: list) -> str:
    path = os.path.join(REPORT_DIR, "holdem_1hand_situation_impact.json")
    if not os.path.exists(path):
        return (
            '  <div class="callout">Hand-equity diagnostics are not generated yet. Run\n'
            '    <code>python3 scripts/analyze_1hand_situations.py</code> and regenerate this report.</div>'
        )
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"<div class='callout'>Could not load hand-equity diagnostics: {html_lib.escape(str(exc))}</div>"
    action_quality = _render_handeq_action_quality(data.get("action_quality_matrix", [])).strip()
    model_matrix = _render_handeq_model_matrix(data, ranked).strip()
    return f"""\
  <div class="callout"><b>Hand-equity preview.</b> This section asks whether each model's action choices line up with the hand-equity signal available at the moment of decision. A positive lift means the chosen action's average equity was higher than the alternative observed actions in the same situation; it is evidence, not a claim of perfect poker optimality.</div>
{action_quality}
{model_matrix}
"""


# ---------------------------------------------------------------------------
def render_html(report: dict) -> str:
    models = report["models"]
    pm = report["per_model"]
    payload = json.dumps(report)
    replay_btn = ('<a class="replaybtn" href="holdem_replay.html?v=17">'
                  '▶ watch hand replays</a>')

    # ranked leaderboard by chip-weighted Elo; raw metrics kept for reference.
    elo = report.get("elo", {})
    ranked = sorted(models, key=lambda m: (elo_key(elo, m), pm[m]["bb_per_100"]),
                    reverse=True)
    strategy_html, strategy_js = _strategy_analysis(report, ranked)
    handeq_html = _hand_equity_analysis(ranked)
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
          <td>{s['avg_latency_s']:.1f}s</td>
          <td>{s['avg_comp_tokens']:,}</td>
          <td>{s['hands']}</td>
        </tr>"""

    # head-to-head matrix
    h2h = report["h2h"]
    hh = "<tr><th></th>" + "".join(f"<th>{display_name(m)}</th>" for m in models) + "</tr>"
    for a in models:
        hh += f"<tr><th class='model'>{model_cell(a)}</th>"
        for b in models:
            if a == b:
                hh += "<td class='diag'>—</td>"
            else:
                v = h2h[a].get(b, 0)
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
  .strategy-cases {{ margin:6px 0 0; padding-left:22px; font-size:12px; line-height:1.45; }}
  .strategy-cases li {{ margin:8px 0; }}
  .case-title {{ font-weight:700; }}
  .case-signal {{ color:var(--fg); margin-top:2px; }}
  .case-meta {{ color:var(--dim); font-size:11px; margin-top:2px; }}
  .case-link {{ display:inline-block; margin-top:5px; border:1px solid var(--line);
    background:var(--faint); color:#4338ca; text-decoration:none; padding:3px 8px; font-size:11px; }}
  .case-link:hover {{ border-color:var(--red); color:var(--fg); }}
  .handeq-action-chart {{ border:1px solid var(--line); background:var(--panel); }}
  .handeq-action-row {{ display:grid; grid-template-columns:220px 1fr; border-top:1px solid var(--line); }}
  .handeq-action-row:first-child {{ border-top:0; }}
  .handeq-situation-label {{ padding:14px; border-right:1px solid var(--line); background:rgba(255,255,255,.35); }}
  .handeq-situation-label b {{ display:block; font-size:13px; }}
  .handeq-situation-label span {{ display:block; color:var(--dim); font-size:11px; margin-top:4px; }}
  .handeq-action-cells {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(126px,1fr)); gap:8px; padding:10px; }}
  .handeq-action-cell {{ border:1px solid var(--line); padding:10px; min-height:94px; display:flex; flex-direction:column; justify-content:space-between; }}
  .handeq-action-name {{ font-size:11px; color:var(--dim); text-transform:uppercase; font-weight:800; }}
  .handeq-action-lift {{ font-size:22px; font-weight:900; margin:2px 0; line-height:1; }}
  .handeq-action-meta {{ font-size:10px; color:var(--fg); line-height:1.25; }}
  .handeq-sample-track {{ height:5px; background:rgba(255,255,255,.5); border:1px solid rgba(0,0,0,.12); margin:5px 0 3px; }}
  .handeq-sample-track i {{ display:block; height:100%; background:rgba(28,28,28,.55); }}
  .wide-scroll {{ overflow-x:auto; border:1px solid var(--line); background:var(--panel); margin-top:14px; }}
  .handeq-model-table {{ min-width:1180px; border-collapse:separate; border-spacing:0; background:var(--panel); }}
  .handeq-model-table th {{ text-align:center; vertical-align:bottom; }}
  .handeq-model-table th span {{ display:block; max-width:140px; margin:0 auto; white-space:normal; line-height:1.25; }}
  .handeq-model-cell {{ min-width:148px; padding:10px; text-align:left; vertical-align:top; }}
  .handeq-cell-main {{ font-size:24px; line-height:1; font-weight:900; color:var(--fg); }}
  .handeq-cell-label {{ font-size:9px; color:var(--dim); margin-top:3px; text-transform:uppercase; font-weight:800; }}
  .handeq-cell-meta {{ font-size:10px; color:var(--fg); line-height:1.25; margin-top:8px; }}
  .handeq-cell-actions {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:7px; min-height:18px; }}
  .handeq-cell-actions span {{ font-size:9px; line-height:1; padding:4px 5px; border:1px solid var(--line); background:rgba(255,255,255,.55); white-space:nowrap; }}
  @media (max-width:860px) {{ .strategy-grid, .strategy-glossary {{ grid-template-columns:1fr; }} }}
  @media (max-width:800px) {{ .handeq-action-row {{ grid-template-columns:1fr; }} .handeq-situation-label {{ border-right:0; border-bottom:1px solid var(--line); }} }}
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
  </div>

  <h2 class="section">1 · 🏆 Results — who won</h2>
  <h3>Leaderboard</h3>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>style</th><th>Elo</th><th>chips/hand</th><th>bb/100</th>
        <th>win%</th><th>think</th><th>tokens/dec</th><th>hands</th></tr>
    {rows}
  </table>
  {_legend('holdem')}
  <div class="note"><b>Elo</b> = chip-weighted Bradley-Terry rating (field mean 1500): a standard Elo
    fit, but fed the chips won in each matchup rather than hand counts, so it rewards <i>how much</i> you
    win and adjusts for opponent strength — the fair comparison when models faced different opponents.
    ± is one bootstrap SD (resampling hands 300×); ratings within ±1 of each other are a statistical tie.
    chips / bb/100 / win% are raw, unadjusted metrics. think = avg seconds per decision.
    tokens/dec = avg completion (reasoning) tokens generated per decision.</div>

  <h2>♟️ Action tendencies</h2>
  <canvas id="actions"></canvas>
  <div class="note">Share of each action across all the model's decisions.</div>

  <h2>⚔️ Head-to-head chip results</h2>
  <div class="sub">Net chips <b>per hand</b> the row model won against the column model
    (normalized by hands played, since pairs played different counts; sums to zero per pair).</div>
  <table class="h2h">{hh}</table>

  <h2 class="section">2 · 🔍 Why — what decides win &amp; loss</h2>
  {handeq_html}

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

  <h2 class="section">3 · 🔬 Additional analysis</h2>
  {strategy_html}

  <h2>⏱️ Avg thinking time</h2>
  <canvas id="latency"></canvas>
  <div class="note">Average seconds per decision. This is an efficiency/agent-runtime metric, not direct poker quality.</div>

  <div class="grid2">
    <div><h2>💸 Bet-sizing distribution</h2><canvas id="betsize"></canvas>
      <div class="note">Of all aggressive actions, what fraction were small (&lt;½ pot),
        medium (½–1 pot), pot-sized (1–1.5 pot), or over-pot (≥1.5 pot).</div></div>
    <div><h2>🧠 Latency vs win rate</h2><canvas id="latVsWin"></canvas>
      <div class="note">Does thinking longer pay off? x = avg seconds per decision,
        y = bb/100.</div></div>
  </div>

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

// action distribution (stacked %)
const ACTS=['fold','check','call','bet','raise','all_in'];
const ACOL={{fold:'#6b7280',check:'#94a3b8',call:'#38bdf8',bet:'#fbbf24',raise:'#fb923c',all_in:'#f87171'}};
new Chart(actions, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:ACTS.map(a=>({{ label:a, backgroundColor:ACOL[a],
      data:MODELS.map(m=>{{const mix=PM[m].action_mix;const tot=Object.values(mix).reduce((x,y)=>x+y,0)||1;
        return 100*mix[a]/tot;}}) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,title:{{display:true,text:'% of actions'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// latency
new Chart(latency, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:[{{data:MODELS.map(m=>PM[m].avg_latency_s),
      backgroundColor:MODELS.map(m=>mcol(m))}}] }},
  options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{title:{{display:true,text:'seconds / decision'}}}}}} }} }});

// bet-sizing distribution (stacked %)
const BS=['small','medium','pot','over'];
const BSC={{small:'#94a3b8',medium:'#38bdf8',pot:'#fbbf24',over:'#f87171'}};
new Chart(betsize, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:BS.map(b=>({{ label:b, backgroundColor:BSC[b],
      data:MODELS.map(m=>PM[m].betsize_dist[b]*100) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100,
      title:{{display:true,text:'% of bets'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// latency vs bb/100 scatter
new Chart(latVsWin, {{ type:'scatter',
  data:{{ datasets:MODELS.map(m=>({{label:dn(m),
      data:[{{x:PM[m].avg_latency_s,y:PM[m].bb_per_100}}],
      backgroundColor:mcol(m), pointRadius:9, pointHoverRadius:12 }})) }},
  options:{{ scales:{{
      x:{{title:{{display:true,text:'avg seconds / decision'}},type:'logarithmic'}},
      y:{{title:{{display:true,text:'bb / 100 hands'}}}} }},
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
</script>
</div></body></html>"""


def main():
    data = strip_coached(json.load(open(DATA)))
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
