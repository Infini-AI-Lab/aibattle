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
import os
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
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=27"></script>'
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
    h2h_out = {a: {b: round(h2h[a][b], 1) for b in models if b != a} for a in models}
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
          <td class='{chip_cls}'>{s['chips']:+.0f}</td>
          <td class='{chip_cls}'>{s['bb_per_100']:+.1f}</td>
          <td>{s['win_rate']*100:.0f}%</td>
          <td>{s['vpip']*100:.0f}%</td>
          <td>{s['pfr']*100:.0f}%</td>
          <td>{s['agg_freq']*100:.0f}%</td>
          <td>{s['fold_to_bet']*100:.0f}%</td>
          <td>{s['allin_freq']*100:.0f}%</td>
          <td>{s['avg_bet_xpot']:.2f}x</td>
          <td>{s['avg_latency_s']:.1f}s</td>
          <td>{s['avg_comp_tokens']:,}</td>
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
                hh += f"<td class='{cls}' style='--v:{v}'>{v:+.0f}</td>"
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
  /* Style cell: label on its own line, behaviour tag(s) on a second line, always
     two lines so the column never wraps mid-phrase. */
  td.stylecell {{ white-space:nowrap; line-height:1.5; }}
  td.stylecell .stags {{ margin-top:2px; }}
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

  <h2>🏆 Leaderboard &amp; player profiles</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>style</th><th>Elo</th><th>chips</th><th>bb/100</th>
        <th>win%</th><th>VPIP</th><th>PFR</th><th>aggr</th><th>fold→bet</th>
        <th>all-in%</th><th>bet size</th><th>think</th><th>tokens/dec</th></tr>
    {rows}
  </table>
  {_legend('holdem')}
  <div class="note"><b>Elo</b> = chip-weighted Bradley-Terry rating (field mean 1500): a standard Elo
    fit, but fed the chips won in each matchup rather than hand counts, so it rewards <i>how much</i> you
    win and adjusts for opponent strength — the fair comparison when models faced different opponents.
    ± is one bootstrap SD (resampling hands 300×); ratings within ±1 of each other are a statistical tie.
    chips / bb/100 / win% are raw, unadjusted metrics. VPIP = how often it voluntarily plays a hand
    (looseness). PFR = preflop raise %.
    aggr = aggression frequency. fold→bet = how often it folds when bet at. bet size = avg bet as a multiple of the pot.
    think = avg seconds per decision. tokens/dec = avg completion (reasoning) tokens generated per decision.</div>

  <div class="grid2">
    <div><h2>🎭 Player-type map</h2><canvas id="scatter"></canvas>
      <div class="note">x = looseness (VPIP), y = aggression. Upper-right = loose-aggressive (LAG), lower-left = nit.</div></div>
    <div><h2>💰 Win rate (bb / 100 hands)</h2><canvas id="bbchart"></canvas></div>
  </div>

  <h2>♟️ Action tendencies</h2>
  <canvas id="actions"></canvas>
  <div class="note">Share of each action across all the model's decisions.</div>

  <div class="grid2">
    <div><h2>🧭 Style radar</h2><canvas id="radar"></canvas></div>
    <div><h2>⏱️ Avg thinking time</h2><canvas id="latency"></canvas></div>
  </div>

  <h2>📈 Aggression by street</h2>
  <canvas id="streetAgg"></canvas>
  <div class="note">Share of each model's actions on that street that were a bet/raise/all-in.
    A model that c-bets but then check-folds turn/river will dip from flop → river.</div>

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

  <h2>🃏 Preflop open-rate by hand strength</h2>
  <div class="sub">% of hands the model voluntarily put chips in with each Chen-formula
    bucket. A model that reads its cards opens premium ≫ trash.</div>
  <table>
    <tr><th class='model'>bucket</th>{''.join(f"<th>{display_name(m)}</th>" for m in models)}</tr>
    {bucket_rows}
  </table>

  <div class="grid2">
    <div><h2>💸 Bet-sizing distribution</h2><canvas id="betsize"></canvas>
      <div class="note">Of all aggressive actions, what fraction were small (&lt;½ pot),
        medium (½–1 pot), pot-sized (1–1.5 pot), or over-pot (≥1.5 pot).</div></div>
    <div><h2>🧠 Latency vs win rate</h2><canvas id="latVsWin"></canvas>
      <div class="note">Does thinking longer pay off? x = avg seconds per decision,
        y = bb/100.</div></div>
  </div>

  <h2>🏅 Made-hand mix at showdown</h2>
  <canvas id="madeHand"></canvas>
  <div class="note">Share of the model's showdown hands that finished as each category
    (high card → full house). Tight selectors reach showdown with stronger made hands.</div>

  <h2>⚔️ Head-to-head chip results</h2>
  <div class="sub">Net chips the row model won against the column model (sums to zero per pair).</div>
  <table class="h2h">{hh}</table>

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

// player-type scatter (VPIP x aggression)
new Chart(scatter, {{ type:'scatter',
  data:{{ datasets: MODELS.map(m=>({{ label:dn(m),
      data:[{{x:PM[m].vpip*100, y:PM[m].agg_freq*100}}],
      backgroundColor:mcol(m), pointRadius:9, pointHoverRadius:12 }})) }},
  options:{{ scales:{{ x:{{title:{{display:true,text:'VPIP % (loose →)'}},min:0,max:100}},
                       y:{{title:{{display:true,text:'aggression % (aggressive ↑)'}},min:0,max:100}} }},
    plugins:{{ legend:{{position:'bottom'}} }} }} }});

// bb/100 bar — colored by model (sign shown by value direction)
const ranked = [...MODELS].sort((a,b)=>PM[b].bb_per_100-PM[a].bb_per_100);
new Chart(bbchart, {{ type:'bar',
  data:{{ labels:ranked.map(dn), datasets:[{{ data:ranked.map(m=>PM[m].bb_per_100),
      backgroundColor:ranked.map(m=>mcol(m)) }}] }},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}} }} }});

// action distribution (stacked %)
const ACTS=['fold','check','call','bet','raise','all_in'];
const ACOL={{fold:'#6b7280',check:'#94a3b8',call:'#38bdf8',bet:'#fbbf24',raise:'#fb923c',all_in:'#f87171'}};
new Chart(actions, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:ACTS.map(a=>({{ label:a, backgroundColor:ACOL[a],
      data:MODELS.map(m=>{{const mix=PM[m].action_mix;const tot=Object.values(mix).reduce((x,y)=>x+y,0)||1;
        return 100*mix[a]/tot;}}) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,title:{{display:true,text:'% of actions'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// style radar (normalized 0-100)
new Chart(radar, {{ type:'radar',
  data:{{ labels:['loose (VPIP)','PF raise','aggression','all-in','bet size','calls bets (1-fold)'],
    datasets:MODELS.map(m=>({{label:dn(m),borderColor:mcol(m),
      backgroundColor:mcol(m)+'22',
      data:[PM[m].vpip*100,PM[m].pfr*100,PM[m].agg_freq*100,PM[m].allin_freq*100,
            Math.min(100,PM[m].avg_bet_xpot*50),(1-PM[m].fold_to_bet)*100]}})) }},
  options:{{ scales:{{r:{{min:0,max:100,ticks:{{display:false}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// latency
new Chart(latency, {{ type:'bar',
  data:{{ labels:MODELS.map(dn), datasets:[{{data:MODELS.map(m=>PM[m].avg_latency_s),
      backgroundColor:MODELS.map(m=>mcol(m))}}] }},
  options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{title:{{display:true,text:'seconds / decision'}}}}}} }} }});

// aggression by street (lines per model)
const STREETS=['preflop','flop','turn','river'];
new Chart(streetAgg, {{ type:'line',
  data:{{ labels:STREETS, datasets:MODELS.map(m=>({{
      label:dn(m), borderColor:mcol(m), backgroundColor:mcol(m),
      tension:0.25,
      data:STREETS.map(s=>PM[m].by_street[s].agg_freq*100) }})) }},
  options:{{ scales:{{y:{{title:{{display:true,text:'aggression %'}},min:0,max:100}}}},
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
