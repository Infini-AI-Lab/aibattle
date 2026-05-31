"""Analyze the tournament data and emit a self-contained interactive HTML report.

Reads runs/tournament/tournament_data.json (games -> episodes -> steps) and
computes per-model poker behavior + results, then writes runs/tournament/report.html.

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

DATA = "runs/tournament/tournament_data.json"
OUT = "runs/tournament/report.html"
# Tracked copy committed to the repo (runs/ is gitignored).
REPORT_DIR = "reports"
BB = 2


def _blank():
    return {
        "hands": 0, "decisions": 0, "chips": 0.0,
        "wins": 0, "losses": 0, "ties": 0,
        "showdown_wins": 0, "fold_wins": 0,
        "acts": defaultdict(int),
        "facing_bet": 0, "fold_facing_bet": 0,
        "vpip_hands": 0, "pfr_hands": 0,
        "betsize_ratios": [], "latencies": [],
        "invalid_actions": 0, "invalid_amounts": 0,
    }


_AMOUNT_REASONS = {"missing_amount", "non_integer_amount", "below_minimum",
                   "above_stack", "unexpected_amount"}


def analyze(data: dict) -> dict:
    models = data["models"]
    stats = {m: _blank() for m in models}
    h2h = defaultdict(lambda: defaultdict(float))  # h2h[a][b] = chips a won vs b
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
            h2h[a][b] += returns[name_seat[a]]
            h2h[b][a] += returns[name_seat[b]]
            if winner_name and reason == "showdown":
                stats[winner_name]["showdown_wins"] += 1
            elif winner_name and reason == "fold":
                stats[winner_name]["fold_wins"] += 1

            # step-level behavior; track per-(player) preflop voluntary action
            vpip_seen = {}      # name -> bool voluntarily put money in preflop
            pfr_seen = {}       # name -> bool raised preflop
            for s in e["steps"]:
                nm = s["agent_name"]
                st = stats[nm]
                st["decisions"] += 1
                act = s["selected_action"]
                st["acts"][act] += 1
                pub = s["observation"]["public"]
                to_call = pub.get("to_call", 0)
                street = pub.get("street")
                pot = pub.get("pot", 0) or 0
                sc = pub.get("your_street_commit", 0) or 0

                if to_call > 0:
                    st["facing_bet"] += 1
                    if act == "fold":
                        st["fold_facing_bet"] += 1

                if street == "preflop":
                    if act in ("call", "bet", "raise", "all_in"):
                        vpip_seen[nm] = True
                    if act in ("raise", "all_in", "bet"):
                        pfr_seen[nm] = True

                if act in ("bet", "raise", "all_in"):
                    amt = s.get("selected_amount")
                    invested = (amt - sc) if amt is not None else None
                    if invested is None and act == "all_in":
                        invested = pub.get("your_stack", 0)
                    if invested and pot > 0:
                        st["betsize_ratios"].append(invested / pot)

                lat = (s.get("response") or {}).get("metadata", {}).get("latency_ms")
                if lat:
                    st["latencies"].append(lat)

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
            "invalid_actions": st["invalid_actions"],
            "invalid_amounts": st["invalid_amounts"],
            "action_mix": {k: st["acts"][k] for k in
                           ("fold", "check", "call", "bet", "raise", "all_in")},
            "style": _style(st["vpip_hands"] / n, agg_freq, st["acts"]["all_in"] / d),
        }
    h2h_out = {a: {b: round(h2h[a][b], 1) for b in models if b != a} for a in models}
    return {"models": models, "per_model": out_models, "h2h": h2h_out,
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

    # ranked leaderboard by bb/100
    ranked = sorted(models, key=lambda m: pm[m]["bb_per_100"], reverse=True)
    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        chip_cls = "pos" if s["chips"] > 0 else ("neg" if s["chips"] < 0 else "")
        tags = " ".join(f"<span class='tag'>{t}</span>" for t in s["style"]["tags"])
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{m}</td>
          <td><b>{s['style']['label']}</b> {tags}</td>
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
        </tr>"""

    # head-to-head matrix
    h2h = report["h2h"]
    hh = "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in models) + "</tr>"
    for a in models:
        hh += f"<tr><th class='model'>{a}</th>"
        for b in models:
            if a == b:
                hh += "<td class='diag'>—</td>"
            else:
                v = h2h[a].get(b, 0)
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                hh += f"<td class='{cls}' style='--v:{v}'>{v:+.0f}</td>"
        hh += "</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Tournament</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
          background: #0f1117; color: #e6e6e6; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 22px 80px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 38px 0 12px; border-bottom: 1px solid #2a2f3a;
        padding-bottom: 6px; }}
  .sub {{ color: #8b93a7; margin-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 7px 9px; text-align: center; border-bottom: 1px solid #20242e; }}
  th {{ color: #9aa3b5; font-weight: 600; }}
  td.model, th.model {{ text-align: left; font-weight: 600; color: #cdd6f4; }}
  .pos {{ color: #4ade80; }} .neg {{ color: #f87171; }}
  .diag {{ color: #3a3f4b; }}
  .tag {{ background: #2a2f3a; color: #a5b4fc; border-radius: 10px; padding: 1px 8px;
          font-size: 11px; margin-left: 3px; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ background: #161a22; border: 1px solid #232838; border-radius: 12px;
           padding: 16px; }}
  canvas {{ max-height: 340px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  .note {{ color: #8b93a7; font-size: 12px; margin-top: 6px; }}
  @media (max-width: 760px) {{ .grid2, .cards {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🃏 AI Battle Arena — Heads-Up Hold'em Tournament</h1>
  <div class="sub">{report['num_games']} games · {report['hands_per_game']} hands each · {len(models)} models · round-robin</div>

  <h2>🏆 Leaderboard &amp; player profiles</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>style</th><th>chips</th><th>bb/100</th>
        <th>win%</th><th>VPIP</th><th>PFR</th><th>aggr</th><th>fold→bet</th>
        <th>all-in%</th><th>bet size</th><th>think</th></tr>
    {rows}
  </table>
  <div class="note">VPIP = how often it voluntarily plays a hand (looseness). PFR = preflop raise %.
    aggr = aggression frequency. fold→bet = how often it folds when bet at. bet size = avg bet as a multiple of the pot.</div>

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

  <h2>⚔️ Head-to-head chip results</h2>
  <div class="sub">Net chips the row model won against the column model (sums to zero per pair).</div>
  <table class="h2h">{hh}</table>

<script>
const R = {payload};
const MODELS = R.models, PM = R.per_model;
const COLORS = ['#60a5fa','#f472b6','#4ade80','#fbbf24','#a78bfa','#22d3ee'];
const col = i => COLORS[i % COLORS.length];
const cssText = getComputedStyle(document.body).color;
Chart.defaults.color = '#9aa3b5'; Chart.defaults.borderColor = '#232838';

// player-type scatter (VPIP x aggression)
new Chart(scatter, {{ type:'scatter',
  data:{{ datasets: MODELS.map((m,i)=>({{ label:m,
      data:[{{x:PM[m].vpip*100, y:PM[m].agg_freq*100}}],
      backgroundColor:col(i), pointRadius:9, pointHoverRadius:12 }})) }},
  options:{{ scales:{{ x:{{title:{{display:true,text:'VPIP % (loose →)'}},min:0,max:100}},
                       y:{{title:{{display:true,text:'aggression % (aggressive ↑)'}},min:0,max:100}} }},
    plugins:{{ legend:{{position:'bottom'}} }} }} }});

// bb/100 bar
const ranked = [...MODELS].sort((a,b)=>PM[b].bb_per_100-PM[a].bb_per_100);
new Chart(bbchart, {{ type:'bar',
  data:{{ labels:ranked, datasets:[{{ data:ranked.map(m=>PM[m].bb_per_100),
      backgroundColor:ranked.map(m=>PM[m].bb_per_100>=0?'#4ade80':'#f87171') }}] }},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}} }} }});

// action distribution (stacked %)
const ACTS=['fold','check','call','bet','raise','all_in'];
const ACOL={{fold:'#6b7280',check:'#94a3b8',call:'#38bdf8',bet:'#fbbf24',raise:'#fb923c',all_in:'#f87171'}};
new Chart(actions, {{ type:'bar',
  data:{{ labels:MODELS, datasets:ACTS.map(a=>({{ label:a, backgroundColor:ACOL[a],
      data:MODELS.map(m=>{{const mix=PM[m].action_mix;const tot=Object.values(mix).reduce((x,y)=>x+y,0)||1;
        return 100*mix[a]/tot;}}) }})) }},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,title:{{display:true,text:'% of actions'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// style radar (normalized 0-100)
new Chart(radar, {{ type:'radar',
  data:{{ labels:['loose (VPIP)','PF raise','aggression','all-in','bet size','calls bets (1-fold)'],
    datasets:MODELS.map((m,i)=>({{label:m,borderColor:col(i),
      backgroundColor:col(i)+'22',
      data:[PM[m].vpip*100,PM[m].pfr*100,PM[m].agg_freq*100,PM[m].allin_freq*100,
            Math.min(100,PM[m].avg_bet_xpot*50),(1-PM[m].fold_to_bet)*100]}})) }},
  options:{{ scales:{{r:{{min:0,max:100,ticks:{{display:false}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

// latency
new Chart(latency, {{ type:'bar',
  data:{{ labels:MODELS, datasets:[{{data:MODELS.map(m=>PM[m].avg_latency_s),
      backgroundColor:MODELS.map((m,i)=>col(i))}}] }},
  options:{{ plugins:{{legend:{{display:false}}}}, scales:{{y:{{title:{{display:true,text:'seconds / decision'}}}}}} }} }});
</script>
</div></body></html>"""


def main():
    data = json.load(open(DATA))
    report = analyze(data)
    html = render_html(report)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    # also dump computed stats as json for inspection
    json.dump(report, open("runs/tournament/analysis.json", "w"), indent=2)
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
