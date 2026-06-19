"""Analyze the Heads-Up Match-mode tournament.

Reads runs/holdem_match/match_data.json and reports match win rate (the
primary metric), head-to-head grid, and match-shape stats (bust vs max-hands,
avg hands/match, avg final-stack margin). Writes a Chart.js HTML report to
runs/holdem_match/match_report.html and reports/match_tournament_report.html
plus the raw numbers to reports/match_tournament_analysis.json. Styling matches
the board-game tournament report.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import poker_behavior as pb
from model_names import strip_coached, display_name, model_cell
from elo_util import bradley_terry, elo_key, bootstrap_elo, wld_from_records
from report_theme import BASE_CSS, CHART_SETUP

# Coached is now the canonical (and only) run set; data lives in per-game folders.
DATA = "runs/holdem_match/match_data.json"
EP_GLOB = "runs/holdem_match/*__vs__*/ep*.json"
OUT_HTML = "runs/holdem_match/match_report.html"
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# pages include those two files in <head> via NAV_HEAD and the bar is injected
# by JS, so the nav markup lives in one place.
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=27"></script>'

# Page-specific styles that used to ride along with the nav CSS.
EXTRA_CSS = ""

_STYLE = BASE_CSS + """
  td.hh { font-weight:700; }
  td.hh .rec { display:block; font-weight:400; font-size:11px; color:var(--dim); margin-top:1px; }
"""


def analyze(data: dict) -> dict:
    models = data["models"]
    played = defaultdict(int); won = defaultdict(int); drew = defaultdict(int)
    stack_margin = defaultdict(float); hands = defaultdict(int); busts = defaultdict(int)
    h2h = {a: {b: 0 for b in models} for a in models}
    h2h_played = {a: {b: 0 for b in models} for a in models}
    elo_records = []  # per-match (a, b, result) for the Elo bootstrap

    for pair in data["pairs"]:
        for e in pair["episodes"]:
            seat = e["seat_assignment"]
            a, b = seat["player_0"], seat["player_1"]
            wname = e.get("winner_name")
            fs = e.get("final_stacks", {})
            for p in ("player_0", "player_1"):
                played[seat[p]] += 1
                hands[seat[p]] += e.get("hands_played", 0)
            if wname is None:
                drew[a] += 1; drew[b] += 1
                elo_records.append((a, b, 0))
            else:
                won[wname] += 1
                loser = b if wname == a else a
                h2h[wname][loser] += 1
                elo_records.append((a, b, 1 if wname == a else -1))
                if fs:
                    stack_margin[wname] += abs(fs.get("player_0", 0) - fs.get("player_1", 0))
                if e.get("reason") == "bust":
                    busts[loser] += 1
            h2h_played[a][b] += 1; h2h_played[b][a] += 1

    # Match mode is win-or-lose by design (chips don't carry meaning past the
    # match outcome), so the Elo is a Bradley-Terry fit over match W/L/D —
    # opponent-adjusted, fair when models faced different opponents.
    wld = {a: {b: (h2h[a][b], h2h[b][a],
                   h2h_played[a][b] - h2h[a][b] - h2h[b][a])
               for b in models if b != a} for a in models}
    _, elo = bradley_terry(models, wld)
    elo_ci = bootstrap_elo(models, elo_records, lambda s: wld_from_records(models, s))

    rows = []
    for m in models:
        n = played[m] or 1
        rows.append({
            "model": m, "matches": played[m], "elo": elo[m], "elo_sd": elo_ci[m]["sd"],
            "win_rate": round(won[m] / n, 3), "wins": won[m], "draws": drew[m],
            "busted_out_rate": round(busts[m] / n, 3),
            "avg_hands_per_match": round(hands[m] / n, 1),
            "avg_win_margin": round(stack_margin[m] / (won[m] or 1), 1),
        })
    rows.sort(key=lambda r: (elo_key(elo, r["model"]), r["win_rate"]), reverse=True)
    return {"models": models, "max_hands": data.get("max_hands"),
            "episodes_per_pair": data.get("episodes_per_pair"), "elo": elo,
            "leaderboard": rows, "h2h_wins": h2h, "h2h_played": h2h_played}


def render_html(rep: dict, beh: dict) -> str:
    models = rep["models"]; lb = rep["leaderboard"]
    labels = [r["model"] for r in lb]          # slugs — key behavior stats / colors
    disp_labels = [display_name(m) for m in labels]   # official names for chart axis
    winpct = [round(r["win_rate"] * 100, 1) for r in lb]
    wincols = pb.colors_for(labels)
    beh_html = pb.profile_table(beh, labels) + pb.behavior_charts(beh, labels)
    replay_btn = ('<a class="replaybtn" href="match_replay.html?v=17">'
                  '▶ watch match replays</a>')

    trows = ""
    for i, r in enumerate(lb, 1):
        if r.get("elo") is None:
            elo_disp = "—"
        elif r.get("elo_sd") is not None:
            elo_disp = f"{r['elo']}<div class='small'>±{r['elo_sd']:.0f}</div>"
        else:
            elo_disp = str(r["elo"])
        trows += (f"<tr><td>{i}</td><td class='model'>{model_cell(r['model'])}</td>"
                  f"<td><b>{elo_disp}</b></td>"
                  f"<td>{r['win_rate']*100:.0f}%</td><td>{r['wins']}/{r['matches']}</td>"
                  f"<td>{r['draws']}</td><td>{r['busted_out_rate']*100:.0f}%</td>"
                  f"<td>{r['avg_hands_per_match']}</td><td>{r['avg_win_margin']}</td></tr>")
    head = "".join(f"<th>{display_name(m)}</th>" for m in models)
    grid = ""
    for a in models:
        cells = ""
        for b in models:
            if a == b:
                cells += "<td class='diag'>—</td>"
                continue
            w = rep['h2h_wins'][a][b]; pl = rep['h2h_played'][a][b]
            if not pl:
                cells += "<td class='hh'>—</td>"
                continue
            pct = 100 * w / pl
            # Diverging red→green heatmap centred on 50% (an even split shows no
            # tint); alpha grows with the distance from even so lopsided cells pop.
            alpha = round(0.6 * abs(pct - 50) / 50, 3)
            rgb = "34,197,94" if pct >= 50 else "244,63,94"
            cells += (f"<td class='hh' style='background:rgba({rgb},{alpha})'>"
                      f"{pct:.0f}%<span class='rec'>{w}/{pl}</span></td>")
        grid += f"<tr><td class='model'>{model_cell(a)}</td>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Match Mode</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}<style>{EXTRA_CSS}{_STYLE}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/holdem/match<span class="cursor"></span></h1>
  <div class="sub">🃏 Hold'em Match · Heads-up · {rep['episodes_per_pair']} matches/pair · up to {rep['max_hands']} hands/match · stacks carried, match-level winner · primary metric: match win rate</div>
  {replay_btn}
  <div class="rules">
    <h3>Setup — Hold'em Match</h3>
    Standard heads-up No-Limit
    <a href="https://en.wikipedia.org/wiki/Texas_hold_%27em" target="_blank" rel="noopener">Texas Hold'em</a>
    (full rules on Wikipedia); the difference from 1-Hand is that here a whole
    <b>sit-and-go match</b> is the unit, not a single hand:
    <ul>
      <li><b>Heads-up sit-and-go:</b> both start with <b>200 chips</b> (blinds
        <b>1 / 2</b>) and play until one is busted, or until a cap of <b>up to
        {rep['max_hands']} hands</b>.</li>
      <li><b>Stacks carry across hands</b> within a match — winning chips early creates a
        real lead, so position and stack pressure matter.</li>
      <li><b>{rep['episodes_per_pair']} matches per pair</b>, seats swapped; the
        <b>match winner</b> is whoever busts the other (or leads at the cap).</li>
    </ul>
    <div class="seq">Win-or-lose by design — chips don't count past the match outcome —
    so the <b>Elo rates match wins/losses</b>, opponent-adjusted. Match win rate is the
    headline metric.</div>
  </div>
  <h2>Match win rate</h2>
  <canvas id="wr"></canvas>
  <h2>Leaderboard <span class="note">(ranked by Elo; raw metrics kept for reference)</span></h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>Elo</th><th>win%</th><th>wins/matches</th>
        <th>draws</th><th>bust-out%</th><th>hands/match</th><th>avg win margin</th></tr>
    {trows}
  </table>
  <div class="note"><b>Elo</b> = Bradley-Terry rating (field mean 1500) over match win/loss results.
    Match mode is win-or-lose — chips don't count past who took the match — so the rating uses match
    outcomes only, opponent-adjusted. ± is one bootstrap SD (resampling matches 300×); ratings within
    ±1 of each other are a statistical tie. win% and the rest are raw, unadjusted metrics.</div>
  <h2>Head-to-head <span class="note">(row's match win % vs column — green = winning, red = losing; raw record below)</span></h2>
  <table><tr><th class='model'></th>{head}</tr>{grid}</table>
  {beh_html}
  <script>
  new Chart(document.getElementById('wr'), {{
    type:'bar',
    data:{{labels:{json.dumps(disp_labels)},datasets:[{{label:'win %',data:{json.dumps(winpct)},backgroundColor:{json.dumps(wincols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},
      scales:{{y:{{beginAtZero:true,max:100,grid:{{color:'#e7e2d8'}},ticks:{{color:'#1c1c1c'}}}},
               x:{{grid:{{color:'#e7e2d8'}},ticks:{{color:'#1c1c1c'}}}}}}}}
  }});
  </script>
</div></body></html>"""


def main():
    data = strip_coached(json.load(open(DATA)))
    rep = analyze(data)
    beh = pb.behavior(EP_GLOB, "match_hand", rep["models"])
    rep["behavior"] = beh
    html = render_html(rep, beh)
    os.makedirs(REPORT_DIR, exist_ok=True)
    for path in (OUT_HTML, os.path.join(REPORT_DIR, "match_tournament_report.html")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    json.dump(rep, open(os.path.join(REPORT_DIR, "match_tournament_analysis.json"), "w"),
              indent=2)
    print(f"Wrote {OUT_HTML} and {REPORT_DIR}/match_tournament_report.html\n")
    print(f"=== Match Mode ({rep['episodes_per_pair']}/pair, {rep['max_hands']} hands) ===")
    print(f"{'model':<18} win%   wins      bust%  hands/m  margin")
    for r in rep["leaderboard"]:
        print(f"{r['model']:<18} {r['win_rate']*100:>3.0f}%  {r['wins']:>3}/{r['matches']:<3}  "
              f"{r['busted_out_rate']*100:>4.0f}%  {r['avg_hands_per_match']:>6}  {r['avg_win_margin']:>6}")


if __name__ == "__main__":
    main()
