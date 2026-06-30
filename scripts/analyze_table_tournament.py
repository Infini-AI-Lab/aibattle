"""Analyze the Multi-Agent Table-mode tournament.

Reads runs/holdem_table/table_data.json (per-model summary: avg_rank,
top1_rate, avg_final_stack) and builds a finishing-place distribution. Writes a
Chart.js HTML report to runs/holdem_table/table_report.html and
reports/table_tournament_report.html plus reports/table_tournament_analysis.json.
Styling matches the board-game tournament report.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import poker_behavior as pb
from model_names import strip_coached, display_name, model_cell
from report_tokens import token_cost_cells, TOKEN_HEADERS, TOKEN_NOTE
from report_theme import BASE_CSS
from report_legends import legend as _legend

# Coached is now the canonical (and only) run set; data lives in per-game folders.
DATA = "runs/holdem_table/table_data.json"
EP_GLOB = "runs/holdem_table/table/ep*.json"
OUT_HTML = "runs/holdem_table/table_report.html"
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# pages include those two files in <head> via NAV_HEAD and the bar is injected
# by JS, so the nav markup lives in one place.
NAV_HEAD = '<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="nav.css?v=7"><script defer src="nav.js?v=32"></script>'

# Page-specific styles that used to ride along with the nav CSS.
EXTRA_CSS = ""

_STYLE = BASE_CSS


def analyze(data: dict) -> dict:
    models = data["models"]; n = data["num_players"]
    dist = {m: defaultdict(int) for m in models}
    busts = defaultdict(int)
    for s in data["session_results"]:
        for m, r in s["result"].items():
            dist[m][r["rank"]] += 1
            if r["final_stack"] <= 0:
                busts[m] += 1
    summary = data["summary"]
    sess = len(data["session_results"]) or 1
    for row in summary:
        m = row["model"]
        row["rank_distribution"] = {str(k): dist[m].get(k, 0) for k in range(1, n + 1)}
        row["bust_rate"] = round(busts[m] / sess, 3)
    # Rank by average finishing rank (lower is better); break ties by the higher
    # top-1 share. Avg rank is the central-tendency metric and rewards consistent
    # finishes, whereas top-1 rate over-rewards high-variance "boom-or-bust" play
    # (a model can lead on top-1 while busting a third of its sessions).
    summary.sort(key=lambda r: (r["avg_rank"], -r["top1_rate"]))
    return {"models": models, "num_players": n, "sessions": data["sessions"],
            "max_hands": data["max_hands"], "leaderboard": summary}


def render_html(rep: dict, beh: dict) -> str:
    n = rep["num_players"]; lb = rep["leaderboard"]
    labels = [r["model"] for r in lb]          # slugs — key behavior stats / colors
    disp_labels = [display_name(m) for m in labels]   # official names for chart axes
    avg_rank = [r["avg_rank"] for r in lb]
    top1 = [round(r["top1_rate"] * 100, 1) for r in lb]
    cols = pb.colors_for(labels)
    beh_html = pb.profile_table(beh, labels) + pb.behavior_charts(beh, labels)
    replay_btn = ('<a class="replaybtn" href="table_replay.html?v=17">'
                  '▶ watch table replays</a>')
    rankhdr = "".join(f"<th>#{k}</th>" for k in range(1, n + 1))
    trows = ""
    for i, r in enumerate(lb, 1):
        rd = r["rank_distribution"]
        rc = "".join(f"<td>{rd.get(str(k),0)}</td>" for k in range(1, n + 1))
        tables = sum(rd.get(str(k), 0) for k in range(1, n + 1))
        trows += (f"<tr><td>{i}</td><td class='model'>{model_cell(r['model'])}</td>"
                  f"<td>{r['avg_rank']}</td><td>{r['top1_rate']*100:.0f}%</td>"
                  f"<td>{r['avg_final_stack']}</td><td>{r['bust_rate']*100:.0f}%</td>"
                  f"<td>{tables}</td>{rc}"
                  f"{token_cost_cells(r['model'], beh.get(r['model'], {}).get('avg_tokens'))}</tr>")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Table Mode</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}<style>{EXTRA_CSS}{_STYLE}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/holdem/table<span class="cursor"></span></h1>
  <div class="sub">🃏 Hold'em Table · {n}-player table · {rep['sessions']} sessions · up to {rep['max_hands']} hands · ranked by average finishing rank (lower is better; ties broken by top-1 share). Top-1 rate is shown alongside but over-rewards high-variance play.</div>
  {replay_btn}
  <div class="rules">
    <h3>Setup — Hold'em Table</h3>
    Standard No-Limit
    <a href="https://en.wikipedia.org/wiki/Texas_hold_%27em" target="_blank" rel="noopener">Texas Hold'em</a>
    (full rules on Wikipedia); here it's a multi-way <b>ring game</b> rather than
    heads-up:
    <ul>
      <li><b>{n}-handed table:</b> {n} models sit at one table, each starting with
        <b>200 chips</b> (blinds <b>1 / 2</b>), and play a full sit-and-go.</li>
      <li><b>{rep['sessions']} sessions</b> of <b>up to {rep['max_hands']} hands</b>;
        within a session stacks carry and busted players are out — so finishing
        <b>place</b> (1st…{n}th) is what's recorded.</li>
      <li>Multi-way play adds position, bubble pressure and bust risk that heads-up
        doesn't have.</li>
    </ul>
    <div class="seq">Ranked by <b>average finishing rank</b> (lower is better), which
    rewards consistent finishes; top-1 rate is shown alongside but over-rewards
    high-variance, boom-or-bust play.</div>
    <div class="seq"><b>What the model sees each turn:</b> its own two hole cards, the community
    board, the pot and every player's stack, its position, the bet it faces, the legal actions, and
    the action history — never any opponent's cards.</div>
  </div>
  <div class="grid2">
    <div><h2>Average rank</h2><canvas id="ar"></canvas></div>
    <div><h2>Top-1 rate</h2><canvas id="t1"></canvas></div>
  </div>
  <h2>Leaderboard <span class="note">(+ finishing-place distribution)</span></h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>avg rank</th><th>top-1%</th>
        <th>avg final stack</th><th>bust%</th><th>tables</th>{rankhdr}{TOKEN_HEADERS}</tr>
    {trows}
  </table>
  {_legend('table')}
  {TOKEN_NOTE}
  {beh_html}
  <script>
  const axc={{grid:{{color:'#e7e2d8'}},ticks:{{color:'#1c1c1c'}}}};
  new Chart(document.getElementById('ar'),{{type:'bar',
    data:{{labels:{json.dumps(disp_labels)},datasets:[{{label:'avg rank',data:{json.dumps(avg_rank)},backgroundColor:{json.dumps(cols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,max:{n},...axc}},x:axc}}}}}});
  new Chart(document.getElementById('t1'),{{type:'bar',
    data:{{labels:{json.dumps(disp_labels)},datasets:[{{label:'top-1 %',data:{json.dumps(top1)},backgroundColor:{json.dumps(cols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,max:100,...axc}},x:axc}}}}}});
  </script>
</div></body></html>"""


def main():
    data = strip_coached(json.load(open(DATA)))
    rep = analyze(data)
    beh = pb.behavior(EP_GLOB, "table_hand", rep["models"])
    rep["behavior"] = beh
    html = render_html(rep, beh)
    os.makedirs(REPORT_DIR, exist_ok=True)
    for path in (OUT_HTML, os.path.join(REPORT_DIR, "table_tournament_report.html")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    json.dump(rep, open(os.path.join(REPORT_DIR, "table_tournament_analysis.json"), "w"),
              indent=2)
    print(f"Wrote {OUT_HTML} and {REPORT_DIR}/table_tournament_report.html\n")
    print(f"=== Table Mode ({rep['num_players']}p, {rep['sessions']} sessions) ===")
    print(f"{'model':<18} avg_rank top1%  avg_stack  bust%")
    for r in rep["leaderboard"]:
        print(f"{r['model']:<18} {r['avg_rank']:>7}  {r['top1_rate']*100:>3.0f}%  "
              f"{r['avg_final_stack']:>8}  {r['bust_rate']*100:>4.0f}%")


if __name__ == "__main__":
    main()
