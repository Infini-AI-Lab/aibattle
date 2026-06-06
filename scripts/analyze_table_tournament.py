"""Analyze the Multi-Agent Table-mode tournament.

Reads runs/table_tournament/table_data.json (per-model summary: avg_rank,
top1_rate, avg_final_stack) and builds a finishing-place distribution. Writes a
Chart.js HTML report to runs/table_tournament/table_report.html and
reports/table_tournament_report.html plus reports/table_tournament_analysis.json.
Styling matches the board-game tournament report.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import poker_behavior as pb

DATA = "runs/table_tournament/table_data.json"
EP_GLOB = "runs/table_tournament/table/ep*.json"
OUT_HTML = "runs/table_tournament/table_report.html"
REPORT_DIR = "reports"

NAV_CSS = """
  .navbar { position:sticky; top:0; z-index:50; display:flex; align-items:center;
    flex-wrap:wrap; gap:6px 18px; padding:0 22px; min-height:52px;
    background:rgba(12,14,20,.92); backdrop-filter:blur(8px);
    border-bottom:1px solid #232838; }
  .navbar .brand { font-weight:700; color:#cdd6f4; text-decoration:none;
    font-size:15px; margin-right:10px; }
  .navbar a.nav { color:#9aa3b5; text-decoration:none; font-size:13px;
    padding:16px 2px; border-bottom:2px solid transparent; }
  .navbar a.nav:hover { color:#e6e6e6; }
  .navbar a.nav.active { color:#fff; border-bottom-color:#60a5fa; }
  .navbar .navgrp { font-size:10px; letter-spacing:.08em; text-transform:uppercase;
    color:#6b7280; align-self:center; padding-left:12px; margin-left:2px;
    border-left:1px solid #2a3142; }
  .navbar .navclust { font-size:13px; color:#8b93a7; align-self:center; margin-left:4px; }
  .navbar a.navarena { text-decoration:none; }
  .navbar a.navarena:hover { color:#cbd5e1; }
  .navbar .soon { font-size:9px; color:#0b1020; background:#6b7280; border-radius:999px;
    padding:1px 6px; margin-left:6px; letter-spacing:.03em; }
  .replaybtn { display:inline-block; margin-top:12px; background:#1b2030; color:#a5b4fc;
    border:1px solid #2a2f3a; border-radius:8px; padding:8px 14px; font-size:13px; text-decoration:none; }
  .replaybtn:hover { border-color:#60a5fa; color:#fff; }
"""


# Top-level grouping mirrors the overview's primary axis — the two arenas. The
# Model Arena lists its game report pages (Hold'em's three variants cluster under
# one label); the Agentic Arena has no pages yet, so it links to the overview's
# #agentic section and is marked "soon".
def _navbar(active: str) -> str:
    def link(href, label, key):
        cls = "nav active" if key == active else "nav"
        return f"<a class='{cls}' href='{href}'>{label}</a>"
    return ("<nav class='navbar'>"
            "<a class='brand' href='index.html'>🎲 AI Battle Arena</a>"
            + link("index.html", "Overview", "overview")
            + "<a class='navgrp navarena' href='index.html#model'>Model Arena</a>"
            + link("connect4_report.html", "🔴 Connect Four", "connect4")
            + link("gomoku_report.html", "⚫ Gomoku", "gomoku")
            + link("kuhn_tournament_report.html", "🃏 Kuhn", "kuhn")
            + "<span class='navclust'>🃏 Hold'em</span>"
            + link("holdem_tournament_report.html", "1-Hand", "holdem")
            + link("match_tournament_report.html", "Match", "match")
            + link("table_tournament_report.html", "Table", "table")
            + "<a class='navgrp navarena' href='index.html#agentic'>Agentic Arena"
              "<span class='soon'>soon</span></a>"
            + "</nav>")

_STYLE = """
  body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }
  .wrap { max-width:1200px; margin:0 auto; padding:28px 28px 80px; }
  h1 { font-size:25px; } h2 { font-size:19px; margin-top:40px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }
  .sub { color:#8b93a7; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin-top:10px; }
  th,td { padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }
  th { color:#9aa3b5; } td.model,th.model { text-align:left; font-weight:600; color:#cdd6f4; }
  .note { color:#8b93a7; font-size:12px; margin:6px 0; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:10px; }
  canvas { max-height:300px; }
  @media (max-width:760px) { .grid2 { grid-template-columns:1fr; } }
"""


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
    labels = [r["model"] for r in lb]
    avg_rank = [r["avg_rank"] for r in lb]
    top1 = [round(r["top1_rate"] * 100, 1) for r in lb]
    cols = pb.colors_for(labels)
    beh_html = pb.profile_table(beh, labels) + pb.behavior_charts(beh, labels)
    rankhdr = "".join(f"<th>#{k}</th>" for k in range(1, n + 1))
    trows = ""
    for i, r in enumerate(lb, 1):
        rd = r["rank_distribution"]
        rc = "".join(f"<td>{rd.get(str(k),0)}</td>" for k in range(1, n + 1))
        trows += (f"<tr><td>{i}</td><td class='model'>{r['model']}</td>"
                  f"<td>{r['avg_rank']}</td><td>{r['top1_rate']*100:.0f}%</td>"
                  f"<td>{r['avg_final_stack']}</td><td>{r['bust_rate']*100:.0f}%</td>{rc}</tr>")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Table Mode</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>{NAV_CSS}{_STYLE}</style></head>
<body>{_navbar("table")}<div class="wrap">
  <h1>🃏 AI Battle Arena — Hold'em Table Mode</h1>
  <div class="sub">{n}-player table · {rep['sessions']} sessions · up to {rep['max_hands']} hands · ranked by average finishing rank (lower is better; ties broken by top-1 share). Top-1 rate is shown alongside but over-rewards high-variance play.</div>
  <a class="replaybtn" href="table_replay.html">▶ Watch table replays</a>
  <div class="grid2">
    <div><h2>Average rank</h2><canvas id="ar"></canvas></div>
    <div><h2>Top-1 rate</h2><canvas id="t1"></canvas></div>
  </div>
  <h2>Leaderboard <span class="note">(+ finishing-place distribution)</span></h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>avg rank</th><th>top-1%</th>
        <th>avg final stack</th><th>bust%</th>{rankhdr}</tr>
    {trows}
  </table>
  {beh_html}
  <script>
  const axc={{grid:{{color:'#20242e'}},ticks:{{color:'#9aa3b5'}}}};
  new Chart(document.getElementById('ar'),{{type:'bar',
    data:{{labels:{json.dumps(labels)},datasets:[{{label:'avg rank',data:{json.dumps(avg_rank)},backgroundColor:{json.dumps(cols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,max:{n},...axc}},x:axc}}}}}});
  new Chart(document.getElementById('t1'),{{type:'bar',
    data:{{labels:{json.dumps(labels)},datasets:[{{label:'top-1 %',data:{json.dumps(top1)},backgroundColor:{json.dumps(cols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,max:100,...axc}},x:axc}}}}}});
  </script>
</div></body></html>"""


def main():
    data = json.load(open(DATA))
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
