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
from model_names import strip_coached

# Coached is now the canonical (and only) run set; data lives in per-game folders.
DATA = "runs/holdem_match/match_data.json"
EP_GLOB = "runs/holdem_match/*__vs__*/ep*.json"
OUT_HTML = "runs/holdem_match/match_report.html"
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# pages include those two files in <head> via NAV_HEAD and the bar is injected
# by JS, so the nav markup lives in one place.
NAV_HEAD = '<link rel="stylesheet" href="nav.css"><script defer src="nav.js"></script>'

# Page-specific styles that used to ride along with the nav CSS.
EXTRA_CSS = """
  .replaybtn { display:inline-block; margin-top:12px; background:#1b2030; color:#a5b4fc;
    border:1px solid #2a2f3a; border-radius:8px; padding:8px 14px; font-size:13px; text-decoration:none; }
  .replaybtn:hover { border-color:#60a5fa; color:#fff; }
"""

_STYLE = """
  body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }
  .wrap { max-width:1200px; margin:0 auto; padding:28px 28px 80px; }
  h1 { font-size:25px; } h2 { font-size:19px; margin-top:40px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }
  .sub { color:#8b93a7; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin-top:10px; }
  th,td { padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }
  th { color:#9aa3b5; } td.model,th.model { text-align:left; font-weight:600; color:#cdd6f4; }
  .pos { color:#4ade80; } .neg { color:#f87171; } .diag { color:#3a3f4b; }
  .note { color:#8b93a7; font-size:12px; margin:6px 0; }
  td.hh { font-weight:600; color:#f3f4f6; }
  td.hh .rec { display:block; font-weight:400; font-size:11px; color:#aab2c5; margin-top:1px; }
  canvas { max-height:300px; margin-top:10px; }
"""


def analyze(data: dict) -> dict:
    models = data["models"]
    played = defaultdict(int); won = defaultdict(int); drew = defaultdict(int)
    stack_margin = defaultdict(float); hands = defaultdict(int); busts = defaultdict(int)
    h2h = {a: {b: 0 for b in models} for a in models}
    h2h_played = {a: {b: 0 for b in models} for a in models}

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
            else:
                won[wname] += 1
                loser = b if wname == a else a
                h2h[wname][loser] += 1
                if fs:
                    stack_margin[wname] += abs(fs.get("player_0", 0) - fs.get("player_1", 0))
                if e.get("reason") == "bust":
                    busts[loser] += 1
            h2h_played[a][b] += 1; h2h_played[b][a] += 1

    rows = []
    for m in models:
        n = played[m] or 1
        rows.append({
            "model": m, "matches": played[m],
            "win_rate": round(won[m] / n, 3), "wins": won[m], "draws": drew[m],
            "busted_out_rate": round(busts[m] / n, 3),
            "avg_hands_per_match": round(hands[m] / n, 1),
            "avg_win_margin": round(stack_margin[m] / (won[m] or 1), 1),
        })
    rows.sort(key=lambda r: r["win_rate"], reverse=True)
    return {"models": models, "max_hands": data.get("max_hands"),
            "episodes_per_pair": data.get("episodes_per_pair"),
            "leaderboard": rows, "h2h_wins": h2h, "h2h_played": h2h_played}


def render_html(rep: dict, beh: dict) -> str:
    models = rep["models"]; lb = rep["leaderboard"]
    labels = [r["model"] for r in lb]
    winpct = [round(r["win_rate"] * 100, 1) for r in lb]
    wincols = pb.colors_for(labels)
    beh_html = pb.profile_table(beh, labels) + pb.behavior_charts(beh, labels)
    # Coached runs have no replay viewer built; omit the button so it never 404s.
    replay_btn = ""

    trows = ""
    for i, r in enumerate(lb, 1):
        trows += (f"<tr><td>{i}</td><td class='model'>{r['model']}</td>"
                  f"<td>{r['win_rate']*100:.0f}%</td><td>{r['wins']}/{r['matches']}</td>"
                  f"<td>{r['draws']}</td><td>{r['busted_out_rate']*100:.0f}%</td>"
                  f"<td>{r['avg_hands_per_match']}</td><td>{r['avg_win_margin']}</td></tr>")
    head = "".join(f"<th>{m.split('-')[0]}</th>" for m in models)
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
        grid += f"<tr><td class='model'>{a}</td>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Match Mode</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}<style>{EXTRA_CSS}{_STYLE}</style></head>
<body><div class="wrap">
  <h1>🃏 AI Battle Arena — Hold'em Match Mode</h1>
  <div class="sub">Heads-up · {rep['episodes_per_pair']} matches/pair · up to {rep['max_hands']} hands/match · stacks carried, match-level winner · primary metric: match win rate</div>
  {replay_btn}
  <h2>Match win rate</h2>
  <canvas id="wr"></canvas>
  <h2>Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>win%</th><th>wins/matches</th>
        <th>draws</th><th>bust-out%</th><th>hands/match</th><th>avg win margin</th></tr>
    {trows}
  </table>
  <h2>Head-to-head <span class="note">(row's match win % vs column — green = winning, red = losing; raw record below)</span></h2>
  <table><tr><th class='model'></th>{head}</tr>{grid}</table>
  {beh_html}
  <script>
  new Chart(document.getElementById('wr'), {{
    type:'bar',
    data:{{labels:{json.dumps(labels)},datasets:[{{label:'win %',data:{json.dumps(winpct)},backgroundColor:{json.dumps(wincols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},
      scales:{{y:{{beginAtZero:true,max:100,grid:{{color:'#20242e'}},ticks:{{color:'#9aa3b5'}}}},
               x:{{grid:{{color:'#20242e'}},ticks:{{color:'#9aa3b5'}}}}}}}}
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
