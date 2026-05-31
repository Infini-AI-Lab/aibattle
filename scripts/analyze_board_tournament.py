"""Analyze the board-game tournament (Connect Four + Gomoku) and emit an
interactive HTML report — the board-game counterpart of analyze_tournament.py.

Poker's VPIP/aggression don't apply; the analogous *skill* signal here is
TACTICAL ACCURACY, computed directly from the board at every decision:

  - win-take rate : when an immediate winning move existed, did the model play one?
  - block rate    : when the opponent had an immediate winning threat (and the
                    model could not just win), did the model's move remove it?
  - blunders      : missed wins + allowed losses (failed blocks)

Plus results (win/draw/loss, net result/game, first-mover win rate), invalid-move
rate, average game length, latency, a head-to-head win matrix, and a move-location
heatmap. Reads runs/board_tournament/<game>_data.json; writes report HTML to
runs/board_tournament/ and a tracked copy under reports/.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from aibattle.games.board import connects, with_cell
from aibattle.games.gomoku import coord_to_rc

GAMES = ["connect4", "gomoku"]
NEED = {"connect4": 4, "gomoku": 5}
DATA_DIR = "runs/board_tournament"
OUT = os.path.join(DATA_DIR, "board_report.html")
REPORT_DIR = "reports"
PLAYERS = ["player_0", "player_1"]


def _other(p):
    return PLAYERS[1 - PLAYERS.index(p)]


def _grid(board):
    return tuple(tuple(row) for row in board)


def _landing(game, grid, move):
    """(r, c) where `move` would place a piece, or None if not a legal landing."""
    if game == "connect4":
        try:
            c = int(move)
        except (ValueError, TypeError):
            return None
        if not (0 <= c < len(grid[0])):
            return None
        for r in range(len(grid) - 1, -1, -1):
            if grid[r][c] is None:
                return (r, c)
        return None
    return coord_to_rc(move)


def _wins_at(game, grid, move, player):
    rc = _landing(game, grid, move)
    if rc is None or grid[rc[0]][rc[1]] is not None:
        return False
    ng = with_cell(grid, rc[0], rc[1], player)
    return connects(ng, rc[0], rc[1], player, NEED[game])


def _has_immediate_win(game, grid, player):
    """True if `player` has any one-move win on `grid`."""
    rows, cols = len(grid), len(grid[0])
    if game == "connect4":
        return any(_wins_at(game, grid, str(c), player) for c in range(cols))
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                ng = with_cell(grid, r, c, player)
                if connects(ng, r, c, player, NEED[game]):
                    return True
    return False


def _blank(size):
    return {
        "games": 0, "wins": 0, "losses": 0, "draws": 0, "net": 0.0,
        "decisions": 0, "invalid": 0,
        "first_moves": 0, "first_move_wins": 0,
        "win_opps": 0, "win_takes": 0,           # had an immediate win / took it
        "block_opps": 0, "blocks": 0,            # forced to defend / defended
        "lengths": [], "latencies": [],
        "heat": [[0] * size[1] for _ in range(size[0])],  # move-location counts
    }


def analyze_game(game: str, data: dict) -> dict:
    models = data["models"]
    # board size from the first observation
    sample = data["games"][0]["episodes"][0]["steps"][0]["observation"]["public"]["board"]
    size = (len(sample), len(sample[0]))
    stats = {m: _blank(size) for m in models}
    h2h = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [w,l,d] for a vs b

    for g in data["games"]:
        a, b = g["a"], g["b"]
        for e in g["episodes"]:
            seat_name = e["seat_assignment"]
            winner_name = e.get("winner_name")
            steps = e["steps"]

            # per-model game result
            for seat, nm in seat_name.items():
                st = stats[nm]
                st["games"] += 1
                st["lengths"].append(e["length"])
                pay = e["returns"][seat]
                st["net"] += pay
                if pay > 0:
                    st["wins"] += 1
                elif pay < 0:
                    st["losses"] += 1
                else:
                    st["draws"] += 1
            # head-to-head (by model name)
            if winner_name == a:
                h2h[a][b][0] += 1; h2h[b][a][1] += 1
            elif winner_name == b:
                h2h[b][a][0] += 1; h2h[a][b][1] += 1
            else:
                h2h[a][b][2] += 1; h2h[b][a][2] += 1

            # first mover (agent of the first recorded decision)
            if steps:
                fm = steps[0]["agent_name"]
                stats[fm]["first_moves"] += 1
                if winner_name == fm:
                    stats[fm]["first_move_wins"] += 1

            # per-decision behavior + tactical accuracy
            for s in steps:
                nm = s["agent_name"]
                seat = s["player"]
                st = stats[nm]
                st["decisions"] += 1
                if s.get("invalid"):
                    st["invalid"] += 1
                lat = (s.get("response") or {}).get("metadata", {}).get("latency_ms")
                if lat:
                    st["latencies"].append(lat)

                grid = _grid(s["observation"]["public"]["board"])
                move = s["selected_action"]
                rc = _landing(game, grid, move)
                if rc is not None:
                    st["heat"][rc[0]][rc[1]] += 1

                me, opp = seat, _other(seat)
                had_win = _has_immediate_win(game, grid, me)
                if had_win:
                    st["win_opps"] += 1
                    if _wins_at(game, grid, move, me):
                        st["win_takes"] += 1

                opp_threat = _has_immediate_win(game, grid, opp)
                if opp_threat and not had_win:
                    # forced to defend: did this move remove the opponent's win?
                    st["block_opps"] += 1
                    if rc is not None:
                        ng = with_cell(grid, rc[0], rc[1], me)
                        if not _has_immediate_win(game, ng, opp):
                            st["blocks"] += 1

    out = {}
    for m in models:
        st = stats[m]
        n = max(st["games"], 1)
        out[m] = {
            "games": st["games"], "wins": st["wins"], "losses": st["losses"],
            "draws": st["draws"],
            "win_rate": round(st["wins"] / n, 4),
            "draw_rate": round(st["draws"] / n, 4),
            "net_per_game": round(st["net"] / n, 4),
            "first_move_win_rate": round(
                st["first_move_wins"] / max(st["first_moves"], 1), 4),
            "win_take_rate": round(st["win_takes"] / max(st["win_opps"], 1), 4),
            "win_opps": st["win_opps"],
            "block_rate": round(st["blocks"] / max(st["block_opps"], 1), 4),
            "block_opps": st["block_opps"],
            "missed_wins": st["win_opps"] - st["win_takes"],
            "allowed_losses": st["block_opps"] - st["blocks"],
            "invalid_rate": round(st["invalid"] / max(st["decisions"], 1), 4),
            "decisions": st["decisions"],
            "avg_len": round(sum(st["lengths"]) / len(st["lengths"]), 2)
                       if st["lengths"] else 0,
            "avg_latency_s": round(sum(st["latencies"]) / len(st["latencies"]) / 1000, 1)
                             if st["latencies"] else 0.0,
            "heat": st["heat"],
        }
    h2h_out = {a: {b: h2h[a][b] for b in models if b != a} for a in models}
    return {"game": game, "size": list(size), "models": models,
            "per_model": out, "h2h": h2h_out,
            "num_games": sum(len(g["episodes"]) for g in data["games"])}


# ---------------------------------------------------------------------------
def render_html(reports: dict) -> str:
    payload = json.dumps(reports)
    games = list(reports.keys())
    any_models = reports[games[0]]["models"]

    sections = ""
    for game in games:
        rep = reports[game]
        pm = rep["per_model"]
        ranked = sorted(rep["models"], key=lambda m: pm[m]["net_per_game"], reverse=True)
        rows = ""
        for i, m in enumerate(ranked, 1):
            s = pm[m]
            net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
            rows += f"""<tr>
              <td>{i}</td><td class='model'>{m}</td>
              <td class='{net_cls}'>{s['net_per_game']:+.2f}</td>
              <td>{s['win_rate']*100:.0f}%</td><td>{s['draw_rate']*100:.0f}%</td>
              <td>{s['first_move_win_rate']*100:.0f}%</td>
              <td>{s['win_take_rate']*100:.0f}%<div class='small'>n={s['win_opps']}</div></td>
              <td>{s['block_rate']*100:.0f}%<div class='small'>n={s['block_opps']}</div></td>
              <td>{s['missed_wins']}/{s['allowed_losses']}</td>
              <td>{s['invalid_rate']*100:.1f}%</td>
              <td>{s['avg_len']:.1f}</td><td>{s['avg_latency_s']:.1f}s</td>
            </tr>"""
        # head-to-head (wins-losses, row vs col)
        hh = "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in rep["models"]) + "</tr>"
        for a in rep["models"]:
            hh += f"<tr><th class='model'>{a}</th>"
            for b in rep["models"]:
                if a == b:
                    hh += "<td class='diag'>—</td>"
                else:
                    w, l, d = rep["h2h"][a][b]
                    cls = "pos" if w > l else ("neg" if l > w else "")
                    hh += f"<td class='{cls}'>{w}-{l}<div class='small'>{d}d</div></td>"
            hh += "</tr>"
        sections += f"""
        <h2>{'🔴 Connect Four' if game=='connect4' else '⚫ Gomoku-Lite'} — {rep['num_games']} games</h2>
        <table>
          <tr><th>#</th><th class='model'>model</th><th>net/game</th><th>win%</th>
              <th>draw%</th><th>1st-move win%</th><th>win-take</th><th>block</th>
              <th>miss/allow</th><th>invalid%</th><th>plies</th><th>think</th></tr>
          {rows}
        </table>
        <div class="note">win-take = took an immediate win when one existed; block = removed an
          opponent's immediate winning threat; miss/allow = missed wins / allowed losses (blunders).
          These are objective tactical-accuracy measures from the board.</div>
        <div class="grid2">
          <div><h3>Tactical accuracy</h3><canvas id="tac_{game}"></canvas></div>
          <div><h3>Win / draw / loss</h3><canvas id="wdl_{game}"></canvas></div>
        </div>
        <h3>Head-to-head (row wins–losses vs column)</h3>
        <table class='h2h'>{hh}</table>
        """

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Board-Game Tournament</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 22px 80px; }}
  h1 {{ font-size:25px; }} h2 {{ font-size:19px; margin-top:40px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }}
  h3 {{ font-size:14px; color:#9aa3b5; }}
  .sub {{ color:#8b93a7; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }}
  th {{ color:#9aa3b5; }} td.model,th.model {{ text-align:left; font-weight:600; color:#cdd6f4; }}
  .pos {{ color:#4ade80; }} .neg {{ color:#f87171; }} .diag {{ color:#3a3f4b; }}
  .small {{ font-size:10px; color:#8b93a7; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:10px; }}
  .note {{ color:#8b93a7; font-size:12px; margin:6px 0; }}
  canvas {{ max-height:300px; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🎲 AI Battle Arena — Board-Game Tournament</h1>
  <div class="sub">Perfect-information games · round-robin · tactical-accuracy analysis</div>
  {sections}
<script>
const R = {payload};
const COLORS=['#60a5fa','#f472b6','#4ade80','#fbbf24','#a78bfa'];
Chart.defaults.color='#9aa3b5'; Chart.defaults.borderColor='#232838';
for (const game of Object.keys(R)) {{
  const rep=R[game], pm=rep.per_model, M=rep.models;
  new Chart(document.getElementById('tac_'+game), {{ type:'bar',
    data:{{ labels:M, datasets:[
      {{label:'win-take %', backgroundColor:'#4ade80', data:M.map(m=>pm[m].win_take_rate*100)}},
      {{label:'block %', backgroundColor:'#60a5fa', data:M.map(m=>pm[m].block_rate*100)}},
    ]}},
    options:{{ scales:{{y:{{min:0,max:100}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
  new Chart(document.getElementById('wdl_'+game), {{ type:'bar',
    data:{{ labels:M, datasets:[
      {{label:'win', backgroundColor:'#4ade80', data:M.map(m=>pm[m].wins)}},
      {{label:'draw', backgroundColor:'#94a3b8', data:M.map(m=>pm[m].draws)}},
      {{label:'loss', backgroundColor:'#f87171', data:M.map(m=>pm[m].losses)}},
    ]}},
    options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
}}
</script>
</div></body></html>"""


def main():
    reports = {}
    for game in GAMES:
        path = os.path.join(DATA_DIR, f"{game}_data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        data = json.load(open(path))
        if not data.get("games"):
            print(f"skip {game}: no completed games yet")
            continue
        reports[game] = analyze_game(game, data)

    if not reports:
        print("No board-tournament data to analyze yet.")
        return

    html = render_html(reports)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    os.makedirs(REPORT_DIR, exist_ok=True)
    repo_html = os.path.join(REPORT_DIR, "board_tournament_report.html")
    with open(repo_html, "w", encoding="utf-8") as f:
        f.write(html)
    json.dump(reports, open(os.path.join(REPORT_DIR, "board_tournament_analysis.json"), "w"),
              indent=2)
    print(f"Wrote {OUT} and {repo_html}")
    for game, rep in reports.items():
        pm = rep["per_model"]
        print(f"\n=== {game} ({rep['num_games']} games) ===")
        for m in sorted(pm, key=lambda x: pm[x]["net_per_game"], reverse=True):
            s = pm[m]
            print(f"  {m:<16} net/g={s['net_per_game']:+.2f} win%={s['win_rate']*100:3.0f} "
                  f"win-take={s['win_take_rate']*100:3.0f}% block={s['block_rate']*100:3.0f}% "
                  f"invalid={s['invalid_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
