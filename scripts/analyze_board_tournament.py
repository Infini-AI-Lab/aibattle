"""Analyze the board-game tournament (Connect Four + Gomoku) and emit one
interactive HTML report PER GAME — the board-game counterpart of
analyze_tournament.py.

Poker's VPIP/aggression don't apply; the analogous *skill* signal here is
TACTICAL ACCURACY, computed directly from the board at every decision:

  - win-take rate : when an immediate winning move existed, did the model play one?
  - block rate    : when the opponent had an immediate winning threat (and the
                    model could not just win), did the model's move remove it?
  - blunders      : missed wins + allowed losses (failed blocks)

Plus results (win/draw/loss, net result/game, first-mover win rate), invalid-move
rate, average game length, latency, a head-to-head win matrix, and:

  - Bradley-Terry / Elo ratings fit from all head-to-head results
  - blunder timing broken into early / mid / late game phases
  - first-mover advantage (overall first-player win rate) + a per-model
    game-length distribution
  - a rendered move-location heatmap per model (center-control / opening bias)

Reads runs/board_tournament/<game>_data.json; writes <game>_report.html to
runs/board_tournament/ and a tracked copy under reports/.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from aibattle.games.board import connects, with_cell
from aibattle.games.gomoku import coord_to_rc

GAMES = ["connect4", "gomoku"]
NEED = {"connect4": 4, "gomoku": 5}
DATA_DIR = "runs/board_tournament"
REPORT_DIR = "reports"
PLAYERS = ["player_0", "player_1"]
PHASES = ["early", "mid", "late"]
TITLE = {"connect4": "🔴 Connect Four", "gomoku": "⚫ Gomoku-Lite"}
FAVICON = {"connect4": "🔴", "gomoku": "⚫"}


def _favicon(emoji: str) -> str:
    """An inline emoji favicon — no asset file, renders in the browser tab."""
    return (f"<link rel=\"icon\" href=\"data:image/svg+xml,"
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            f"<text y='.9em' font-size='90'>{emoji}</text></svg>\">")


# Shared top navigation across every report page. All targets are sibling files
# in reports/, so relative hrefs resolve. (key) marks the active tab.
NAV_ITEMS = [
    ("index.html", "Overview", "overview"),
    ("connect4_report.html", "🔴 Connect Four", "connect4"),
    ("gomoku_report.html", "⚫ Gomoku-Lite", "gomoku"),
    ("holdem_tournament_report.html", "🃏 Hold'em 1-Hand", "holdem"),
    ("match_tournament_report.html", "🃏 Hold'em Match", "match"),
    ("table_tournament_report.html", "🃏 Hold'em Table", "table"),
    ("kuhn_tournament_report.html", "🃏 Kuhn", "kuhn"),
]

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
"""


def _navbar(active: str) -> str:
    links = "".join(
        f"<a class='nav{' active' if key == active else ''}' href='{href}'>{label}</a>"
        for href, label, key in NAV_ITEMS)
    return ("<nav class='navbar'>"
            "<a class='brand' href='index.html'>🎲 AI Battle Arena</a>"
            f"{links}</nav>")


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


def _phase(step_idx, length):
    frac = step_idx / max(length, 1)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def _blank(size):
    return {
        "games": 0, "wins": 0, "losses": 0, "draws": 0, "net": 0.0,
        "decisions": 0, "invalid": 0,
        "first_moves": 0, "first_move_wins": 0,
        "win_opps": 0, "win_takes": 0,           # had an immediate win / took it
        "block_opps": 0, "blocks": 0,            # forced to defend / defended
        "lengths": [], "latencies": [],
        "heat": [[0] * size[1] for _ in range(size[0])],  # move-location counts
        # tactical opportunities split by game phase
        "phase": {ph: {"win_opps": 0, "win_takes": 0,
                       "block_opps": 0, "blocks": 0} for ph in PHASES},
    }


# ---------------------------------------------------------------------------
def bradley_terry(models, h2h, iters=300):
    """Fit Bradley-Terry strengths from pairwise results; return (strength, elo).

    Draws count as half a win to each side. Elo is the BT log-strength on the
    400/decade scale, recentred so the field averages 1500.
    """
    W = {m: 0.0 for m in models}
    N = defaultdict(lambda: defaultdict(float))
    for a in models:
        for b in models:
            if a == b:
                continue
            w, l, d = h2h[a][b]
            W[a] += w + 0.5 * d
            N[a][b] += w + l + d

    p = {m: 1.0 for m in models}
    for _ in range(iters):
        newp = {}
        for i in models:
            denom = sum(N[i][j] / (p[i] + p[j])
                        for j in models if j != i and N[i][j])
            newp[i] = (W[i] / denom) if denom > 0 else p[i]
        gm = math.exp(sum(math.log(max(v, 1e-9)) for v in newp.values()) / len(newp))
        p = {i: newp[i] / gm for i in models}

    raw = {m: 400 * math.log10(max(p[m], 1e-9)) for m in models}
    mean = sum(raw.values()) / len(raw)
    elo = {m: int(round(1500 + raw[m] - mean)) for m in models}
    return p, elo


def _histogram(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            hi_ok = v < edges[i + 1] or (i == len(edges) - 2 and v <= edges[i + 1])
            if edges[i] <= v and hi_ok:
                counts[i] += 1
                break
    return counts


def analyze_game(game: str, data: dict) -> dict:
    models = data["models"]
    sample = data["games"][0]["episodes"][0]["steps"][0]["observation"]["public"]["board"]
    size = (len(sample), len(sample[0]))
    stats = {m: _blank(size) for m in models}
    h2h = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [w,l,d] for a vs b
    fp_games = 0
    fp_wins = 0

    for g in data["games"]:
        a, b = g["a"], g["b"]
        for e in g["episodes"]:
            seat_name = e["seat_assignment"]
            winner_name = e.get("winner_name")
            steps = e["steps"]
            length = e["length"]

            for seat, nm in seat_name.items():
                st = stats[nm]
                st["games"] += 1
                st["lengths"].append(length)
                pay = e["returns"][seat]
                st["net"] += pay
                if pay > 0:
                    st["wins"] += 1
                elif pay < 0:
                    st["losses"] += 1
                else:
                    st["draws"] += 1

            if winner_name == a:
                h2h[a][b][0] += 1; h2h[b][a][1] += 1
            elif winner_name == b:
                h2h[b][a][0] += 1; h2h[a][b][1] += 1
            else:
                h2h[a][b][2] += 1; h2h[b][a][2] += 1

            if steps:
                fm = steps[0]["agent_name"]
                stats[fm]["first_moves"] += 1
                fp_games += 1
                if winner_name == fm:
                    stats[fm]["first_move_wins"] += 1
                    fp_wins += 1

            for idx, s in enumerate(steps):
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

                ph = _phase(idx, length)
                me, opp = seat, _other(seat)
                had_win = _has_immediate_win(game, grid, me)
                if had_win:
                    st["win_opps"] += 1
                    st["phase"][ph]["win_opps"] += 1
                    if _wins_at(game, grid, move, me):
                        st["win_takes"] += 1
                        st["phase"][ph]["win_takes"] += 1

                opp_threat = _has_immediate_win(game, grid, opp)
                if opp_threat and not had_win:
                    st["block_opps"] += 1
                    st["phase"][ph]["block_opps"] += 1
                    if rc is not None:
                        ng = with_cell(grid, rc[0], rc[1], me)
                        if not _has_immediate_win(game, ng, opp):
                            st["blocks"] += 1
                            st["phase"][ph]["blocks"] += 1

    # length histogram (shared bins across models)
    all_lengths = [ln for m in models for ln in stats[m]["lengths"]]
    lo, hi = min(all_lengths), max(all_lengths)
    nbins = min(12, max(hi - lo + 1, 1))
    step = max((hi - lo) / nbins, 1)
    edges = [lo + i * step for i in range(nbins + 1)]
    edges[-1] = hi  # close the last bin on the max
    bin_labels = [f"{int(round(edges[i]))}–{int(round(edges[i + 1]))}"
                  for i in range(len(edges) - 1)]

    out = {}
    for m in models:
        st = stats[m]
        n = max(st["games"], 1)
        phase_blunder = {}
        phase_opps = {}
        for ph in PHASES:
            p = st["phase"][ph]
            opps = p["win_opps"] + p["block_opps"]
            blunders = (p["win_opps"] - p["win_takes"]) + (p["block_opps"] - p["blocks"])
            phase_opps[ph] = opps
            phase_blunder[ph] = round(blunders / opps, 4) if opps else 0.0
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
            "phase_blunder_rate": phase_blunder,
            "phase_opps": phase_opps,
            "len_hist": _histogram(st["lengths"], edges),
        }

    strength, elo = bradley_terry(models, h2h)
    h2h_out = {a: {b: h2h[a][b] for b in models if b != a} for a in models}
    return {
        "game": game, "size": list(size), "models": models,
        "per_model": out, "h2h": h2h_out, "elo": elo,
        "len_bins": bin_labels,
        "first_player_win_rate": round(fp_wins / max(fp_games, 1), 4),
        "num_games": sum(len(g["episodes"]) for g in data["games"]),
    }


# ---------------------------------------------------------------------------
def _heat_html(heat):
    mx = max((c for row in heat for c in row), default=0) or 1
    cols = len(heat[0])
    out = [f"<div class='board' style='grid-template-columns:repeat({cols},14px)'>"]
    for row in heat:
        for c in row:
            a = c / mx
            out.append(f"<div class='cell' style='background:rgba(96,165,250,{a:.3f})'"
                       f" title='{c}'></div>")
    out.append("</div>")
    return "".join(out)


def render_game(game: str, rep: dict) -> str:
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    ranked = sorted(models, key=lambda m: rep["elo"][m], reverse=True)

    # results / tactics table
    rows = ""
    for i, m in enumerate(sorted(models, key=lambda x: pm[x]["net_per_game"],
                                 reverse=True), 1):
        s = pm[m]
        net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{m}</td>
          <td>{rep['elo'][m]}</td>
          <td class='{net_cls}'>{s['net_per_game']:+.2f}</td>
          <td>{s['win_rate']*100:.0f}%</td><td>{s['draw_rate']*100:.0f}%</td>
          <td>{s['first_move_win_rate']*100:.0f}%</td>
          <td>{s['win_take_rate']*100:.0f}%<div class='small'>n={s['win_opps']}</div></td>
          <td>{s['block_rate']*100:.0f}%<div class='small'>n={s['block_opps']}</div></td>
          <td>{s['missed_wins']}/{s['allowed_losses']}</td>
          <td>{s['invalid_rate']*100:.1f}%</td>
          <td>{s['avg_len']:.1f}</td><td>{s['avg_latency_s']:.1f}s</td>
        </tr>"""

    # head-to-head
    hh = "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in models) + "</tr>"
    for a in models:
        hh += f"<tr><th class='model'>{a}</th>"
        for b in models:
            if a == b:
                hh += "<td class='diag'>—</td>"
            else:
                w, l, d = rep["h2h"][a][b]
                cls = "pos" if w > l else ("neg" if l > w else "")
                hh += f"<td class='{cls}'>{w}-{l}<div class='small'>{d}d</div></td>"
        hh += "</tr>"

    # heatmaps
    heat_cards = ""
    for m in models:
        heat_cards += (f"<div class='heatcard'><div class='hlabel'>{m}</div>"
                       f"{_heat_html(pm[m]['heat'])}</div>")

    fpw = rep["first_player_win_rate"] * 100
    # Per-move replay viewer exists for both board games.
    replay_btn = (
        f"<a class='replaybtn' href='{game}_replay.html'>▶ Watch game replays</a>"
        if game in ("connect4", "gomoku") else "")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {TITLE[game]}</title>
{_favicon(FAVICON[game])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  {NAV_CSS}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 22px 80px; }}
  h1 {{ font-size:25px; }} h2 {{ font-size:18px; margin-top:38px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }}
  h3 {{ font-size:14px; color:#9aa3b5; }}
  .sub {{ color:#8b93a7; }}
  .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }}
  .kpi {{ background:#171a23; border:1px solid #232838; border-radius:10px; padding:12px 16px; }}
  .kpi .v {{ font-size:22px; font-weight:700; color:#cdd6f4; }}
  .kpi .l {{ font-size:11px; color:#8b93a7; text-transform:uppercase; letter-spacing:.04em; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }}
  th {{ color:#9aa3b5; }} td.model,th.model {{ text-align:left; font-weight:600; color:#cdd6f4; }}
  .pos {{ color:#4ade80; }} .neg {{ color:#f87171; }} .diag {{ color:#3a3f4b; }}
  .small {{ font-size:10px; color:#8b93a7; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:10px; }}
  .note {{ color:#8b93a7; font-size:12px; margin:6px 0; }}
  canvas {{ max-height:300px; }}
  .heatwrap {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; }}
  .heatcard {{ text-align:center; }}
  .hlabel {{ font-size:11px; color:#9aa3b5; margin-bottom:6px; }}
  .board {{ display:grid; gap:1px; background:#232838; padding:1px; border-radius:4px; }}
  .cell {{ width:14px; height:14px; }}
  .replaybtn {{ display:inline-block; margin-top:12px; background:#1b2030; color:#a5b4fc;
    border:1px solid #2a2f3a; border-radius:8px; padding:8px 14px; font-size:13px;
    text-decoration:none; }}
  .replaybtn:hover {{ border-color:#60a5fa; color:#fff; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style></head>
<body>{_navbar(game)}<div class="wrap">
  <h1>🎲 AI Battle Arena — {TITLE[game]}</h1>
  <div class="sub">Perfect-information game · round-robin · {rep['num_games']} games · board {rep['size'][0]}×{rep['size'][1]}</div>
  {replay_btn}

  <div class="kpis">
    <div class="kpi"><div class="v">{rep['elo'][ranked[0]]}</div><div class="l">top Elo · {ranked[0]}</div></div>
    <div class="kpi"><div class="v">{fpw:.0f}%</div><div class="l">first-mover win rate</div></div>
    <div class="kpi"><div class="v">{rep['num_games']}</div><div class="l">games played</div></div>
  </div>

  <h2>Leaderboard &amp; tactical accuracy</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>Elo</th><th>net/game</th><th>win%</th>
        <th>draw%</th><th>1st-move win%</th><th>win-take</th><th>block</th>
        <th>miss/allow</th><th>invalid%</th><th>plies</th><th>think</th></tr>
    {rows}
  </table>
  <div class="note">Elo from a Bradley-Terry fit over all head-to-head results (field mean 1500).
    win-take = took an immediate win when one existed; block = removed an opponent's immediate
    winning threat; miss/allow = missed wins / allowed losses (blunders). These are objective
    tactical-accuracy measures read from the board.</div>

  <div class="grid2">
    <div><h3>Elo rating</h3><canvas id="elo"></canvas></div>
    <div><h3>Win / draw / loss</h3><canvas id="wdl"></canvas></div>
  </div>
  <div class="grid2">
    <div><h3>Tactical accuracy (win-take / block %)</h3><canvas id="tac"></canvas></div>
    <div><h3>Blunder rate by game phase</h3><canvas id="phase"></canvas></div>
  </div>
  <div class="grid2">
    <div><h3>Game-length distribution (plies)</h3><canvas id="len"></canvas></div>
    <div></div>
  </div>

  <h2>Head-to-head (row wins–losses vs column)</h2>
  <table class='h2h'>{hh}</table>

  <h2>Move-location heatmap</h2>
  <div class="note">Where each model places pieces (brighter = more frequent). Reveals
    center-control bias and opening preferences.</div>
  <div class="heatwrap">{heat_cards}</div>

<script>
const R = {payload};
const pm=R.per_model, M=R.models, COLORS=['#60a5fa','#f472b6','#4ade80','#fbbf24','#a78bfa'];
Chart.defaults.color='#9aa3b5'; Chart.defaults.borderColor='#232838';
const ranked=[...M].sort((a,b)=>R.elo[b]-R.elo[a]);

new Chart(document.getElementById('elo'), {{ type:'bar',
  data:{{ labels:ranked, datasets:[{{label:'Elo', backgroundColor:'#a78bfa',
    data:ranked.map(m=>R.elo[m])}}]}},
  options:{{ indexAxis:'y', scales:{{x:{{min:Math.min(...Object.values(R.elo))-40}}}},
    plugins:{{legend:{{display:false}}}} }} }});

new Chart(document.getElementById('wdl'), {{ type:'bar',
  data:{{ labels:M, datasets:[
    {{label:'win', backgroundColor:'#4ade80', data:M.map(m=>pm[m].wins)}},
    {{label:'draw', backgroundColor:'#94a3b8', data:M.map(m=>pm[m].draws)}},
    {{label:'loss', backgroundColor:'#f87171', data:M.map(m=>pm[m].losses)}},
  ]}},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});

new Chart(document.getElementById('tac'), {{ type:'bar',
  data:{{ labels:M, datasets:[
    {{label:'win-take %', backgroundColor:'#4ade80', data:M.map(m=>pm[m].win_take_rate*100)}},
    {{label:'block %', backgroundColor:'#60a5fa', data:M.map(m=>pm[m].block_rate*100)}},
  ]}},
  options:{{ scales:{{y:{{min:0,max:100}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});

const PH=['early','mid','late'];
new Chart(document.getElementById('phase'), {{ type:'line',
  data:{{ labels:PH, datasets:M.map((m,i)=>({{label:m, borderColor:COLORS[i%COLORS.length],
    backgroundColor:COLORS[i%COLORS.length], tension:.2,
    data:PH.map(ph=>pm[m].phase_blunder_rate[ph]*100)}}))}},
  options:{{ scales:{{y:{{min:0, title:{{display:true,text:'blunder %'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});

new Chart(document.getElementById('len'), {{ type:'bar',
  data:{{ labels:R.len_bins, datasets:M.map((m,i)=>({{label:m,
    backgroundColor:COLORS[i%COLORS.length], data:pm[m].len_hist}}))}},
  options:{{ scales:{{y:{{title:{{display:true,text:'games'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});
</script>
</div></body></html>"""


def _index_card(href, title, meta, champ_line):
    return f"""
        <a class="card" href="{href}">
          <div class="ctitle">{title}</div>
          <div class="cmeta">{meta}</div>
          <div class="champ">{champ_line}</div>
          <div class="cgo">View analysis →</div>
        </a>"""


def render_index(reps: dict) -> str:
    """Unified landing page over the board games plus the Hold'em tournament.

    Written to reports/, where every <name>_report.html lives, so the relative
    links resolve. Hold'em stats are read from reports/holdem_tournament_analysis.json
    if present (it has no Elo, so we rank by bb/100).
    """
    cards = ""
    for game in GAMES:
        rep = reps.get(game)
        if not rep:
            continue
        champ = max(rep["models"], key=lambda m: rep["elo"][m])
        cards += _index_card(
            f"{game}_report.html", TITLE[game],
            f"{rep['num_games']} games · board {rep['size'][0]}×{rep['size'][1]}"
            f" · first-mover {rep['first_player_win_rate']*100:.0f}%",
            f"🏆 {champ} <span class='metric'>Elo {rep['elo'][champ]}</span>")

    # Three Hold'em formats, distinct enough to stand alone: 1-Hand (each hand
    # scored independently, bb/100), Match (heads-up, stacks carried, win the
    # match), and Table (5-handed ring, scored by finishing rank).
    holdem_path = os.path.join(REPORT_DIR, "holdem_tournament_analysis.json")
    if os.path.exists(holdem_path):
        h = json.load(open(holdem_path))
        pm = h["per_model"]
        champ = max(h["models"], key=lambda m: pm[m]["bb_per_100"])
        cards += _index_card(
            "holdem_tournament_report.html", "🃏 Hold'em 1-Hand",
            f"heads-up · {h['num_games']} tables · {h['hands_per_game']} hands each"
            f" · per-hand bb/100",
            f"🏆 {champ} <span class='metric'>{pm[champ]['bb_per_100']:+.1f} bb/100</span>")

    match_path = os.path.join(REPORT_DIR, "match_tournament_analysis.json")
    if os.path.exists(match_path):
        m = json.load(open(match_path))
        champ = m["leaderboard"][0]
        cards += _index_card(
            "match_tournament_report.html", "🃏 Hold'em Match",
            f"heads-up · {m['episodes_per_pair']} matches/pair · up to "
            f"{m['max_hands']} hands · stacks carried",
            f"🏆 {champ['model']} <span class='metric'>{champ['win_rate']*100:.0f}% match wins</span>")

    table_path = os.path.join(REPORT_DIR, "table_tournament_analysis.json")
    if os.path.exists(table_path):
        t = json.load(open(table_path))
        champ = max(t["leaderboard"], key=lambda r: (r["top1_rate"], -r["avg_rank"]))
        cards += _index_card(
            "table_tournament_report.html", "🃏 Hold'em Table",
            f"{t['num_players']}-handed · {t['sessions']} sessions · up to "
            f"{t['max_hands']} hands · top-1 rate",
            f"🏆 {champ['model']} <span class='metric'>{champ['top1_rate']*100:.0f}% top-1</span>")

    # Kuhn is a solved game scored against GTO; the leaderboard is pre-ranked, so
    # the champion is simply the first entry. Net chips/hand is the headline.
    kuhn_path = os.path.join(REPORT_DIR, "kuhn_tournament_analysis.json")
    if os.path.exists(kuhn_path):
        k = json.load(open(kuhn_path))
        lb = k["leaderboard"]
        champ = lb[0]
        cards += _index_card(
            "kuhn_tournament_report.html", "🃏 Kuhn Poker",
            f"{k['episodes_per_pair']} hands/pair · solved game · GTO scoring",
            f"🏆 {champ['model']} <span class='metric'>{champ['net_per_hand']:+.3f} net/hand</span>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Tournaments</title>
{_favicon("🎲")}
<style>
  {NAV_CSS}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:40px 22px; }}
  h1 {{ font-size:28px; }} .sub {{ color:#8b93a7; margin-bottom:32px; }}
  .cards {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .card {{ display:block; text-decoration:none; color:inherit; background:#171a23;
    border:1px solid #232838; border-radius:14px; padding:24px; transition:.15s; }}
  .card:hover {{ border-color:#60a5fa; transform:translateY(-2px); }}
  .ctitle {{ font-size:21px; font-weight:700; color:#cdd6f4; }}
  .cmeta {{ font-size:12px; color:#8b93a7; margin:8px 0 16px; }}
  .champ {{ font-size:14px; color:#e6e6e6; }} .metric {{ color:#a78bfa; font-weight:600; }}
  .cgo {{ margin-top:16px; font-size:13px; color:#60a5fa; }}
  @media (max-width:640px) {{ .cards {{ grid-template-columns:1fr; }} }}
</style></head>
<body>{_navbar("overview")}<div class="wrap">
  <h1>🎲 AI Battle Arena — Tournaments</h1>
  <div class="sub">LLMs playing perfect-information board games &amp; Texas Hold'em · round-robin · skill analysis</div>
  <div class="cards">{cards}</div>
</div></body></html>"""


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    reps = {}
    for game in GAMES:
        path = os.path.join(DATA_DIR, f"{game}_data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        data = json.load(open(path))
        if not data.get("games"):
            print(f"skip {game}: no completed games yet")
            continue

        rep = analyze_game(game, data)
        reps[game] = rep
        html = render_game(game, rep)

        out = os.path.join(DATA_DIR, f"{game}_report.html")
        repo_html = os.path.join(REPORT_DIR, f"{game}_report.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        with open(repo_html, "w", encoding="utf-8") as f:
            f.write(html)
        json.dump(rep, open(os.path.join(REPORT_DIR, f"{game}_analysis.json"), "w"),
                  indent=2)
        print(f"Wrote {out} and {repo_html}")

        pm = rep["per_model"]
        print(f"=== {game} ({rep['num_games']} games · "
              f"first-mover win {rep['first_player_win_rate']*100:.0f}%) ===")
        for m in sorted(pm, key=lambda x: rep["elo"][x], reverse=True):
            s = pm[m]
            print(f"  {m:<16} elo={rep['elo'][m]:>4} net/g={s['net_per_game']:+.2f} "
                  f"win%={s['win_rate']*100:3.0f} win-take={s['win_take_rate']*100:3.0f}% "
                  f"block={s['block_rate']*100:3.0f}% invalid={s['invalid_rate']*100:.1f}%")

    if reps:
        # Unified index lives in reports/, where every *_report.html (board games
        # AND holdem) resides, so all relative links resolve.
        index = render_index(reps)
        with open(os.path.join(REPORT_DIR, "index.html"), "w", encoding="utf-8") as f:
            f.write(index)
        print(f"Wrote index.html to {REPORT_DIR}")


if __name__ == "__main__":
    main()
