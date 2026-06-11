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

Reads runs/<game>/<game>_data.json (per-game coached folders) for each of
connect4, gomoku; writes <game>_report.html to runs/<game>/ and a tracked copy
under reports/. Also regenerates the unified reports/index.html.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from aibattle.games.board import connects, with_cell
from aibattle.games.gomoku import coord_to_rc
from model_names import strip_coached

GAMES = ["connect4", "gomoku"]
NEED = {"connect4": 4, "gomoku": 5}
# Coached is now the canonical (and only) run set. connect4 and gomoku each live
# in their own per-game folder (runs/connect4, runs/gomoku) holding
# <game>_data.json + the per-pair match dirs.
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")
PLAYERS = ["player_0", "player_1"]
PHASES = ["early", "mid", "late"]
TITLE = {"connect4": "🔴 Connect Four", "gomoku": "⚫ Gomoku-Lite"}
FAVICON = {"connect4": "🔴", "gomoku": "⚫"}


def _favicon(emoji: str) -> str:
    """An inline emoji favicon — no asset file, renders in the browser tab."""
    return (f"<link rel=\"icon\" href=\"data:image/svg+xml,"
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            f"<text y='.9em' font-size='90'>{emoji}</text></svg>\">")


# The site navbar is a shared client-side component — see reports/nav.css and
# reports/nav.js. Every generated page includes those two files in <head> (via
# NAV_HEAD) and the bar is injected by JS, so the nav markup lives in one place.
NAV_HEAD = '<link rel="stylesheet" href="nav.css"><script defer src="nav.js"></script>'


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
    400/decade scale, recentred so the rated field averages 1500.

    A model with no wins/draws (or no losses/draws) has no finite BT rating —
    its strength diverges to ±infinity. Such degenerate records are excluded
    from the fit and reported with elo=None (rendered as "—"), so a model that
    e.g. went 0-40 doesn't drag its own and the field's ratings to nonsense.
    """
    wins = {m: 0.0 for m in models}
    losses = {m: 0.0 for m in models}
    draws = {m: 0.0 for m in models}
    for a in models:
        for b in models:
            if a == b:
                continue
            w, l, d = h2h[a][b]
            wins[a] += w
            losses[a] += l
            draws[a] += d

    # Rated = mixed record (at least one win-or-draw AND one loss-or-draw).
    rated = [m for m in models
             if (wins[m] + draws[m]) > 0 and (losses[m] + draws[m]) > 0]

    W = {m: 0.0 for m in rated}
    N = defaultdict(lambda: defaultdict(float))
    for a in rated:
        for b in rated:  # ignore games vs unrated models in the fit
            if a == b:
                continue
            w, l, d = h2h[a][b]
            W[a] += w + 0.5 * d
            N[a][b] += w + l + d

    p = {m: 1.0 for m in rated}
    for _ in range(iters):
        newp = {}
        for i in rated:
            denom = sum(N[i][j] / (p[i] + p[j])
                        for j in rated if j != i and N[i][j])
            newp[i] = (W[i] / denom) if denom > 0 else p[i]
        gm = math.exp(sum(math.log(max(v, 1e-9)) for v in newp.values())
                      / len(newp)) if newp else 1.0
        p = {i: newp[i] / gm for i in rated}

    elo = {m: None for m in models}
    if rated:
        raw = {m: 400 * math.log10(max(p[m], 1e-9)) for m in rated}
        mean = sum(raw.values()) / len(raw)
        for m in rated:
            elo[m] = int(round(1500 + raw[m] - mean))
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


def _elo_key(elo, m):
    """Sort key putting unrated (elo=None) models last in a descending sort."""
    return elo[m] if elo[m] is not None else float("-inf")


def _elo_txt(e):
    return "—" if e is None else str(e)


def render_game(game: str, rep: dict) -> str:
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    elo = rep["elo"]
    ranked = sorted(models, key=lambda m: _elo_key(elo, m), reverse=True)

    # results / tactics table
    rows = ""
    for i, m in enumerate(sorted(models, key=lambda x: pm[x]["net_per_game"],
                                 reverse=True), 1):
        s = pm[m]
        net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{m}</td>
          <td>{_elo_txt(rep['elo'][m])}</td>
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
    # Per-move replay viewer exists for both board games. The coached variant
    # has no replay viewers built (they fetch run data at runtime), so omit it.
    # Coached runs have no replay viewer built; omit the button so it never 404s.
    replay_btn = ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {TITLE[game]}</title>
{_favicon(FAVICON[game])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:28px 28px 80px; }}
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
<body><div class="wrap">
  <h1>🎲 AI Battle Arena — {TITLE[game]}</h1>
  <div class="sub">Perfect-information game · round-robin · {rep['num_games']} games · board {rep['size'][0]}×{rep['size'][1]}</div>
  {replay_btn}

  <div class="kpis">
    <div class="kpi"><div class="v">{_elo_txt(rep['elo'][ranked[0]])}</div><div class="l">top Elo · {ranked[0]}</div></div>
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
  <div class="note">Elo from a Bradley-Terry fit over head-to-head results (rated field mean 1500).
    A model with no wins or no losses has no finite rating and is shown as “—” (excluded from the fit).
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
const eloRanked=ranked.filter(m=>R.elo[m]!=null);  // unrated (—) models omitted from the chart

new Chart(document.getElementById('elo'), {{ type:'bar',
  data:{{ labels:eloRanked, datasets:[{{label:'Elo', backgroundColor:'#a78bfa',
    data:eloRanked.map(m=>R.elo[m])}}]}},
  options:{{ indexAxis:'y', scales:{{x:{{min:Math.min(...eloRanked.map(m=>R.elo[m]))-40}}}},
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


# Per-game taxonomy: which arena section, the information class (perfect vs
# imperfect), and the descriptor badges shown on each card. The "group" key
# drives the perfect/imperfect sub-grouping inside the Model Arena.
GAME_TAXONOMY = {
    "connect4":     {"group": "perfect",   "badges": ["Perfect info", "2P", "Deterministic"]},
    "gomoku":       {"group": "perfect",   "badges": ["Perfect info", "2P", "Deterministic"]},
    "holdem":       {"group": "imperfect", "badges": ["Imperfect info", "Heads-up", "Stochastic"]},
    "match":        {"group": "imperfect", "badges": ["Imperfect info", "Heads-up", "Stochastic"]},
    "table":        {"group": "imperfect", "badges": ["Imperfect info", "5-handed", "Stochastic"]},
    "kuhn":         {"group": "imperfect", "badges": ["Imperfect info", "Heads-up", "Stochastic"]},
}


def _index_card(entry: dict) -> str:
    badges = "".join(f"<span class='badge'>{b}</span>" for b in entry["badges"])
    return f"""
        <a class="card" href="{entry['href']}">
          <div class="ctitle">{entry['title']}</div>
          <div class="badges">{badges}</div>
          <div class="cmeta">{entry['meta']}</div>
          <div class="champ">{entry['champ_line']}</div>
          <div class="cgo">View analysis →</div>
        </a>"""


def _arena_scores(entries: list) -> list:
    """Cross-game normalized model ranking.

    Each game contributes a per-model score in [0,1] from its finishing order:
    best = 1.0, worst = 0.0, evenly spaced (``(N-1-rank)/(N-1)``). A model's
    Arena Score is the mean of its per-game scores ×100, so it does not reward
    breadth on its own — but we also surface coverage (games played) so a model
    that has only entered one game reads as provisional.

    Different games use different native metrics (Elo, bb/100, finishing rank),
    which cannot be added directly; normalizing to within-game rank is the
    apples-to-apples bridge.
    """
    agg = {}  # model -> [scores]
    for e in entries:
        ranking = e["ranking"]
        n = len(ranking)
        for rank, model in enumerate(ranking):
            score = 1.0 if n == 1 else (n - 1 - rank) / (n - 1)
            agg.setdefault(model, []).append((score, e["title"]))
    rows = []
    for model, scs in agg.items():
        vals = [s for s, _ in scs]
        best_title = max(scs, key=lambda x: x[0])[1]
        rows.append({
            "model": model,
            "score": round(100 * sum(vals) / len(vals), 1),
            "games": len(vals),
            "best": best_title,
        })
    rows.sort(key=lambda r: (-r["score"], -r["games"]))
    return rows


def _arena_board(entries: list) -> str:
    rows = _arena_scores(entries)
    if not rows:
        return ""
    total = len(entries)
    body = ""
    for i, r in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}")
        body += (
            f"<tr><td class='rk'>{medal}</td>"
            f"<td class='model'>{r['model']}</td>"
            f"<td class='scorecell'><span class='bar' style='width:{r['score']}%'></span>"
            f"<span class='sval'>{r['score']:.0f}</span></td>"
            f"<td class='cov'>{r['games']}/{total}</td>"
            f"<td class='best'>{r['best']}</td></tr>")
    return f"""
  <section class="board">
    <div class="arena-head"><h2>🏅 Cross-game model leaderboard</h2>
      <span class="arena-tag">normalized rank · all model-arena games</span></div>
    <div class="note">Arena Score = mean within-game finishing position (best 100, worst 0)
      across the games a model has entered. Coverage shows games played; treat low coverage as provisional.</div>
    <table class="lb">
      <tr><th class='rk'>#</th><th class='model'>model</th><th>Arena Score</th>
        <th>coverage</th><th>best game</th></tr>
      {body}
    </table>
  </section>"""


def render_index(reps: dict) -> str:
    """Two-arena landing page: a fair Model Arena (one generic pipeline for every
    model, grouped by perfect vs imperfect information) plus an open Agentic
    Arena. A cross-game normalized leaderboard sits on top.

    Written to reports/, where every <name>_report.html lives, so the relative
    links resolve. Hold'em stats are read from reports/*_analysis.json.
    """
    entries = []  # one dict per game; powers both the cards and the leaderboard

    for game in GAMES:
        rep = reps.get(game)
        if not rep:
            continue
        ordered = sorted(rep["models"], key=lambda m: _elo_key(rep["elo"], m), reverse=True)
        champ = ordered[0]
        entries.append({
            "key": game, "title": TITLE[game], "href": f"{game}_report.html",
            "meta": (f"{rep['num_games']} games · board {rep['size'][0]}×{rep['size'][1]}"
                     f" · first-mover {rep['first_player_win_rate']*100:.0f}%"),
            "champ_line": f"🏆 {champ} <span class='metric'>Elo {_elo_txt(rep['elo'][champ])}</span>",
            "ranking": ordered, **GAME_TAXONOMY[game],
        })

    # Kuhn first inside the imperfect group: it is the simplest game (a solved,
    # tiny-state game scored against GTO), so it reads as the natural entry point.
    kuhn_path = os.path.join(REPORT_DIR, "kuhn_tournament_analysis.json")
    if os.path.exists(kuhn_path):
        k = json.load(open(kuhn_path))
        lb = k["leaderboard"]
        champ = lb[0]
        entries.append({
            "key": "kuhn", "title": "🃏 Kuhn Poker",
            "href": "kuhn_tournament_report.html",
            "meta": f"{k['episodes_per_pair']} hands/pair · solved game · GTO scoring",
            "champ_line": f"🏆 {champ['model']} <span class='metric'>{champ['net_per_hand']:+.3f} net/hand</span>",
            "ranking": [r["model"] for r in lb], **GAME_TAXONOMY["kuhn"],
        })

    # Three Hold'em formats, distinct enough to stand alone: 1-Hand (each hand
    # scored independently, bb/100), Match (heads-up, stacks carried, win the
    # match), and Table (5-handed ring, scored by finishing rank).
    holdem_path = os.path.join(REPORT_DIR, "holdem_tournament_analysis.json")
    if os.path.exists(holdem_path):
        h = json.load(open(holdem_path))
        pm = h["per_model"]
        ordered = sorted(h["models"], key=lambda m: pm[m]["bb_per_100"], reverse=True)
        champ = ordered[0]
        entries.append({
            "key": "holdem", "title": "🃏 Hold'em 1-Hand",
            "href": "holdem_tournament_report.html",
            "meta": (f"heads-up · {h['num_games']} tables · {h['hands_per_game']} hands each"
                     f" · per-hand bb/100"),
            "champ_line": f"🏆 {champ} <span class='metric'>{pm[champ]['bb_per_100']:+.1f} bb/100</span>",
            "ranking": ordered, **GAME_TAXONOMY["holdem"],
        })

    match_path = os.path.join(REPORT_DIR, "match_tournament_analysis.json")
    if os.path.exists(match_path):
        m = json.load(open(match_path))
        lb = m["leaderboard"]
        champ = lb[0]
        entries.append({
            "key": "match", "title": "🃏 Hold'em Match",
            "href": "match_tournament_report.html",
            "meta": (f"heads-up · {m['episodes_per_pair']} matches/pair · up to "
                     f"{m['max_hands']} hands · stacks carried"),
            "champ_line": f"🏆 {champ['model']} <span class='metric'>{champ['win_rate']*100:.0f}% match wins</span>",
            "ranking": [r["model"] for r in lb], **GAME_TAXONOMY["match"],
        })

    table_path = os.path.join(REPORT_DIR, "table_tournament_analysis.json")
    if os.path.exists(table_path):
        t = json.load(open(table_path))
        lb = t["leaderboard"]  # pre-sorted by avg finishing rank (lower better)
        champ = lb[0]
        entries.append({
            "key": "table", "title": "🃏 Hold'em Table",
            "href": "table_tournament_report.html",
            "meta": (f"{t['num_players']}-handed · {t['sessions']} sessions · up to "
                     f"{t['max_hands']} hands · avg finishing rank"),
            "champ_line": f"🏆 {champ['model']} <span class='metric'>{champ['avg_rank']} avg rank</span>",
            "ranking": [r["model"] for r in lb], **GAME_TAXONOMY["table"],
        })

    # New-games tournament (independent_blackjack, leduc, blotto, othello). Its
    # analyzer (analyze_new_games.py) writes a compact entry list that already
    # carries title/href/group/badges/ranking/meta/champ_line, so each row drops
    # straight into both the cards and the cross-game Arena Score.
    newgames_path = os.path.join(REPORT_DIR, "new_games_index.json")
    if os.path.exists(newgames_path):
        entries.extend(json.load(open(newgames_path)))

    def _group(name):
        cards = "".join(_index_card(e) for e in entries if e["group"] == name)
        return f"<div class='cards'>{cards}</div>" if cards else \
            "<div class='empty'>No games yet.</div>"

    board = _arena_board(entries)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena</title>
{_favicon("🎲")}
{NAV_HEAD}
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:40px 28px; }}
  h1 {{ font-size:28px; margin-bottom:6px; }} .sub {{ color:#8b93a7; margin-bottom:28px; }}
  h2 {{ font-size:20px; color:#cdd6f4; margin:0; }}
  .note {{ color:#8b93a7; font-size:12px; margin:6px 0 14px; }}

  .arena {{ margin-top:36px; border:1px solid #232838; border-radius:16px;
    padding:22px 22px 26px; background:linear-gradient(180deg,#141823,#10131b);
    scroll-margin-top:64px; }}
  .arena-head {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px; }}
  .arena-tag {{ font-size:11px; color:#9aa3b5; background:#1c2130; border:1px solid #2a3142;
    padding:3px 9px; border-radius:999px; }}
  .group-label {{ display:flex; align-items:center; gap:10px; font-size:16px;
    font-weight:700; color:#e9edf7; margin:24px 0 13px; padding-left:12px;
    border-left:3px solid #3b82f6; }}
  .group-label.perfect {{ border-left-color:#4ade80; }}
  .group-label.imperfect {{ border-left-color:#f59e0b; }}
  .group-label .gl-sub {{ font-size:11px; font-weight:400; color:#8b93a7; }}

  .cards {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card {{ display:block; text-decoration:none; color:inherit; background:#171a23;
    border:1px solid #232838; border-radius:14px; padding:20px; transition:.15s; }}
  .card:hover {{ border-color:#60a5fa; transform:translateY(-2px); }}
  .ctitle {{ font-size:19px; font-weight:700; color:#cdd6f4; }}
  .badges {{ margin:8px 0; display:flex; gap:6px; flex-wrap:wrap; }}
  .badge {{ font-size:10px; color:#aab2c5; background:#1c2130; border:1px solid #2a3142;
    padding:2px 7px; border-radius:999px; }}
  .cmeta {{ font-size:12px; color:#8b93a7; margin:0 0 14px; }}
  .champ {{ font-size:14px; color:#e6e6e6; }} .metric {{ color:#a78bfa; font-weight:600; }}
  .cgo {{ margin-top:14px; font-size:13px; color:#60a5fa; }}

  table.lb {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.lb th, table.lb td {{ padding:7px 10px; border-bottom:1px solid #20242f; text-align:center; }}
  table.lb th {{ color:#9aa3b5; font-weight:600; }}
  table.lb td.model, table.lb th.model {{ text-align:left; font-weight:600; color:#cdd6f4; }}
  table.lb td.rk, table.lb th.rk {{ width:34px; color:#8b93a7; }}
  table.lb td.cov {{ color:#8b93a7; }} table.lb td.best {{ color:#aab2c5; }}
  .scorecell {{ position:relative; min-width:140px; }}
  .scorecell .bar {{ position:absolute; left:0; top:50%; transform:translateY(-50%);
    height:18px; border-radius:4px; background:linear-gradient(90deg,#3b82f6,#a78bfa); opacity:.5; }}
  .scorecell .sval {{ position:relative; font-weight:600; color:#f3f4f6; }}

  .cta {{ display:flex; align-items:center; justify-content:space-between; gap:16px;
    flex-wrap:wrap; border:1px dashed #34405c; border-radius:14px; padding:22px;
    background:#141a27; }}
  .cta .ctatext b {{ color:#cdd6f4; font-size:15px; }}
  .cta .ctatext div {{ color:#8b93a7; font-size:13px; margin-top:4px; }}
  .cta .pill {{ font-size:13px; color:#0b1020; background:#60a5fa; font-weight:600;
    padding:9px 16px; border-radius:999px; white-space:nowrap; }}
  .empty {{ color:#8b93a7; font-size:13px; padding:8px 0; }}
  @media (max-width:640px) {{ .cards {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🎲 AI Battle Arena</h1>
  <div class="sub">Two arenas, same games. <b>Model Arena</b> pits raw models through one
    identical pipeline; <b>Agentic Arena</b> is open to any model + any scaffolding.</div>

  <a href="gpt_vs_claude/index.html" style="display:flex;align-items:center;justify-content:space-between;
    gap:16px;flex-wrap:wrap;text-decoration:none;color:inherit;margin-top:8px;
    border:1px solid #2a3142;border-radius:14px;padding:18px 22px;
    background:linear-gradient(90deg,#10241d,#141823 55%,#1d1630);transition:.15s"
    onmouseover="this.style.borderColor='#60a5fa'" onmouseout="this.style.borderColor='#2a3142'">
    <div><b style="color:#cdd6f4;font-size:15px">🥊 GPT vs Claude — coached head-to-head</b>
      <div style="color:#8b93a7;font-size:13px;margin-top:4px">gpt-5.5 / gpt-5.4 vs
        claude-opus-4.8 / claude-sonnet-4.6 across four games · family scoreboard + per-game analysis</div></div>
    <span style="font-size:13px;color:#0b1020;background:#60a5fa;font-weight:600;
      padding:9px 16px;border-radius:999px;white-space:nowrap">View →</span>
  </a>
  {board}

  <section class="arena" id="model">
    <div class="arena-head"><h2>🤖 Model Arena</h2>
      <span class="arena-tag">fair · one identical generic pipeline</span></div>
    <div class="note">Every model plays through the same prompt/parse/retry wrapper — this
      measures the model, not the scaffolding.</div>
    <div class="group-label perfect">♟ Perfect information
      <span class="gl-sub">full state visible · deterministic</span></div>
    {_group("perfect")}
    <div class="group-label imperfect">🎭 Imperfect information
      <span class="gl-sub">hidden cards · stochastic</span></div>
    {_group("imperfect")}
  </section>

  <section class="arena" id="agentic">
    <div class="arena-head"><h2>🛠️ Agentic Arena</h2>
      <span class="arena-tag">open · any model · any pipeline</span></div>
    <div class="note">Bring your own scaffolding — tools, search, memory, self-play. Ranked
      on the same games; uplift over the underlying model is the headline metric.</div>
    <div class="cta">
      <div class="ctatext"><b>Open for submissions →</b>
        <div>No agents entered yet. Wire one up via the external-agent (HTTP) interface.</div></div>
      <span class="pill">Coming soon</span>
    </div>
  </section>
</div></body></html>"""


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    reps = {}
    for game in GAMES:
        data_dir = f"runs/{game}"
        path = os.path.join(data_dir, f"{game}_data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        data = strip_coached(json.load(open(path)))
        if not data.get("games"):
            print(f"skip {game}: no completed games yet")
            continue

        rep = analyze_game(game, data)
        reps[game] = rep
        html = render_game(game, rep)

        out = os.path.join(data_dir, f"{game}_report.html")
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
        for m in sorted(pm, key=lambda x: _elo_key(rep["elo"], x), reverse=True):
            s = pm[m]
            print(f"  {m:<16} elo={_elo_txt(rep['elo'][m]):>4} net/g={s['net_per_game']:+.2f} "
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
