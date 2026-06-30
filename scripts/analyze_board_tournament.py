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
from model_names import strip_coached, display_name, model_cell
from report_tokens import tokens_from_episodes, token_cost_cells, TOKEN_HEADERS
from elo_util import bootstrap_elo, wld_from_records
from report_theme import BASE_CSS, CHART_SETUP
from report_legends import legend as _legend

GAMES = ["connect4", "gomoku"]

# Hand-authored "Why win / why lose" analysis block (defense table, threat-axis
# miss-rate, ↘/↙ diagonals) folded into the board reports — originally added by
# hand (PR #7). Kept as an editable HTML file under
# scripts/board_analysis/<game>_why.html so re-running the pipeline reproduces it.
# The leaderboard/charts above are data-driven; this block is a pinned snapshot.
_ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_analysis")

NEED = {"connect4": 4, "gomoku": 5}
# Coached is now the canonical (and only) run set. connect4 and gomoku each live
# in their own per-game folder (runs/connect4, runs/gomoku) holding
# <game>_data.json + the per-pair match dirs.
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")
PLAYERS = ["player_0", "player_1"]
PHASES = ["early", "mid", "late"]
TITLE = {"connect4": "🔴 Connect Four", "gomoku": "⚫ Gomoku-Lite"}
FAVICON = {"connect4": "🔴", "gomoku": "⚫"}
# Short intro per game (shown under the header). A value that begins with "<"
# is emitted as-is (a structured .rules block, like the Kuhn page); a plain
# string is wrapped in a .callout.
INTRO = {
    "connect4": (
        '<input type="checkbox" class="rules-toggle" id="rules-toggle" hidden>'
        '<label class="rules-summary" for="rules-toggle">Setup &amp; rules<span class="rules-hint"> · expand</span></label>'
        '<div class="rules">'
        "<h3>How Connect Four works</h3>"
        "A two-player game on a vertical <b>7-column × 6-row</b> grid; the two models "
        "alternate dropping discs of their own color. One game plays out like this:"
        "<ul>"
        "<li>On your turn you pick a <b>column</b>; the disc falls to the "
        "<b>lowest empty cell</b> in it. A full column can't be chosen.</li>"
        "<li><b>Player 0 moves first</b> — a real edge in a solved game.</li>"
        "<li>The first to line up <b>four of their discs in a row</b> — horizontally, "
        "vertically, or diagonally — wins immediately.</li>"
        "<li>If the board fills with no four-in-a-row, the game is a <b>draw</b>.</li>"
        "</ul>"
        '<div class="seq">Connect Four is <b>solved</b>: with perfect play the first '
        "player always wins. There's no hidden information or chance, so every position "
        "has a known best move — which lets us score <b>tactical accuracy</b> directly: "
        "did the model take an available immediate win, and did it block the opponent's "
        "immediate winning threat?</div>"
        '<div class="seq"><b>What the model sees each turn:</b> the full current board, whose '
        "turn it is, and the legal (non-full) columns — perfect information.</div>"
        "</div>"),
    "gomoku": (
        '<input type="checkbox" class="rules-toggle" id="rules-toggle" hidden>'
        '<label class="rules-summary" for="rules-toggle">Setup &amp; rules<span class="rules-hint"> · expand</span></label>'
        '<div class="rules">'
        "<h3>How Gomoku-Lite works</h3>"
        "A two-player game on a <b>9×9</b> board (columns A–I, rows 1–9). The two models "
        "alternate placing one stone of their own color on any empty cell:"
        "<ul>"
        "<li>On your turn, name an empty cell — e.g. <code>E5</code> — and a stone is "
        "placed there.</li>"
        "<li><b>Player 0 moves first.</b></li>"
        "<li>The first to make <b>five of their stones in a row</b> — horizontally, "
        "vertically, or diagonally — wins immediately.</li>"
        "<li>If the board fills with no five-in-a-row, the game is a <b>draw</b> (rare "
        "on 9×9).</li>"
        "</ul>"
        '<div class="seq">Free-style Gomoku — no forbidden-move restrictions — and like '
        "Connect Four there's no hidden information or chance, so we score "
        "<b>tactical accuracy</b>: did the model complete an available five (win-take) "
        "and block the opponent's immediate five-threat (block rate)?</div>"
        '<div class="seq"><b>What the model sees each turn:</b> the full current board, whose '
        "turn it is, and that any empty cell is a legal move — perfect information.</div>"
        "</div>"),
}


def _intro_html(game: str) -> str:
    """Structured .rules block as-is, else wrap the plain blurb in a .callout."""
    body = INTRO[game]
    return body if body.lstrip().startswith("<") else f'<div class="callout">{body}</div>'


def _favicon(emoji: str) -> str:
    """An inline emoji favicon — no asset file, renders in the browser tab."""
    return (f"<link rel=\"icon\" href=\"data:image/svg+xml,"
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            f"<text y='.9em' font-size='90'>{emoji}</text></svg>\">")


# The site navbar is a shared client-side component — see reports/nav.css and
# reports/nav.js. Every generated page includes those two files in <head> (via
# NAV_HEAD) and the bar is injected by JS, so the nav markup lives in one place.
NAV_HEAD = '<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="nav.css?v=7"><script defer src="nav.js?v=32"></script>'


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
        "second_moves": 0, "second_move_wins": 0,
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
    records = []  # per-game (a, b, result) for the Elo bootstrap; +1 a wins, -1 b
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
                records.append((a, b, 1))
            elif winner_name == b:
                h2h[b][a][0] += 1; h2h[a][b][1] += 1
                records.append((a, b, -1))
            else:
                h2h[a][b][2] += 1; h2h[b][a][2] += 1
                records.append((a, b, 0))

            if steps:
                fm = steps[0]["agent_name"]
                stats[fm]["first_moves"] += 1
                fp_games += 1
                if winner_name == fm:
                    stats[fm]["first_move_wins"] += 1
                    fp_wins += 1
                # the other seat is the second mover
                sm = next((nm for nm in seat_name.values() if nm != fm), None)
                if sm is not None:
                    stats[sm]["second_moves"] += 1
                    if winner_name == sm:
                        stats[sm]["second_move_wins"] += 1

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
            "second_move_win_rate": round(
                st["second_move_wins"] / max(st["second_moves"], 1), 4),
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
    elo_ci = bootstrap_elo(models, records, lambda s: wld_from_records(models, s))
    h2h_out = {a: {b: h2h[a][b] for b in models if b != a} for a in models}
    return {
        "game": game, "size": list(size), "models": models,
        "per_model": out, "h2h": h2h_out, "elo": elo, "elo_ci": elo_ci,
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


def _elo_cell(e, ci):
    """Elo with a small ±1-bootstrap-SD error bar under it for the table."""
    if e is None:
        return "—"
    sd = (ci or {}).get("sd")
    if sd is None:
        return str(e)
    return f"{e}<div class='small'>±{sd:.0f}</div>"


def _charts_js(payload: str, models: list, wdl_pct: bool = False) -> str:
    """The Chart.js init block shared by both report layouts. Charts bind to
    canvases by id, so the canvases may live in any section of the page as long
    as this script runs once after them. wdl_pct renders win/draw/loss as a
    100%-stacked share (comparable across models with unequal game counts)
    instead of raw counts."""
    dn_map = json.dumps({m: display_name(m) for m in models})
    # The win/draw/loss chart (built as a plain string so the False branch stays
    # byte-identical to the original count-based chart).
    if wdl_pct:
        pct_helper = ("const wpct=(m,k)=>{const t=pm[m].wins+pm[m].draws+pm[m].losses;"
                      " return t?100*pm[m][k]/t:0;};\n")
        wdl_chart = """new Chart(document.getElementById('wdl'), { type:'bar',
  data:{ labels:M.map(dn), datasets:[
    {label:'win %', backgroundColor:'#4ade80', data:M.map(m=>wpct(m,'wins'))},
    {label:'draw %', backgroundColor:'#94a3b8', data:M.map(m=>wpct(m,'draws'))},
    {label:'loss %', backgroundColor:'#f87171', data:M.map(m=>wpct(m,'losses'))},
  ]},
  options:{ scales:{x:{stacked:true},y:{stacked:true,max:100,title:{display:true,text:'% of games'}}}, plugins:{legend:{position:'bottom'}} } });"""
    else:
        pct_helper = ""
        wdl_chart = """new Chart(document.getElementById('wdl'), { type:'bar',
  data:{ labels:M.map(dn), datasets:[
    {label:'win', backgroundColor:'#4ade80', data:M.map(m=>pm[m].wins)},
    {label:'draw', backgroundColor:'#94a3b8', data:M.map(m=>pm[m].draws)},
    {label:'loss', backgroundColor:'#f87171', data:M.map(m=>pm[m].losses)},
  ]},
  options:{ scales:{x:{stacked:true},y:{stacked:true}}, plugins:{legend:{position:'bottom'}} } });"""
    return f"""<script>
const R = {payload};
const pm=R.per_model, M=R.models;
const DN = {dn_map};
const dn = m => DN[m] || m;
{pct_helper}{CHART_SETUP}
const COLORS=PALETTE;
const ranked=[...M].sort((a,b)=>R.elo[b]-R.elo[a]);
const eloRanked=ranked.filter(m=>R.elo[m]!=null);  // unrated (—) models omitted from the chart

const ELO_CI = R.elo_ci || {{}};
// Draw a horizontal ±1-SD whisker over each Elo bar (bootstrap uncertainty).
const eloWhiskers = {{ id:'eloWhiskers', afterDatasetsDraw(c) {{
  const {{ctx, scales:{{x, y}}}} = c;
  ctx.save(); ctx.strokeStyle='#1c1c1c'; ctx.lineWidth=1.5;
  eloRanked.forEach((m, i) => {{
    const ci = ELO_CI[m]; if (!ci || ci.sd == null) return;
    const yc = y.getPixelForValue(i), cap = 5;
    const x1 = x.getPixelForValue(R.elo[m]-ci.sd), x2 = x.getPixelForValue(R.elo[m]+ci.sd);
    ctx.beginPath();
    ctx.moveTo(x1, yc); ctx.lineTo(x2, yc);
    ctx.moveTo(x1, yc-cap); ctx.lineTo(x1, yc+cap);
    ctx.moveTo(x2, yc-cap); ctx.lineTo(x2, yc+cap);
    ctx.stroke();
  }});
  ctx.restore();
}} }};
new Chart(document.getElementById('elo'), {{ type:'bar',
  data:{{ labels:eloRanked.map(dn), datasets:[{{label:'Elo', backgroundColor:ACCENT,
    data:eloRanked.map(m=>R.elo[m])}}]}},
  options:{{ indexAxis:'y',
    scales:{{x:{{min:Math.min(...eloRanked.map(m=>R.elo[m]-(ELO_CI[m]?.sd||0)))-20}}}},
    plugins:{{legend:{{display:false}}}} }}, plugins:[eloWhiskers] }});

{wdl_chart}

new Chart(document.getElementById('tac'), {{ type:'bar',
  data:{{ labels:M.map(dn), datasets:[
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
</script>"""


def _why_analysis(game: str) -> str:
    """The 'why win / why lose' static block (defense + threat-axis + diagonals)
    shown in section 2, kept in editable HTML under
    scripts/board_analysis/<game>_why.html."""
    path = os.path.join(_ANALYSIS_DIR, f"{game}_why.html")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def render_game(game: str, rep: dict) -> str:
    """Conclusion-first layout in three sections: (1) Results — who won,
    (2) Why — what separates them, (3) Analysis."""
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    elo = rep["elo"]
    ranked = sorted(models, key=lambda m: _elo_key(elo, m), reverse=True)

    # Slim "who won" table: outcomes only (Elo, net/game, win%, 1st/2nd-move).
    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{model_cell(m)}</td>
          <td>{_elo_cell(rep['elo'][m], rep.get('elo_ci', {}).get(m))}</td>
          <td class='{net_cls}'>{s['net_per_game']:+.2f}</td>
          <td>{s['win_rate']*100:.0f}%</td>
          <td>{s['first_move_win_rate']*100:.0f}%</td>
          <td>{s['second_move_win_rate']*100:.0f}%</td>
          <td>{s['games']}</td>
          {token_cost_cells(m, rep['avg_tokens'].get(m))}
        </tr>"""

    # head-to-head
    hh = "<tr><th></th>" + "".join(f"<th>{display_name(m)}</th>" for m in models) + "</tr>"
    for a in models:
        hh += f"<tr><th class='model'>{model_cell(a)}</th>"
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
        heat_cards += (f"<div class='heatcard'><div class='hlabel'>{display_name(m)}</div>"
                       f"{_heat_html(pm[m]['heat'])}</div>")

    fpw = rep["first_player_win_rate"] * 100
    replay_btn = (f'<a class="replaybtn" href="{game}_replay.html?cacheBust=19">'
                  f'🎬 Watch featured replays →</a>')
    emoji, name = TITLE[game].split(" ", 1)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {TITLE[game]}</title>
{_favicon(FAVICON[game])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>{BASE_CSS}
  .heatwrap {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; }}
  .heatcard {{ text-align:center; }}
  .hlabel {{ font-size:11px; color:var(--dim); margin-bottom:6px; }}
  .board {{ display:grid; gap:1px; background:var(--line); padding:1px; }}
  .cell {{ width:14px; height:14px; }}
  /* Prominent dividers for the three top-level sections (Results / Why / More). */
  h2.section {{ font-size:23px; margin:56px 0 18px; padding-top:16px;
    border-top:3px solid var(--red); color:var(--red); letter-spacing:.01em; }}
  h2.section:first-of-type {{ margin-top:32px; }}
</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/{game}<span class="cursor"></span></h1>
  <div class="sub">{emoji} {name} · Perfect-information game · round-robin · {rep['num_games']} games · board {rep['size'][0]}×{rep['size'][1]}</div>
  {replay_btn}

  {_intro_html(game)}

  <h2 class="section">1 · Results — who won</h2>
  <table>
    <tr><th>#</th><th class='model'>Model</th><th>Elo</th><th>Net/game</th><th>Win%</th>
        <th>1st-move win%</th><th>2nd-move win%</th><th>Games</th>{TOKEN_HEADERS}</tr>
    {rows}
  </table>
  {_legend('board_results')}
  <div class="grid2">
    <div><h3>Elo rating</h3><canvas id="elo"></canvas>
      <div class="note">Whiskers show ±1 bootstrap SD (resampling games 300×) — wider bars mean
        fewer/less-decisive games, so ratings within a whisker of each other are a near tie.</div></div>
    <div><h3>Win / draw / loss</h3><canvas id="wdl"></canvas></div>
  </div>
  <h3>⚔️ Head-to-head (row wins–losses vs column)</h3>
  <table class='h2h'>{hh}</table>

  <h2 class="section">2 · Why — what decides win &amp; loss</h2>
  <div class="grid2">
    <div><h3>Tactical accuracy (win-take / block %)</h3><canvas id="tac"></canvas></div>
    <div><h3>Blunder rate by game phase</h3><canvas id="phase"></canvas></div>
  </div>
  {_why_analysis(game)}

  <h2 class="section">3 · Analysis</h2>
  <div class="grid2">
    <div><h3>Game-length distribution (plies)</h3><canvas id="len"></canvas></div>
    <div></div>
  </div>
  <h3>🗺️ Move-location heatmap</h3>
  <div class="note">Where each model places pieces (brighter = more frequent). Reveals
    center-control bias and opening preferences.</div>
  <div class="heatwrap">{heat_cards}</div>

{_charts_js(payload, models, wdl_pct=True)}
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


def _minirank(entry: dict) -> str:
    """Compact full ranking for a game card: rank · model · Elo bar."""
    ranking = entry.get("ranking") or []
    ratings = entry.get("ratings") or {}
    if not ranking:
        return f"<div class='champ'>{entry.get('champ_line', '')}</div>"
    vals = [ratings.get(m) for m in ranking if ratings.get(m) is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    rng = (hi - lo) or 1
    # Elo-scale ratings (≈1500) vs small per-hand metrics (e.g. blackjack net/hand)
    elo_mode = max((abs(v) for v in vals), default=0) > 50
    rows = ""
    for i, m in enumerate(ranking, 1):
        r = ratings.get(m)
        pct = 25 + 75 * (r - lo) / rng if r is not None else 0
        val = "" if r is None else (_elo_txt(r) if elo_mode else f"{r:+.3f}")
        bar = (f"<span class='mr-bar'><i style='width:{pct:.0f}%'></i></span>"
               if r is not None else "<span class='mr-bar'></span>")
        rows += (f"<div class='mr-row'><span class='mr-rk'>{i}</span>"
                 f"<span class='mr-nm'>{model_cell(m)}</span>"
                 f"{bar}<span class='mr-val'>{val}</span></div>")
    return f"<div class='minirank'>{rows}</div>"


def _index_card(entry: dict) -> str:
    badges = "".join(f"<span class='badge'>{b}</span>" for b in entry["badges"])
    return f"""
        <a class="card" href="{entry['href']}">
          <div class="ctitle">{entry['title']}</div>
          <div class="badges">{badges}</div>
          <div class="cmeta">{entry['meta']}</div>
          {_minirank(entry)}
          <div class="cgo">View analysis →</div>
        </a>"""


# Games that count toward the overall (cross-game) Arena Score. Every game still
# gets its own report and landing-page card; these are just the ones averaged
# into the headline ranking (a curated, comparable subset). Blackjack is excluded
# — it's vs-dealer (no head-to-head Elo) and luck-dominated, so it would add noise
# to both the rank score and the Elo composite.
ARENA_GAMES = {"connect4", "gomoku", "holdem", "match",
               "repeated_colonel_blotto", "leduc_poker"}


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


def _arena_elo_scores(entries: list) -> dict:
    """Cross-game Elo composite keyed by model.

    Each game's native ratings (Bradley-Terry / chip-weighted Elo; for Blackjack,
    mean chips/hand, which has no head-to-head Elo) live on different scales and
    spreads, so we **z-score within each game** — (rating − field mean) / field
    SD — then average a model's z across the games it played. That standardized
    mean is rescaled to a 1500-centered, Elo-like number (±150 per SD) so it
    reads next to the per-game Elos. Unlike the rank score it keeps *margin*:
    winning a game by a mile beats winning by a hair.
    """
    per_model = {}  # model -> [z, ...]
    for e in entries:
        ratings = e.get("ratings") or {}
        vals = [v for v in ratings.values() if v is not None]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if sd == 0:
            continue
        for m, v in ratings.items():
            if v is not None:
                per_model.setdefault(m, []).append((v - mean) / sd)
    out = {}
    for m, zs in per_model.items():
        mean_z = sum(zs) / len(zs)
        out[m] = {"arena_elo": round(1500 + 150 * mean_z), "z": round(mean_z, 2),
                  "games": len(zs)}
    return out


_EXCLUDE_OVERVIEW = {"gpt-oss-120b"}   # kept off the cross-game ranking (incomplete schedule)


def _arena_board(entries: list) -> str:
    # Drop excluded models from each game's ranking/ratings so the cross-game
    # ranks and per-game z-scores are recomputed without them.
    arena_entries = []
    for e in entries:
        if e["key"] not in ARENA_GAMES:
            continue
        fe = dict(e)
        fe["ranking"] = [m for m in e.get("ranking", []) if m not in _EXCLUDE_OVERVIEW]
        fe["ratings"] = {m: v for m, v in (e.get("ratings") or {}).items()
                         if m not in _EXCLUDE_OVERVIEW}
        arena_entries.append(fe)
    rows = _arena_scores(arena_entries)
    if not rows:
        return ""
    elo = _arena_elo_scores(arena_entries)
    total = len(arena_entries)
    body = ""
    for i, r in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}")
        ae = elo.get(r["model"], {})
        ae_disp = (f"{ae['arena_elo']}<div class='small'>{ae['z']:+.2f} SD</div>"
                   if ae else "—")
        ae_elo = ae.get("arena_elo", "") if ae else ""
        body += (
            f"<tr data-score='{r['score']}' data-elo='{ae_elo}'><td class='rk'>{medal}</td>"
            f"<td class='model'>{model_cell(r['model'])}</td>"
            f"<td class='scorecell'><span class='bar' style='width:{r['score']}%'></span>"
            f"<span class='sval'>{r['score']:.0f}</span></td>"
            f"<td><b>{ae_disp}</b></td>"
            f"<td class='cov'>{r['games']}/{total}</td>"
            f"<td class='best'>{r['best']}</td></tr>")
    return f"""
  <section class="board">
    <div class="arena-head"><h2>🏅 Model leaderboard</h2>
      <span class="arena-tag">6 core games · click a score column to re-rank</span></div>
    <table class="lb">
      <thead>
      <tr><th class='rk'>#</th><th class='model'>Model</th>
        <th class="sortable sorted" data-sort="score"><span class="hcell">Arena Rank Score<button class="info" aria-controls="info-score"
          aria-label="How Arena Rank Score is calculated">i</button>
          <span class="infobox" id="info-score" hidden><b>Arena Rank Score — ordinal.</b>
          In each game a model's finish becomes a 0–1 score, evenly spaced
          (1st&nbsp;=&nbsp;1.0, last&nbsp;=&nbsp;0.0) via <code>(N−1−rank)/(N−1)</code> where N is the
          field size. The score is the mean of those across the games played, ×100. It ignores
          <i>margin</i> — winning by a mile or a hair both score 1.0.</span></span></th>
        <th class="sortable" data-sort="elo"><span class="hcell">Arena Elo<button class="info" aria-controls="info-elo"
          aria-label="How Arena Elo is calculated">i</button>
          <span class="infobox" id="info-elo" hidden><b>Arena Elo — margin-aware.</b>
          Each game's Elo (Bradley-Terry / chip-weighted) is standardized within its field,
          <code>z = (rating − mean) / SD</code>. A model's z is averaged across games and rescaled to
          <code>1500 + 150·z</code>, so a model one SD above the field reads as 1650. Unlike the rank
          score it rewards <i>how much</i> you win by. Built from the six head-to-head games only —
          Blackjack is excluded (no opponent Elo, luck-dominated).</span></span></th>
        <th>coverage</th><th>best game</th></tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    <div class="note lb-note">Two cross-game summaries over six head-to-head games: Connect Four, Gomoku,
      Hold'em 1-Hand, Hold'em Match, Colonel Blotto and Leduc Holdem. Click a <b>score header</b> to
      sort by it, or the <b>ⓘ</b> for how it's computed. Coverage = games a model has entered (treat
      low coverage as provisional).</div>
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
            "ranking": ordered, "ratings": dict(rep["elo"]), **GAME_TAXONOMY[game],
        })

    # Kuhn first inside the imperfect group: it is the simplest game (a solved,
    # tiny-state game scored against GTO), so it reads as the natural entry point.
    # Kuhn omitted from the overview cards (still has its own report/nav entry).

    # Three Hold'em formats, distinct enough to stand alone: 1-Hand (each hand
    # scored independently, bb/100), Match (heads-up, stacks carried, win the
    # match), and Table (5-handed ring, scored by finishing rank).
    holdem_path = os.path.join(REPORT_DIR, "holdem_tournament_analysis.json")
    if os.path.exists(holdem_path):
        h = json.load(open(holdem_path))
        pm = h["per_model"]
        ordered = sorted(h["models"], key=lambda m: _elo_key(h["elo"], m), reverse=True)
        champ = ordered[0]
        entries.append({
            "key": "holdem", "title": "🃏 Hold'em 1-Hand",
            "href": "holdem_tournament_report.html",
            "meta": (f"heads-up · {h['num_games']} tables · {h['hands_per_game']} hands each"
                     f" · chip-weighted Elo"),
            "champ_line": f"🏆 {champ} <span class='metric'>Elo {_elo_txt(h['elo'][champ])}</span>",
            "ranking": ordered, "ratings": dict(h["elo"]), **GAME_TAXONOMY["holdem"],
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
            "champ_line": f"🏆 {champ['model']} <span class='metric'>Elo {_elo_txt(champ['elo'])}</span>",
            "ranking": [r["model"] for r in lb],
            "ratings": {r["model"]: r.get("elo") for r in lb}, **GAME_TAXONOMY["match"],
        })

    # Hold'em Table omitted from the overview cards (still has its own report).

    # New-games tournament (independent_blackjack, leduc, blotto, othello). Its
    # analyzer (analyze_new_games.py) writes a compact entry list that already
    # carries title/href/group/badges/ranking/meta/champ_line, so each row drops
    # straight into both the cards and the cross-game Arena Score.
    newgames_path = os.path.join(REPORT_DIR, "new_games_index.json")
    if os.path.exists(newgames_path):
        # Othello omitted from the overview cards (still has its own report).
        entries.extend(e for e in json.load(open(newgames_path))
                       if e.get("key") != "othello_lite_6x6")

    def _group(name):
        cards = "".join(_index_card(e) for e in entries if e["group"] == name)
        return f"<div class='cards'>{cards}</div>" if cards else \
            "<div class='empty'>No games yet.</div>"

    board = _arena_board(entries)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena</title>
{_favicon("🎲")}
{NAV_HEAD}
<style>{BASE_CSS}
  /* Leaderboard note shows on desktop only; on mobile it lives in the Q&A. */
  @media (max-width:760px) {{ .lb-note {{ display:none; }} }}
  .arena {{ margin-top:34px; border:1px solid var(--line); padding:20px 20px 24px;
    background:var(--panel); scroll-margin-top:60px; }}
  .arena-head {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px; }}
  .arena-tag {{ font-size:11px; color:var(--dim); }}
  .arena-tag::before {{ content:"["; color:var(--red); }} .arena-tag::after {{ content:"]"; color:var(--red); }}
  .group-label {{ display:flex; align-items:center; gap:10px; font-size:13px;
    font-weight:700; margin:24px 0 12px; padding-left:10px; border-left:3px solid var(--red); }}
  .group-label.perfect {{ border-left-color:var(--pos); }}
  .group-label.imperfect {{ border-left-color:#b45309; }}
  .group-label .gl-sub {{ font-size:11px; font-weight:400; color:var(--dim); }}
  .card {{ display:block; text-decoration:none; color:inherit; transition:border-color .15s; }}
  .card:hover {{ border-color:var(--red); }}
  .ctitle {{ font-size:15px; font-weight:700; color:var(--red); }}
  .ctitle::before {{ content:"> "; color:var(--dim); }}
  .badges {{ margin:8px 0; display:flex; gap:10px; flex-wrap:wrap; }}
  .badge {{ font-size:10px; color:var(--dim); }}
  .badge::before {{ content:"["; }} .badge::after {{ content:"]"; }}
  .cmeta {{ font-size:12px; color:var(--dim); margin:0 0 12px; }}
  .champ {{ font-size:13px; }} .metric {{ color:var(--red); font-weight:700; }}
  .minirank {{ margin:8px 0 4px; }}
  .mr-row {{ display:grid; grid-template-columns:18px 1fr 76px 56px; align-items:center;
    gap:8px; font-size:12px; padding:2px 0; }}
  .mr-rk {{ color:var(--dim); text-align:right; font-variant-numeric:tabular-nums; }}
  .mr-nm {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .mr-bar {{ position:relative; height:9px; background:var(--faint); border:1px solid var(--line); }}
  .mr-bar i {{ position:absolute; left:0; top:0; bottom:0; background:var(--red); opacity:.5; }}
  .mr-val {{ text-align:right; color:var(--red); font-weight:600; font-variant-numeric:tabular-nums; }}
  .cgo {{ margin-top:12px; font-size:12px; color:var(--red); }}
  table.lb td, table.lb th {{ text-align:center; }}
  table.lb td.rk, table.lb th.rk {{ width:34px; color:var(--dim); }}
  table.lb td.cov {{ color:var(--dim); }} table.lb td.best {{ color:var(--dim); }}
  .scorecell {{ position:relative; min-width:140px; }}
  .scorecell .bar {{ position:absolute; left:0; top:50%; transform:translateY(-50%);
    height:16px; background:var(--red); opacity:.16; }}
  .scorecell .sval {{ position:relative; font-weight:700; color:var(--red); }}
  .cta {{ display:flex; align-items:center; justify-content:space-between; gap:16px;
    flex-wrap:wrap; border:1px dashed var(--line); padding:22px; background:var(--faint); }}
  .cta .ctatext b {{ color:var(--fg); font-size:15px; }}
  .cta .ctatext div {{ color:var(--dim); font-size:13px; margin-top:4px; }}
  .cta .pill {{ font-size:13px; color:var(--panel); background:var(--red); font-weight:700;
    padding:9px 16px; white-space:nowrap; }}
  .empty {{ color:var(--dim); font-size:13px; padding:8px 0; }}
</style></head>
<body><div class="wrap">
  <h1>$ ai-battle-arena -<span class="cursor"></span></h1>
  <div class="sub">Two arenas, same games. <b>Model Arena</b> pits raw models through one
    identical pipeline; <b>Harness Arena</b> is open to any model + any scaffolding.</div>

  {board}

  <section class="arena" id="model">
    <div class="arena-head"><h2>🤖 Model Arena</h2>
      <span class="arena-tag">fair · one identical generic pipeline</span></div>
    <div class="note">Every model plays through the same prompt/parse/retry wrapper — this
      measures the model, not the scaffolding. <b>Open-weight models are served via
      Fireworks AI</b>; the closed models (Claude, GPT-5.x) run on their own provider APIs.</div>
    <div class="group-label imperfect">🎭 Imperfect information
      <span class="gl-sub">hidden cards · stochastic</span></div>
    {_group("imperfect")}
    <div class="group-label perfect">♟ Perfect information
      <span class="gl-sub">full state visible · deterministic</span></div>
    {_group("perfect")}
  </section>

  <section class="arena" id="agentic">
    <div class="arena-head"><h2>🛠️ Harness Arena</h2></div>
    <div class="cta">
      <span class="pill">Coming soon</span>
    </div>
  </section>
</div>
<script>
  // Click an .info button to open its popover (closing any other); click outside
  // or press Escape to dismiss. No dependencies.
  document.addEventListener('click', function (e) {{
    var btn = e.target.closest('.info');
    var boxes = document.querySelectorAll('.infobox');
    if (btn) {{
      var box = document.getElementById(btn.getAttribute('aria-controls'));
      var wasHidden = box && box.hasAttribute('hidden');
      boxes.forEach(function (b) {{ b.setAttribute('hidden', ''); }});
      if (box && wasHidden) box.removeAttribute('hidden');
      e.stopPropagation();
    }} else if (!e.target.closest('.infobox')) {{
      boxes.forEach(function (b) {{ b.setAttribute('hidden', ''); }});
    }}
  }});
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'Escape')
      document.querySelectorAll('.infobox').forEach(function (b) {{ b.setAttribute('hidden', ''); }});
  }});

  // Click a score column header to re-rank the leaderboard by it (desc). The
  // rank column (medals) is renumbered to match.
  (function () {{
    var table = document.querySelector('table.lb');
    if (!table || !table.tBodies.length) return;
    var tbody = table.tBodies[0];
    var ths = table.querySelectorAll('th[data-sort]');
    var MEDALS = ['🥇', '🥈', '🥉'];
    function sortBy(key) {{
      var rows = Array.prototype.slice.call(tbody.rows);
      rows.sort(function (a, b) {{
        var x = parseFloat(a.dataset[key]), y = parseFloat(b.dataset[key]);
        if (isNaN(x)) return 1;
        if (isNaN(y)) return -1;
        return y - x;
      }});
      rows.forEach(function (r, i) {{
        var rk = r.querySelector('.rk');
        if (rk) rk.textContent = i < 3 ? MEDALS[i] : (i + 1);
        tbody.appendChild(r);
      }});
      ths.forEach(function (th) {{ th.classList.toggle('sorted', th.dataset.sort === key); }});
    }}
    ths.forEach(function (th) {{
      th.addEventListener('click', function (e) {{
        if (e.target.closest('.info')) return;  // let the ⓘ open its popover
        sortBy(th.dataset.sort);
      }});
    }});
  }})();
</script>
</body></html>"""


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
        rep["avg_tokens"] = tokens_from_episodes(
            ep for g in data["games"] for ep in g["episodes"])
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
