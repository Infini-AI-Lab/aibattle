"""Analyze the new-games tournament (independent_blackjack, leduc_poker,
repeated_colonel_blotto, othello_lite_6x6) and emit one interactive HTML report
PER GAME — the new-games counterpart of analyze_board_tournament.py.

Two report shapes, picked by game structure:

  * versus  (leduc / blotto / othello, round-robin seat-swapped): per-model
    results (win/draw/loss, net/game, invalid%, plies, latency), a head-to-head
    win matrix, and Bradley-Terry / Elo ratings fit from all head-to-head
    results. Mirrors the board-game report minus the board-specific tactical
    signals.

  * dealer  (independent_blackjack, model vs the built-in dealer): per-model
    profit, mean/hand, win/push/loss split, plus blackjack-specific play signals
    (bust rate, double rate, natural rate) read from the per-hand episode files.

Reads each game's coached run folder (runs/blackjack, runs/leduc_poker,
runs/colonel_blotto, runs/othello — see GAMES[*]["dir"]); writes <name>_report.html
per game to reports/, plus a
small new_games_index.json that analyze_board_tournament.render_index() reads to
add the four games to the landing page and the cross-game Arena Score.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# Per-game presentation + taxonomy. "dir" is the per-game run folder under runs/
# (coached is canonical, one flat folder per game). "kind" selects the report
# shape; "group" (perfect/imperfect) and "badges" drive the landing-page card,
# matching the vocabulary in analyze_board_tournament.GAME_TAXONOMY.
GAMES = {
    "independent_blackjack": {
        "dir": "blackjack", "kind": "dealer", "title": "🃏 Blackjack", "emoji": "🃏",
        "href": "blackjack_report.html", "group": "imperfect",
        "badges": ["Imperfect info", "vs Dealer", "Stochastic"],
        "blurb": "Model vs the built-in dealer · hit/stand/double · scored by chip profit",
    },
    "leduc_poker": {
        "dir": "leduc_poker", "kind": "versus", "title": "🃏 Leduc Poker", "emoji": "🃏",
        "href": "leduc_report.html", "group": "imperfect",
        "badges": ["Imperfect info", "Heads-up", "Stochastic"],
        "blurb": "Imperfect-information poker · 6-card deck · round-robin, seat-swapped",
    },
    "repeated_colonel_blotto": {
        "dir": "colonel_blotto", "kind": "versus", "title": "⚔️ Colonel Blotto", "emoji": "⚔️",
        "href": "blotto_report.html", "group": "imperfect",
        "badges": ["Imperfect info", "Heads-up", "Simultaneous"],
        "blurb": "Simultaneous resource allocation · repeated rounds · round-robin, seat-swapped",
    },
    "othello_lite_6x6": {
        "dir": "othello", "kind": "versus", "title": "⚫ Othello 6×6", "emoji": "⚫",
        "href": "othello_report.html", "group": "perfect",
        "badges": ["Perfect info", "2P", "Deterministic"],
        "blurb": "Perfect-information board game · 6×6 board · round-robin, seat-swapped",
    },
}

NAV_HEAD = '<link rel="stylesheet" href="nav.css"><script defer src="nav.js"></script>'


def _favicon(emoji: str) -> str:
    return (f"<link rel=\"icon\" href=\"data:image/svg+xml,"
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            f"<text y='.9em' font-size='90'>{emoji}</text></svg>\">")


# ---------------------------------------------------------------------------
def bradley_terry(models, h2h, iters=300):
    """Fit Bradley-Terry strengths from pairwise results; return (strength, elo).

    Draws count as half a win to each side. Elo is the BT log-strength on the
    400/decade scale, recentred so the rated field averages 1500. A model with
    no wins/draws (or no losses/draws) has no finite rating and is reported with
    elo=None (rendered as "—"), excluded from the fit so a degenerate record does
    not drag the whole field to nonsense. Same fit as the board-game report.
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

    rated = [m for m in models
             if (wins[m] + draws[m]) > 0 and (losses[m] + draws[m]) > 0]

    W = {m: 0.0 for m in rated}
    N = defaultdict(lambda: defaultdict(float))
    for a in rated:
        for b in rated:
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


def _len_bins(all_lengths):
    lo, hi = min(all_lengths), max(all_lengths)
    nbins = min(12, max(hi - lo + 1, 1))
    step = max((hi - lo) / nbins, 1)
    edges = [lo + i * step for i in range(nbins + 1)]
    edges[-1] = hi
    labels = [f"{int(round(edges[i]))}–{int(round(edges[i + 1]))}"
              for i in range(len(edges) - 1)]
    return edges, labels


def _elo_txt(e):
    return "—" if e is None else str(e)


def _elo_key(elo, m):
    return elo[m] if elo[m] is not None else float("-inf")


def _latency(step):
    return (step.get("response") or {}).get("metadata", {}).get("latency_ms")


# ---------------------------------------------------------------------------
def analyze_versus(game: str, data: dict) -> dict:
    models = data["models"]
    stats = {m: {"games": 0, "wins": 0, "losses": 0, "draws": 0, "net": 0.0,
                 "decisions": 0, "invalid": 0, "lengths": [], "latencies": [],
                 "first_moves": 0, "first_move_wins": 0} for m in models}
    h2h = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [w,l,d] a vs b
    fp_games = fp_wins = 0

    for pair in data["pairs"]:
        a, b = pair["a"], pair["b"]
        for e in pair["episodes"]:
            seat = e["seat_assignment"]
            winner = e.get("winner_name")
            length = e["length"]
            for s_, nm in seat.items():
                st = stats[nm]
                st["games"] += 1
                st["lengths"].append(length)
                pay = e["returns"][s_]
                st["net"] += pay
                if pay > 0:
                    st["wins"] += 1
                elif pay < 0:
                    st["losses"] += 1
                else:
                    st["draws"] += 1

            if winner == a:
                h2h[a][b][0] += 1; h2h[b][a][1] += 1
            elif winner == b:
                h2h[b][a][0] += 1; h2h[a][b][1] += 1
            else:
                h2h[a][b][2] += 1; h2h[b][a][2] += 1

            steps = e["steps"]
            if steps:
                fm = steps[0]["agent_name"]
                stats[fm]["first_moves"] += 1
                fp_games += 1
                if winner == fm:
                    stats[fm]["first_move_wins"] += 1
                    fp_wins += 1
            for s in steps:
                nm = s["agent_name"]
                st = stats[nm]
                st["decisions"] += 1
                if s.get("invalid"):
                    st["invalid"] += 1
                lat = _latency(s)
                if lat:
                    st["latencies"].append(lat)

    all_lengths = [ln for m in models for ln in stats[m]["lengths"]]
    edges, bin_labels = _len_bins(all_lengths)

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
            "invalid_rate": round(st["invalid"] / max(st["decisions"], 1), 4),
            "decisions": st["decisions"],
            "avg_len": round(sum(st["lengths"]) / len(st["lengths"]), 2)
                       if st["lengths"] else 0,
            "avg_latency_s": round(sum(st["latencies"]) / len(st["latencies"]) / 1000, 1)
                             if st["latencies"] else 0.0,
            "len_hist": _histogram(st["lengths"], edges),
        }

    _, elo = bradley_terry(models, h2h)
    h2h_out = {a: {b: h2h[a][b] for b in models if b != a} for a in models}
    return {
        "game": game, "kind": "versus", "models": models, "per_model": out,
        "h2h": h2h_out, "elo": elo, "len_bins": bin_labels,
        "first_player_win_rate": round(fp_wins / max(fp_games, 1), 4),
        "num_games": sum(len(p["episodes"]) for p in data["pairs"]),
        "episodes_per_pair": data.get("episodes_per_pair"),
    }


def analyze_dealer(game: str, data: dict) -> dict:
    models = data["models"]
    out = {}
    run_dir = os.path.join("runs", GAMES[game]["dir"])
    for m in models:
        eps = sorted(glob.glob(os.path.join(run_dir, f"{m}__vs__dealer", "ep*.json")))
        hands = wins = losses = pushes = busts = doubles = naturals = 0
        invalid = decisions = 0
        profit = 0.0
        lats = []
        for path in eps:
            e = json.load(open(path))
            # The model always occupies player_0 against the dealer.
            pay = e["returns"]["player_0"]
            hands += 1
            profit += pay
            if pay > 0:
                wins += 1
            elif pay < 0:
                losses += 1
            else:
                pushes += 1
            if e.get("player_bust"):
                busts += 1
            if e.get("doubled"):
                doubles += 1
            if e.get("player_natural"):
                naturals += 1
            for s in e.get("steps", []):
                if s.get("agent_name") != m:
                    continue
                decisions += 1
                if s.get("invalid"):
                    invalid += 1
                lat = _latency(s)
                if lat:
                    lats.append(lat)
        h = max(hands, 1)
        out[m] = {
            "hands": hands, "profit": round(profit, 2),
            "mean_per_hand": round(profit / h, 4),
            "win_rate": round(wins / h, 4), "loss_rate": round(losses / h, 4),
            "push_rate": round(pushes / h, 4),
            "bust_rate": round(busts / h, 4), "double_rate": round(doubles / h, 4),
            "natural_rate": round(naturals / h, 4),
            "invalid_rate": round(invalid / max(decisions, 1), 4),
            "avg_latency_s": round(sum(lats) / len(lats) / 1000, 1) if lats else 0.0,
        }
    total_hands = sum(out[m]["hands"] for m in models)
    field_profit = sum(out[m]["profit"] for m in models)
    return {
        "game": game, "kind": "dealer", "models": models, "per_model": out,
        "total_hands": total_hands, "field_profit": round(field_profit, 2),
    }


# ---------------------------------------------------------------------------
_HEAD_CSS = """
  body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }
  .wrap { max-width:1200px; margin:0 auto; padding:28px 28px 80px; }
  h1 { font-size:25px; } h2 { font-size:18px; margin-top:38px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }
  h3 { font-size:14px; color:#9aa3b5; }
  .sub { color:#8b93a7; }
  .kpis { display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }
  .kpi { background:#171a23; border:1px solid #232838; border-radius:10px; padding:12px 16px; }
  .kpi .v { font-size:22px; font-weight:700; color:#cdd6f4; }
  .kpi .l { font-size:11px; color:#8b93a7; text-transform:uppercase; letter-spacing:.04em; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th,td { padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }
  th { color:#9aa3b5; } td.model,th.model { text-align:left; font-weight:600; color:#cdd6f4; }
  .pos { color:#4ade80; } .neg { color:#f87171; } .diag { color:#3a3f4b; }
  .small { font-size:10px; color:#8b93a7; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:10px; }
  .note { color:#8b93a7; font-size:12px; margin:6px 0; }
  canvas { max-height:300px; }
  @media (max-width:760px) { .grid2 { grid-template-columns:1fr; } }
"""


def render_versus(rep: dict) -> str:
    cfg = GAMES[rep["game"]]
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    elo = rep["elo"]
    ranked = sorted(models, key=lambda m: _elo_key(elo, m), reverse=True)

    rows = ""
    for i, m in enumerate(sorted(models, key=lambda x: pm[x]["net_per_game"],
                                 reverse=True), 1):
        s = pm[m]
        net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{m}</td>
          <td>{_elo_txt(elo[m])}</td>
          <td class='{net_cls}'>{s['net_per_game']:+.2f}</td>
          <td>{s['win_rate']*100:.0f}%</td><td>{s['draw_rate']*100:.0f}%</td>
          <td>{s['first_move_win_rate']*100:.0f}%</td>
          <td>{s['invalid_rate']*100:.1f}%</td>
          <td>{s['avg_len']:.1f}</td><td>{s['avg_latency_s']:.1f}s</td>
        </tr>"""

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

    fpw = rep["first_player_win_rate"] * 100
    deals = (rep["episodes_per_pair"] or 0) // 2
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {cfg['title']}</title>
{_favicon(cfg['emoji'])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>{_HEAD_CSS}</style></head>
<body><div class="wrap">
  <h1>🎲 AI Battle Arena — {cfg['title']}</h1>
  <div class="sub">{cfg['blurb']} · {rep['num_games']} games</div>

  <div class="kpis">
    <div class="kpi"><div class="v">{_elo_txt(elo[ranked[0]])}</div><div class="l">top Elo · {ranked[0]}</div></div>
    <div class="kpi"><div class="v">{fpw:.0f}%</div><div class="l">first-mover win rate</div></div>
    <div class="kpi"><div class="v">{rep['num_games']}</div><div class="l">games played</div></div>
    <div class="kpi"><div class="v">{deals}</div><div class="l">deals/pair (seat-swapped)</div></div>
  </div>

  <h2>Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>Elo</th><th>net/game</th><th>win%</th>
        <th>draw%</th><th>1st-move win%</th><th>invalid%</th><th>plies</th><th>think</th></tr>
    {rows}
  </table>
  <div class="note">Elo from a Bradley-Terry fit over head-to-head results (rated field mean 1500).
    A model with no wins or no losses has no finite rating and is shown as “—” (excluded from the fit).
    net/game is the average game payoff (+1 win / −1 loss / 0 draw). Seats are swapped within every
    pair, so first-mover advantage is balanced across the field.</div>

  <div class="grid2">
    <div><h3>Elo rating</h3><canvas id="elo"></canvas></div>
    <div><h3>Win / draw / loss</h3><canvas id="wdl"></canvas></div>
  </div>
  <div class="grid2">
    <div><h3>Net result per game</h3><canvas id="net"></canvas></div>
    <div><h3>Game-length distribution (plies)</h3><canvas id="len"></canvas></div>
  </div>

  <h2>Head-to-head (row wins–losses vs column)</h2>
  <table class='h2h'>{hh}</table>

<script>
const R = {payload};
const pm=R.per_model, M=R.models, COLORS=['#60a5fa','#f472b6','#4ade80','#fbbf24','#a78bfa'];
Chart.defaults.color='#9aa3b5'; Chart.defaults.borderColor='#232838';
const eloRanked=[...M].filter(m=>R.elo[m]!=null).sort((a,b)=>R.elo[b]-R.elo[a]);

new Chart(document.getElementById('elo'), {{ type:'bar',
  data:{{ labels:eloRanked, datasets:[{{label:'Elo', backgroundColor:'#a78bfa',
    data:eloRanked.map(m=>R.elo[m])}}]}},
  options:{{ indexAxis:'y', scales:{{x:{{min:eloRanked.length?Math.min(...eloRanked.map(m=>R.elo[m]))-40:0}}}},
    plugins:{{legend:{{display:false}}}} }} }});

new Chart(document.getElementById('wdl'), {{ type:'bar',
  data:{{ labels:M, datasets:[
    {{label:'win', backgroundColor:'#4ade80', data:M.map(m=>pm[m].wins)}},
    {{label:'draw', backgroundColor:'#94a3b8', data:M.map(m=>pm[m].draws)}},
    {{label:'loss', backgroundColor:'#f87171', data:M.map(m=>pm[m].losses)}},
  ]}},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});

new Chart(document.getElementById('net'), {{ type:'bar',
  data:{{ labels:M, datasets:[{{label:'net/game', backgroundColor:M.map(m=>pm[m].net_per_game>=0?'#4ade80':'#f87171'),
    data:M.map(m=>pm[m].net_per_game)}}]}},
  options:{{ plugins:{{legend:{{display:false}}}} }} }});

new Chart(document.getElementById('len'), {{ type:'bar',
  data:{{ labels:R.len_bins, datasets:M.map((m,i)=>({{label:m,
    backgroundColor:COLORS[i%COLORS.length], data:pm[m].len_hist}}))}},
  options:{{ scales:{{y:{{title:{{display:true,text:'games'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});
</script>
</div></body></html>"""


def render_dealer(rep: dict) -> str:
    cfg = GAMES[rep["game"]]
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    ranked = sorted(models, key=lambda m: pm[m]["profit"], reverse=True)

    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        pcls = "pos" if s["profit"] > 0 else ("neg" if s["profit"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{m}</td>
          <td class='{pcls}'>{s['profit']:+.1f}</td>
          <td class='{pcls}'>{s['mean_per_hand']:+.3f}</td>
          <td>{s['win_rate']*100:.0f}%</td><td>{s['push_rate']*100:.0f}%</td>
          <td>{s['loss_rate']*100:.0f}%</td>
          <td>{s['bust_rate']*100:.0f}%</td><td>{s['double_rate']*100:.0f}%</td>
          <td>{s['natural_rate']*100:.0f}%</td>
          <td>{s['invalid_rate']*100:.1f}%</td>
          <td>{s['hands']}</td><td>{s['avg_latency_s']:.1f}s</td>
        </tr>"""

    champ = ranked[0]
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {cfg['title']}</title>
{_favicon(cfg['emoji'])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>{_HEAD_CSS}</style></head>
<body><div class="wrap">
  <h1>🎲 AI Battle Arena — {cfg['title']}</h1>
  <div class="sub">{cfg['blurb']} · {rep['total_hands']} hands total</div>

  <div class="kpis">
    <div class="kpi"><div class="v">{pm[champ]['profit']:+.1f}</div><div class="l">top profit · {champ}</div></div>
    <div class="kpi"><div class="v">{rep['field_profit']:+.1f}</div><div class="l">field net vs dealer</div></div>
    <div class="kpi"><div class="v">{rep['total_hands']}</div><div class="l">hands played</div></div>
  </div>

  <h2>Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>profit</th><th>mean/hand</th>
        <th>win%</th><th>push%</th><th>loss%</th><th>bust%</th><th>double%</th>
        <th>natural%</th><th>invalid%</th><th>hands</th><th>think</th></tr>
    {rows}
  </table>
  <div class="note">Each model plays the same independent hands against the built-in dealer; the dealer
    holds an inherent house edge, so a negative field net is expected. profit = total chips won/lost
    (a doubled hand pays ±2). bust = player busted; double = chose to double down; natural = dealt a
    blackjack. These read directly from the per-hand outcomes.</div>

  <div class="grid2">
    <div><h3>Chip profit</h3><canvas id="profit"></canvas></div>
    <div><h3>Win / push / loss</h3><canvas id="wpl"></canvas></div>
  </div>
  <div class="grid2">
    <div><h3>Play style (bust / double / natural %)</h3><canvas id="style"></canvas></div>
    <div></div>
  </div>

<script>
const R = {payload};
const pm=R.per_model, M=R.models;
Chart.defaults.color='#9aa3b5'; Chart.defaults.borderColor='#232838';
const ranked=[...M].sort((a,b)=>pm[b].profit-pm[a].profit);

new Chart(document.getElementById('profit'), {{ type:'bar',
  data:{{ labels:ranked, datasets:[{{label:'profit', backgroundColor:ranked.map(m=>pm[m].profit>=0?'#4ade80':'#f87171'),
    data:ranked.map(m=>pm[m].profit)}}]}},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}} }} }});

new Chart(document.getElementById('wpl'), {{ type:'bar',
  data:{{ labels:M, datasets:[
    {{label:'win', backgroundColor:'#4ade80', data:M.map(m=>pm[m].win_rate*100)}},
    {{label:'push', backgroundColor:'#94a3b8', data:M.map(m=>pm[m].push_rate*100)}},
    {{label:'loss', backgroundColor:'#f87171', data:M.map(m=>pm[m].loss_rate*100)}},
  ]}},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});

new Chart(document.getElementById('style'), {{ type:'bar',
  data:{{ labels:M, datasets:[
    {{label:'bust %', backgroundColor:'#f87171', data:M.map(m=>pm[m].bust_rate*100)}},
    {{label:'double %', backgroundColor:'#fbbf24', data:M.map(m=>pm[m].double_rate*100)}},
    {{label:'natural %', backgroundColor:'#60a5fa', data:M.map(m=>pm[m].natural_rate*100)}},
  ]}},
  options:{{ scales:{{y:{{min:0}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
</script>
</div></body></html>"""


# ---------------------------------------------------------------------------
def _index_entry(rep: dict) -> dict:
    """The compact record analyze_board_tournament.render_index() consumes to add
    this game as a landing-page card and a row in the cross-game Arena Score."""
    cfg = GAMES[rep["game"]]
    pm = rep["per_model"]
    base = {"key": rep["game"], "title": cfg["title"], "href": cfg["href"],
            "group": cfg["group"], "badges": cfg["badges"]}
    if rep["kind"] == "versus":
        ranked = sorted(rep["models"], key=lambda m: _elo_key(rep["elo"], m), reverse=True)
        champ = ranked[0]
        deals = (rep["episodes_per_pair"] or 0) // 2
        base.update({
            "ranking": ranked,
            "meta": (f"{rep['num_games']} games · {deals} deals/pair · "
                     f"first-mover {rep['first_player_win_rate']*100:.0f}%"),
            "champ_line": f"🏆 {champ} <span class='metric'>Elo {_elo_txt(rep['elo'][champ])}</span>",
        })
    else:  # dealer
        ranked = sorted(rep["models"], key=lambda m: pm[m]["profit"], reverse=True)
        champ = ranked[0]
        per_model_hands = rep["per_model"][champ]["hands"]
        base.update({
            "ranking": ranked,
            "meta": f"vs dealer · {per_model_hands} hands/model · chip profit",
            "champ_line": f"🏆 {champ} <span class='metric'>{pm[champ]['profit']:+.1f} profit</span>",
        })
    return base


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    index_entries = []
    for game, cfg in GAMES.items():
        path = os.path.join("runs", cfg["dir"], "data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        data = json.load(open(path))
        if cfg["kind"] == "versus":
            if not data.get("pairs"):
                print(f"skip {game}: no completed pairs yet")
                continue
            rep = analyze_versus(game, data)
            html = render_versus(rep)
        else:
            rep = analyze_dealer(game, data)
            if rep["total_hands"] == 0:
                print(f"skip {game}: no hands found under runs/{cfg['dir']}")
                continue
            html = render_dealer(rep)

        out = os.path.join(REPORT_DIR, cfg["href"])
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        json.dump(rep, open(os.path.join(REPORT_DIR, f"{game}_analysis.json"), "w"),
                  indent=2)
        index_entries.append(_index_entry(rep))
        print(f"Wrote {out}")

        pm = rep["per_model"]
        if rep["kind"] == "versus":
            order = sorted(pm, key=lambda x: _elo_key(rep["elo"], x), reverse=True)
            print(f"=== {game} ({rep['num_games']} games) ===")
            for m in order:
                s = pm[m]
                print(f"  {m:<16} elo={_elo_txt(rep['elo'][m]):>4} "
                      f"net/g={s['net_per_game']:+.2f} win%={s['win_rate']*100:3.0f} "
                      f"draw%={s['draw_rate']*100:3.0f} invalid={s['invalid_rate']*100:.1f}%")
        else:
            order = sorted(pm, key=lambda x: pm[x]["profit"], reverse=True)
            print(f"=== {game} ({rep['total_hands']} hands) ===")
            for m in order:
                s = pm[m]
                print(f"  {m:<16} profit={s['profit']:+.1f} mean/hand={s['mean_per_hand']:+.3f} "
                      f"win%={s['win_rate']*100:3.0f} bust%={s['bust_rate']*100:3.0f} "
                      f"invalid={s['invalid_rate']*100:.1f}%")

    if index_entries:
        json.dump(index_entries,
                  open(os.path.join(REPORT_DIR, "new_games_index.json"), "w"), indent=2)
        print(f"\nWrote {REPORT_DIR}/new_games_index.json ({len(index_entries)} games)")
        print("Now run analyze_board_tournament.py to refresh index.html with these entries.")


if __name__ == "__main__":
    main()
