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

Reads each game's run folder under runs/new_games_experiment/<game> (see
GAMES[*]["dir"]); writes terminal-styled <name>_report.html per game to reports/,
plus a small new_games_index.json (rankings/meta the index leaderboard uses).
Replay data for these games is built separately by build_new_games_replays.py.
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
        "dir": "new_games_experiment/independent_blackjack", "kind": "dealer",
        "title": "🂡 Blackjack", "emoji": "🂡",
        "href": "blackjack_report.html", "replay": "blackjack_replay.html", "area": "blackjack", "replay_verb": "hand",
        "group": "imperfect",
        "badges": ["Imperfect info", "vs Dealer", "Stochastic"],
        "blurb": "Model vs the built-in dealer · hit/stand/double · scored by chip profit",
    },
    "leduc_poker": {
        "dir": "new_games_experiment/leduc_poker", "kind": "versus",
        "title": "🎴 Leduc Poker", "emoji": "🎴",
        "href": "leduc_report.html", "replay": "leduc_replay.html", "area": "leduc", "replay_verb": "hand",
        "group": "imperfect",
        "badges": ["Imperfect info", "Heads-up", "Stochastic"],
        "blurb": "Imperfect-information poker · 6-card deck · round-robin, seat-swapped",
    },
    "repeated_colonel_blotto": {
        "dir": "new_games_experiment/repeated_colonel_blotto", "kind": "versus",
        "title": "⚔️ Colonel Blotto", "emoji": "⚔️",
        "href": "blotto_report.html", "replay": "blotto_replay.html", "area": "blotto", "replay_verb": "game",
        "group": "imperfect",
        "badges": ["Imperfect info", "Heads-up", "Simultaneous"],
        "blurb": "Simultaneous resource allocation · repeated rounds · round-robin, seat-swapped",
    },
    "othello_lite_6x6": {
        "dir": "new_games_experiment/othello_lite_6x6", "kind": "versus",
        "title": "⚫ Othello 6×6", "emoji": "⚫",
        "href": "othello_report.html", "replay": "othello_replay.html", "area": "othello", "replay_verb": "game",
        "group": "perfect",
        "badges": ["Perfect info", "2P", "Deterministic"],
        "blurb": "Perfect-information board game · 6×6 board · round-robin, seat-swapped",
    },
}

NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=4"><script defer src="nav.js?v=13"></script>'


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
_TERM_CSS = """
  :root { --bg:#fbfbf8; --fg:#1c1c1c; --red:#8f1d1d; --dim:#6b6b6b;
    --line:#ddd8cf; --panel:#ffffff; --faint:#f4f1ea; --green:#1a7f37; --neg:#b91c1c; }
  body { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    margin:0; background:var(--bg); color:var(--fg); }
  .wrap { max-width:1100px; margin:0 auto; padding:36px 24px; }
  h1 { font-size:22px; margin:0 0 4px; color:var(--red); font-weight:700; }
  h1 .cursor { display:inline-block; width:11px; height:1em; background:var(--red);
    vertical-align:text-bottom; margin-left:5px; animation:blink 1.1s steps(1) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  .sub { color:var(--dim); font-size:13px; margin-bottom:30px; }
  h2 { font-size:15px; margin:0; font-weight:700; }
  h2::before { content:"## "; color:var(--red); }
  .note { color:var(--dim); font-size:12px; margin:6px 0 14px; }

  .arena { margin-top:34px; border:1px solid var(--line); padding:20px 20px 24px;
    background:var(--panel); }
  .arena-head { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
  .arena-tag { font-size:11px; color:var(--dim); }
  .arena-tag::before { content:"["; color:var(--red); }
  .arena-tag::after { content:"]"; color:var(--red); }

  table.lb { border-collapse:collapse; width:100%; font-size:13px; margin-top:4px; }
  table.lb th, table.lb td { padding:7px 10px; border-bottom:1px solid var(--line); text-align:center; }
  table.lb th { color:var(--dim); font-weight:600; }
  table.lb td.model, table.lb th.model { text-align:left; font-weight:700; }
  table.lb td.rk, table.lb th.rk { width:34px; color:var(--dim); }
  .pos { color:var(--green); } .neg { color:var(--neg); }
  .scorecell { position:relative; min-width:120px; }
  .scorecell .bar { position:absolute; left:0; top:50%; transform:translateY(-50%);
    height:16px; background:var(--red); opacity:.16; }
  .scorecell .sval { position:relative; font-weight:700; color:var(--red); }

  table.h2h { border-collapse:collapse; width:100%; font-size:12px; margin-top:4px; }
  table.h2h th, table.h2h td { padding:6px 8px; border-bottom:1px solid var(--line); text-align:center; }
  table.h2h th { color:var(--dim); font-weight:600; }
  table.h2h td.model, table.h2h th.model { text-align:left; font-weight:700; }
  table.h2h td.diag { color:var(--line); }
  table.h2h .small { font-size:10px; color:var(--dim); }

  /* canonical replay button — matches the board/holdem reports: faint inset,
     indigo label, sits directly under the subtitle. */
  .replaybtn { display:inline-block; margin-top:12px; background:var(--faint); color:#4338ca;
    border:1px solid var(--line); padding:8px 14px; font-size:13px; text-decoration:none; }
  .replaybtn:hover { border-color:var(--red); color:var(--fg); }
"""


def _page(cfg: dict, area: str, sub: str, body: str, replay_label: str) -> str:
    """Terminal-style page shell shared by both report kinds."""
    btn = (f'  <a class="replaybtn" href="{cfg["replay"]}">▶ watch {replay_label} replays</a>\n'
           if cfg.get("replay") else "")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Battle Arena — {cfg['title']}</title>
{_favicon(cfg['emoji'])}
{NAV_HEAD}
<style>{_TERM_CSS}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/{area}<span class="cursor"></span></h1>
  <div class="sub">{cfg['emoji']} {sub}</div>
{btn}{body}
</div></body></html>"""


def _bar_cell(value: float, lo: float, hi: float, text: str) -> str:
    """A scorecell: faint red bar scaled to [lo,hi] behind a bold red value."""
    span = (hi - lo) or 1
    w = max(0.0, min(100.0, (value - lo) / span * 100))
    return (f"<td class='scorecell'><span class='bar' style='width:{w:.0f}%'></span>"
            f"<span class='sval'>{text}</span></td>")


def _medal(i: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))


def render_versus(rep: dict) -> str:
    cfg = GAMES[rep["game"]]
    pm = rep["per_model"]
    models = rep["models"]
    elo = rep["elo"]
    ranked = sorted(models, key=lambda m: _elo_key(elo, m), reverse=True)

    rated = [elo[m] for m in models if elo[m] is not None]
    lo, hi = (min(rated), max(rated)) if rated else (0, 1)

    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
        elo_cell = (_bar_cell(elo[m], lo, hi, _elo_txt(elo[m])) if elo[m] is not None
                    else "<td class='scorecell'><span class='sval'>—</span></td>")
        rows += (
            f"<tr><td class='rk'>{_medal(i)}</td><td class='model'>{m}</td>"
            f"{elo_cell}"
            f"<td class='{net_cls}'>{s['net_per_game']:+.2f}</td>"
            f"<td>{s['win_rate']*100:.0f}%</td><td>{s['draw_rate']*100:.0f}%</td>"
            f"<td>{s['first_move_win_rate']*100:.0f}%</td>"
            f"<td>{s['invalid_rate']*100:.1f}%</td>"
            f"<td>{s['avg_len']:.1f}</td><td>{s['avg_latency_s']:.1f}s</td></tr>")

    hh = ("<tr><th class='model'>model</th>"
          + "".join(f"<th>{m.split('-')[0]}</th>" for m in models) + "</tr>")
    for a in models:
        hh += f"<tr><td class='model'>{a}</td>"
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
    body = f"""  <section class="arena">
    <div class="arena-head"><h2>leaderboard</h2>
      <span class="arena-tag">{rep['num_games']} games · {deals} deals/pair · first-mover {fpw:.0f}%</span></div>
    <div class="note">Elo from a Bradley-Terry fit over head-to-head results (rated field mean 1500);
      a model with no wins or no losses has no finite rating and is shown as “—”. net/game is the average
      game payoff (+1 win / −1 loss / 0 draw). Seats swap within every pair, so first-mover advantage is balanced.</div>
    <table class="lb">
      <tr><th class='rk'>#</th><th class='model'>model</th><th>Elo</th><th>net/game</th><th>win%</th>
          <th>draw%</th><th>1st-move</th><th>invalid%</th><th>plies</th><th>think</th></tr>
      {rows}
    </table>
  </section>

  <section class="arena">
    <div class="arena-head"><h2>head-to-head</h2>
      <span class="arena-tag">row wins–losses vs column · d = draws</span></div>
    <table class="h2h">{hh}</table>
  </section>"""
    return _page(cfg, cfg["area"], f"{cfg['blurb']} · {rep['num_games']} games",
                 body, cfg["replay_verb"])


def render_dealer(rep: dict) -> str:
    cfg = GAMES[rep["game"]]
    pm = rep["per_model"]
    models = rep["models"]
    ranked = sorted(models, key=lambda m: pm[m]["profit"], reverse=True)
    profits = [pm[m]["profit"] for m in models]
    lo, hi = min(profits + [0]), max(profits + [0])

    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        pcls = "pos" if s["profit"] > 0 else ("neg" if s["profit"] < 0 else "")
        ptxt = f"{s['profit']:+.1f}"
        rows += (
            f"<tr><td class='rk'>{_medal(i)}</td><td class='model'>{m}</td>"
            f"{_bar_cell(s['profit'], lo, hi, ptxt)}"
            f"<td class='{pcls}'>{s['mean_per_hand']:+.3f}</td>"
            f"<td>{s['win_rate']*100:.0f}%</td><td>{s['push_rate']*100:.0f}%</td>"
            f"<td>{s['loss_rate']*100:.0f}%</td>"
            f"<td>{s['bust_rate']*100:.0f}%</td><td>{s['double_rate']*100:.0f}%</td>"
            f"<td>{s['natural_rate']*100:.0f}%</td>"
            f"<td>{s['invalid_rate']*100:.1f}%</td><td>{s['hands']}</td></tr>")

    body = f"""  <section class="arena">
    <div class="arena-head"><h2>leaderboard</h2>
      <span class="arena-tag">{rep['total_hands']} hands · field net {rep['field_profit']:+.1f}</span></div>
    <div class="note">Each model plays the same independent hands against the built-in dealer; the dealer
      holds an inherent house edge, so a negative field net is expected. profit = total chips won/lost
      (a doubled hand pays ±2). bust = player busted; double = chose to double down; natural = dealt a blackjack.</div>
    <table class="lb">
      <tr><th class='rk'>#</th><th class='model'>model</th><th>profit</th><th>mean/hand</th>
          <th>win%</th><th>push%</th><th>loss%</th><th>bust%</th><th>double%</th>
          <th>natural%</th><th>invalid%</th><th>hands</th></tr>
      {rows}
    </table>
  </section>"""
    return _page(cfg, cfg["area"], f"{cfg['blurb']} · {rep['total_hands']} hands total",
                 body, cfg["replay_verb"])


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
