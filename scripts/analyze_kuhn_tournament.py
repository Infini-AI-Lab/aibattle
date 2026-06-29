"""Analyze the Kuhn Poker ("poker light") round-robin tournament.

Reads runs/kuhn_poker/kuhn_data.json and, because Kuhn has a fully solved
Nash equilibrium, scores models on *poker fundamentals* rather than just chips
(30 hands/pair is far too few for chips to be reliable — that number is
directional only). Three lenses:

  1. Leaderboard       — net chips/hand, win rate, invalid/truncated rates.
  2. GTO fundamentals  — the two unambiguous blunders Kuhn allows:
        · folding the King (the nuts) to a bet — you fold a hand that always
          wins the showdown;
        · calling a bet with the Jack (the worst card) — you can never win the
          showdown (the opponent must hold Q or K), so it is pure spew.
     Neither depends on equilibrium mixing, so they are clean skill signals.
  3. Betting style     — bet frequency by card when first to act. The solved
     equilibrium is *polarized*: bet the King ~always, bluff the Jack ~1/3 of
     the time, and (almost) never bet the Queen. How close a model gets to that
     polarized shape says more in 30 hands than its chip count does.

Writes a Chart.js HTML report to runs/kuhn_poker/kuhn_report.html and
reports/kuhn_tournament_report.html, plus the raw numbers to
reports/kuhn_tournament_analysis.json. Styling matches the other tournament
reports.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from model_names import strip_coached, display_name, model_cell
from report_tokens import token_cost_cells, TOKEN_HEADERS
from elo_util import bradley_terry, bootstrap_elo, wld_from_records
from report_theme import BASE_CSS, CHART_SETUP
from report_legends import legend as _legend

# Coached is now the canonical (and only) run set; data lives in per-game folders.
DATA = "runs/kuhn_poker/kuhn_data.json"
OUT_HTML = "runs/kuhn_poker/kuhn_report.html"
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# this page includes those two files in <head> via NAV_HEAD and the bar is
# injected by JS, so the nav markup lives in one place.
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=30"></script>'

_STYLE = BASE_CSS + """
  a { text-decoration:none; } a:hover { text-decoration:underline; }
  .good { color:var(--pos); } .warn { color:#b45309; } .bad { color:var(--neg); }
  .rules { background:var(--faint); border:1px solid var(--line);
    padding:18px 20px; margin:18px 0; font-size:13.5px; color:var(--fg); line-height:1.55; }
  .rules h3 { margin:0 0 8px; font-size:15px; }
  .rules h3::before { content:""; }
  .rules ul { margin:6px 0 0; padding-left:20px; } .rules li { margin:3px 0; }
  .rules .card { display:inline-block; min-width:16px; padding:1px 6px; margin:0 1px;
    background:var(--fg); color:var(--bg); font-weight:700; font-size:12px;
    text-align:center; vertical-align:middle; line-height:1.4; }
  .rules code { background:#fff; border:1px solid var(--line); padding:1px 6px; color:var(--red); font-size:12px; }
  .rules .seq { color:var(--dim); font-size:12.5px; margin-top:10px; }
"""

# Canonical Nash equilibrium bet frequencies when first to act (α = 1/3).
GTO_BET = {"K": 1.0, "Q": 0.0, "J": 1 / 3}


def _acts(obs: dict) -> list:
    return [h["action"] for h in obs.get("history", [])]


def analyze(data: dict) -> dict:
    models = data["models"]
    hands = defaultdict(int); wins = defaultdict(int); net = defaultdict(float)
    decisions = defaultdict(int); invalid = defaultdict(int); trunc = defaultdict(int)
    tokens = defaultdict(int); tok_n = defaultdict(int)

    # first-to-act betting: how many times each model could bet, and did, per card
    bet_chance = {m: defaultdict(int) for m in models}
    bet_made = {m: defaultdict(int) for m in models}
    # facing a bet: blunder tracking
    k_fold_chance = defaultdict(int); k_fold = defaultdict(int)   # fold the nuts
    j_call_chance = defaultdict(int); j_call = defaultdict(int)   # call with worst hand
    # net chips per hand, row model vs column model
    h2h_net = {a: {b: 0.0 for b in models} for a in models}
    h2h_hands = {a: {b: 0 for b in models} for a in models}
    h2h_wins = {a: {b: 0 for b in models} for a in models}  # a's hand wins vs b
    elo_records = []  # per-hand (a, b, result) for the Elo bootstrap

    for g in data["pairs"]:
        for e in g["episodes"]:
            seat = e["seat_assignment"]
            a, b = seat["player_0"], seat["player_1"]
            wname = e.get("winner_name")
            for p in ("player_0", "player_1"):
                nm = seat[p]; opp = seat["player_1" if p == "player_0" else "player_0"]
                hands[nm] += 1
                net[nm] += e["returns"][p]
                h2h_net[nm][opp] += e["returns"][p]
                h2h_hands[nm][opp] += 1
            if wname:
                wins[wname] += 1
                h2h_wins[wname][b if wname == a else a] += 1
                elo_records.append((a, b, 1 if wname == a else -1))
            else:
                elo_records.append((a, b, 0))

            for s in e.get("steps", []):
                obs = s.get("observation", {})
                nm = s.get("agent_name") or seat.get(s.get("player"))
                card = (obs.get("private") or {}).get("card")
                legal = obs.get("legal_actions", [])
                action = s.get("selected_action")
                decisions[nm] += 1
                if s.get("invalid"):
                    invalid[nm] += 1
                meta = (s.get("response") or {}).get("metadata", {})
                if meta.get("truncated"):
                    trunc[nm] += 1
                ct = meta.get("completion_tokens")
                if isinstance(ct, (int, float)):
                    tokens[nm] += ct; tok_n[nm] += 1

                if card is None:
                    continue
                if set(legal) == {"check", "bet"}:        # first to act this street
                    bet_chance[nm][card] += 1
                    if action == "bet":
                        bet_made[nm][card] += 1
                elif set(legal) == {"call", "fold"}:       # facing a bet
                    if card == "K":
                        k_fold_chance[nm] += 1
                        if action == "fold":
                            k_fold[nm] += 1
                    elif card == "J":
                        j_call_chance[nm] += 1
                        if action == "call":
                            j_call[nm] += 1

    # Opponent-adjusted Elo over per-hand W/L/D (kept as a reference column; the
    # leaderboard still ranks by fewest blunders, the cleaner skill signal here).
    wld = {a: {b: (h2h_wins[a][b], h2h_wins[b][a],
                   h2h_hands[a][b] - h2h_wins[a][b] - h2h_wins[b][a])
               for b in models if b != a} for a in models}
    _, elo = bradley_terry(models, wld)
    elo_ci = bootstrap_elo(models, elo_records, lambda s: wld_from_records(models, s))

    rows = []
    for m in models:
        h = hands[m] or 1; d = decisions[m] or 1
        def betpct(card):
            c = bet_chance[m][card]
            return round(bet_made[m][card] / c, 3) if c else None
        blunders = k_fold[m] + j_call[m]
        blunder_spots = k_fold_chance[m] + j_call_chance[m]
        rows.append({
            "model": m, "hands": hands[m], "decisions": decisions[m],
            "elo": elo[m], "elo_sd": elo_ci[m]["sd"],
            "win_rate": round(wins[m] / h, 3),
            "net_per_hand": round(net[m] / h, 3),
            "invalid_rate": round(invalid[m] / d, 4),
            "truncated_rate": round(trunc[m] / d, 4),
            "avg_tokens": round(tokens[m] / (tok_n[m] or 1)),
            # fundamentals
            "k_folds": k_fold[m], "k_fold_chances": k_fold_chance[m],
            "j_calls": j_call[m], "j_call_chances": j_call_chance[m],
            "blunders": blunders, "blunder_spots": blunder_spots,
            "blunder_rate": round(blunders / blunder_spots, 3) if blunder_spots else 0.0,
            # style
            "bet_K": betpct("K"), "bet_Q": betpct("Q"), "bet_J": betpct("J"),
        })
    # rank by fewest blunders, then chips (chips alone is too noisy at 30 hands)
    rows.sort(key=lambda r: (r["blunder_rate"], -r["net_per_hand"]))

    return {"models": models, "episodes_per_pair": data.get("episodes_per_pair"),
            "gto_bet": GTO_BET, "leaderboard": rows, "elo": elo,
            "h2h_net": h2h_net, "h2h_hands": h2h_hands}


def _cls(v, good, bad, invert=False):
    """Colour class: green if good-side, red if bad-side."""
    if v is None:
        return ""
    if invert:
        return "good" if v <= good else ("bad" if v >= bad else "warn")
    return "good" if v >= good else ("bad" if v <= bad else "warn")


def render_html(rep: dict) -> str:
    models = rep["models"]; lb = rep["leaderboard"]
    labels = [display_name(r["model"]) for r in lb]   # chart axis: official names

    def pct(v):
        return "—" if v is None else f"{v*100:.0f}%"

    # leaderboard
    lb_rows = ""
    for i, r in enumerate(lb, 1):
        ncls = "pos" if r["net_per_hand"] >= 0 else "neg"
        if r.get("elo") is None:
            elo_disp = "—"
        elif r.get("elo_sd") is not None:
            elo_disp = f"{r['elo']}<div class='small'>±{r['elo_sd']:.0f}</div>"
        else:
            elo_disp = str(r["elo"])
        lb_rows += (
            f"<tr><td>{i}</td><td class='model'>{model_cell(r['model'])}</td>"
            f"<td><b>{elo_disp}</b></td>"
            f"<td class='{ncls}'>{r['net_per_hand']:+.3f}</td>"
            f"<td>{r['win_rate']*100:.0f}%</td>"
            f"<td>{r['hands']}</td><td>{r['decisions']}</td>"
            f"<td>{r['invalid_rate']*100:.1f}%</td>"
            f"<td>{r['truncated_rate']*100:.1f}%</td>"
            f"{token_cost_cells(r['model'], r['avg_tokens'])}</tr>")

    # fundamentals (blunders)
    fund_rows = ""
    for i, r in enumerate(lb, 1):
        bcls = _cls(r["blunder_rate"], 0.0, 0.25, invert=True)
        kf = f"{r['k_folds']}/{r['k_fold_chances']}" if r["k_fold_chances"] else "0/0"
        jc = f"{r['j_calls']}/{r['j_call_chances']}" if r["j_call_chances"] else "0/0"
        fund_rows += (
            f"<tr><td>{i}</td><td class='model'>{model_cell(r['model'])}</td>"
            f"<td>{kf}</td><td>{jc}</td>"
            f"<td class='{bcls}'>{r['blunders']}/{r['blunder_spots']}"
            f" ({r['blunder_rate']*100:.0f}%)</td></tr>")

    # betting style table
    style_rows = ""
    for r in lb:
        bk = _cls(r["bet_K"], 0.8, 0.4)                      # want high
        # bluff J: closeness to ~1/3; flag 0% (never bluffs) and 100% (over-bluff)
        bj = "warn"
        if r["bet_J"] is not None:
            bj = "good" if 0.15 <= r["bet_J"] <= 0.6 else "bad"
        bq = _cls(r["bet_Q"], 0.0, 0.5, invert=True)          # want low
        style_rows += (
            f"<tr><td class='model'>{model_cell(r['model'])}</td>"
            f"<td class='{bk}'>{pct(r['bet_K'])}</td>"
            f"<td class='{bq}'>{pct(r['bet_Q'])}</td>"
            f"<td class='{bj}'>{pct(r['bet_J'])}</td></tr>")

    # bet-by-card grouped bar chart data
    bet_k = [round((r["bet_K"] or 0) * 100, 1) for r in lb]
    bet_q = [round((r["bet_Q"] or 0) * 100, 1) for r in lb]
    bet_j = [round((r["bet_J"] or 0) * 100, 1) for r in lb]

    # head-to-head net chips/hand grid
    head = "".join(f"<th>{display_name(m)}</th>" for m in models)
    grid = ""
    for a in models:
        cells = ""
        for b in models:
            if a == b:
                cells += "<td class='diag'>—</td>"
            else:
                n = rep["h2h_hands"][a][b]
                v = rep["h2h_net"][a][b] / n if n else 0.0
                cls = "pos" if v >= 0 else "neg"
                cells += f"<td class='{cls}'>{v:+.2f}</td>"
        grid += f"<tr><td class='model'>{model_cell(a)}</td>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Kuhn Poker</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🃏</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}<style>{_STYLE}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/kuhn<span class="cursor"></span></h1>
  <div class="sub">🃏 Kuhn Poker · {len(models)} models · round-robin · {rep['episodes_per_pair']} seat-swapped hands/pair · deck {{J,Q,K}}, ~2 decisions/hand</div>
  <a class="replaybtn" href="kuhn_replay.html?cacheBust=19">🎬 Watch featured replays →</a>

  <div class="rules">
    <h3>How Kuhn Poker works</h3>
    A minimal 2-player poker on a 3-card deck — <span class="card">J</span><span class="card">Q</span><span class="card">K</span>
    (ranked J &lt; Q &lt; K). One hand plays out like this:
    <ul>
      <li>Both players <b>ante 1 chip</b> (pot starts at 2). Each is dealt one private card; the third is unused.</li>
      <li><b>Player 0 acts first</b> and may <code>check</code> or <code>bet</code> (1 chip).</li>
      <li>Facing a <code>bet</code>, the opponent may <code>call</code> (match it) or <code>fold</code> (forfeit the pot).</li>
      <li>Facing a <code>check</code>, the opponent may <code>check</code> back (go to showdown) or <code>bet</code>.</li>
      <li>At <b>showdown the higher card wins</b> the pot; a fold hands the pot to the other player. Zero-sum: the winner gains exactly what the loser put in.</li>
    </ul>
    <div class="seq">Every hand is one of five lines: <code>check-check</code>, <code>check-bet-fold</code>,
    <code>check-bet-call</code>, <code>bet-fold</code>, or <code>bet-call</code> — so a player makes ~1–2 decisions per hand.</div>
    <div class="seq"><b>What the model sees each turn:</b> its own private card, the betting so far, the
    pot, and its legal actions — never the opponent's card.</div>
  </div>

  <div class="callout">Kuhn Poker is <a href="https://en.wikipedia.org/wiki/Kuhn_poker" target="_blank" rel="noopener"><b>fully solved</b></a>, so we judge play against the Nash
  equilibrium rather than chips alone. At only {rep['episodes_per_pair']} hands/pair the chip totals
  are <b>high-variance and directional</b> — the <i>fundamentals</i> and <i>betting-style</i> sections
  below are the real skill signal. The leaderboard is ranked by fewest blunders, then chips;
  the <b>Elo</b> column (opponent-adjusted Bradley-Terry rating over per-hand results, field mean
  1500) is shown for reference, with ±1 bootstrap SD — note how wide those error bars are at this
  sample size, which is exactly why the ranking leans on blunders instead.</div>

  <h2>Bet frequency by card <span class="note">(when first to act)</span></h2>
  <div class="note">Equilibrium is <b>polarized</b>: bet the King (green) ~100%, bluff the Jack (blue) ~33%,
  and almost never bet the Queen (red). The closer a model matches that shape, the more GTO it plays.</div>
  <canvas id="bet"></canvas>

  <h2>GTO fundamentals <span class="note">(unambiguous blunders)</span></h2>
  <div class="note"><b>Fold K to a bet</b> = folding the nuts (always wins the showdown).
  <b>Call a bet with J</b> = calling with the worst card (can never win the showdown). Both are
  pure mistakes independent of mixing. Shown as occurrences / opportunities.</div>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>fold K vs bet</th>
        <th>call J vs bet</th><th>total blunders</th></tr>
    {fund_rows}
  </table>
  {_legend('kuhn')}

  <h2>Betting style <span class="note">(bet % by card, first to act · GTO: K 100% / Q 0% / J ~33%)</span></h2>
  <table>
    <tr><th class='model'>model</th><th>bet K (value)</th><th>bet Q (trap)</th><th>bet J (bluff)</th></tr>
    {style_rows}
  </table>

  <h2>Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>Elo</th><th>net/hand</th><th>win%</th>
        <th>hands</th><th>decisions</th><th>invalid%</th><th>trunc%</th>{TOKEN_HEADERS}</tr>
    {lb_rows}
  </table>

  <h2>Head-to-head <span class="note">(row's net chips/hand vs column)</span></h2>
  <table><tr><th class='model'></th>{head}</tr>{grid}</table>

  <script>
  const labels = {json.dumps(labels)};
  const gto = {{K:{GTO_BET['K']*100}, Q:{GTO_BET['Q']*100}, J:{round(GTO_BET['J']*100,1)}}};
  new Chart(document.getElementById('bet'), {{
    type:'bar',
    data:{{labels:labels,datasets:[
      {{label:'bet K %',data:{json.dumps(bet_k)},backgroundColor:'#4ade80'}},
      {{label:'bet Q %',data:{json.dumps(bet_q)},backgroundColor:'#f87171'}},
      {{label:'bet J %',data:{json.dumps(bet_j)},backgroundColor:'#60a5fa'}}
    ]}},
    options:{{plugins:{{legend:{{labels:{{color:'#1c1c1c'}}}}}},
      scales:{{y:{{beginAtZero:true,max:100,grid:{{color:'#e7e2d8'}},
        ticks:{{color:'#1c1c1c',callback:v=>v+'%'}}}},
        x:{{grid:{{color:'#e7e2d8'}},ticks:{{color:'#1c1c1c'}}}}}}}}
  }});
  </script>
</div></body></html>"""


def main():
    data = strip_coached(json.load(open(DATA)))
    rep = analyze(data)
    html = render_html(rep)
    os.makedirs(REPORT_DIR, exist_ok=True)
    for path in (OUT_HTML, os.path.join(REPORT_DIR, "kuhn_tournament_report.html")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    json.dump(rep, open(os.path.join(REPORT_DIR, "kuhn_tournament_analysis.json"), "w"),
              indent=2)
    print(f"Wrote {OUT_HTML} and {REPORT_DIR}/kuhn_tournament_report.html\n")
    print(f"=== Kuhn Poker ({rep['episodes_per_pair']} hands/pair) — ranked by fewest blunders ===")
    print(f"{'model':<18} net/hand  blunders   betK  betQ  betJ  tok/dec")
    for r in rep["leaderboard"]:
        def p(v):
            return " — " if v is None else f"{v*100:>3.0f}%"
        print(f"{r['model']:<18} {r['net_per_hand']:>+7.3f}  "
              f"{r['blunders']:>2}/{r['blunder_spots']:<3} ({r['blunder_rate']*100:>3.0f}%)  "
              f"{p(r['bet_K'])} {p(r['bet_Q'])} {p(r['bet_J'])}  {r['avg_tokens']:>5,}")


if __name__ == "__main__":
    main()
