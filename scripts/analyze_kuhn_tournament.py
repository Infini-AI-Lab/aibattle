"""Analyze the Kuhn Poker ("poker light") round-robin tournament.

Reads runs/kuhn_tournament/kuhn_data.json and, because Kuhn has a fully solved
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

Writes a Chart.js HTML report to runs/kuhn_tournament/kuhn_report.html and
reports/kuhn_tournament_report.html, plus the raw numbers to
reports/kuhn_tournament_analysis.json. Styling matches the other tournament
reports.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

DATA = "runs/kuhn_tournament/kuhn_data.json"
OUT_HTML = "runs/kuhn_tournament/kuhn_report.html"
REPORT_DIR = "reports"

_STYLE = """
  .navbar { position:sticky; top:0; z-index:50; display:flex; align-items:center;
    flex-wrap:wrap; gap:6px 18px; padding:0 22px; min-height:52px;
    background:rgba(12,14,20,.92); backdrop-filter:blur(8px);
    border-bottom:1px solid #232838; }
  .navbar .brand { font-weight:700; color:#cdd6f4; text-decoration:none; font-size:15px; margin-right:10px; }
  .navbar a.nav { color:#9aa3b5; text-decoration:none; font-size:13px; padding:16px 2px; border-bottom:2px solid transparent; }
  .navbar a.nav:hover { color:#e6e6e6; }
  .navbar a.nav.active { color:#fff; border-bottom-color:#60a5fa; }
  body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }
  .wrap { max-width:1080px; margin:0 auto; padding:28px 22px 80px; }
  h1 { font-size:25px; } h2 { font-size:19px; margin-top:40px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }
  .sub { color:#8b93a7; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin-top:10px; }
  th,td { padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }
  th { color:#9aa3b5; } td.model,th.model { text-align:left; font-weight:600; color:#cdd6f4; }
  .pos { color:#4ade80; } .neg { color:#f87171; } .diag { color:#3a3f4b; }
  .good { color:#4ade80; } .warn { color:#fbbf24; } .bad { color:#f87171; }
  .note { color:#8b93a7; font-size:12px; margin:6px 0; }
  .callout { background:#171a23; border:1px solid #232838; border-left:3px solid #60a5fa;
    border-radius:8px; padding:12px 16px; margin:14px 0; font-size:13px; color:#c7cedd; }
  a { color:#60a5fa; text-decoration:none; } a:hover { text-decoration:underline; }
  .rules { background:#141821; border:1px solid #232838; border-radius:12px;
    padding:18px 20px; margin:18px 0; font-size:13.5px; color:#c7cedd; line-height:1.55; }
  .rules h3 { margin:0 0 8px; font-size:15px; color:#cdd6f4; }
  .rules ul { margin:6px 0 0; padding-left:20px; } .rules li { margin:3px 0; }
  .rules .card { display:inline-block; min-width:16px; padding:1px 6px; margin:0 1px;
    border-radius:4px; background:#e6e6e6; color:#0f1117; font-weight:700; font-size:12px;
    text-align:center; vertical-align:middle; line-height:1.4; }
  .rules code { background:#1e2430; padding:1px 6px; border-radius:4px; color:#9fc7ff; font-size:12px; }
  .rules .seq { color:#8b93a7; font-size:12.5px; margin-top:10px; }
  canvas { max-height:320px; margin-top:10px; }
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
            "gto_bet": GTO_BET, "leaderboard": rows,
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
    labels = [r["model"] for r in lb]

    def pct(v):
        return "—" if v is None else f"{v*100:.0f}%"

    # leaderboard
    lb_rows = ""
    for i, r in enumerate(lb, 1):
        ncls = "pos" if r["net_per_hand"] >= 0 else "neg"
        lb_rows += (
            f"<tr><td>{i}</td><td class='model'>{r['model']}</td>"
            f"<td class='{ncls}'>{r['net_per_hand']:+.3f}</td>"
            f"<td>{r['win_rate']*100:.0f}%</td>"
            f"<td>{r['hands']}</td><td>{r['decisions']}</td>"
            f"<td>{r['invalid_rate']*100:.1f}%</td>"
            f"<td>{r['truncated_rate']*100:.1f}%</td>"
            f"<td>{r['avg_tokens']:,}</td></tr>")

    # fundamentals (blunders)
    fund_rows = ""
    for i, r in enumerate(lb, 1):
        bcls = _cls(r["blunder_rate"], 0.0, 0.25, invert=True)
        kf = f"{r['k_folds']}/{r['k_fold_chances']}" if r["k_fold_chances"] else "0/0"
        jc = f"{r['j_calls']}/{r['j_call_chances']}" if r["j_call_chances"] else "0/0"
        fund_rows += (
            f"<tr><td>{i}</td><td class='model'>{r['model']}</td>"
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
            f"<tr><td class='model'>{r['model']}</td>"
            f"<td class='{bk}'>{pct(r['bet_K'])}</td>"
            f"<td class='{bq}'>{pct(r['bet_Q'])}</td>"
            f"<td class='{bj}'>{pct(r['bet_J'])}</td></tr>")

    # bet-by-card grouped bar chart data
    bet_k = [round((r["bet_K"] or 0) * 100, 1) for r in lb]
    bet_q = [round((r["bet_Q"] or 0) * 100, 1) for r in lb]
    bet_j = [round((r["bet_J"] or 0) * 100, 1) for r in lb]

    # head-to-head net chips/hand grid
    head = "".join(f"<th>{m.split('-')[0]}</th>" for m in models)
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
        grid += f"<tr><td class='model'>{a}</td>{cells}</tr>"

    nav = ("<nav class='navbar'><a class='brand' href='index.html'>🎲 AI Battle Arena</a>"
           "<a class='nav' href='index.html'>Overview</a>"
           "<a class='nav' href='connect4_report.html'>🔴 Connect Four</a>"
           "<a class='nav' href='gomoku_report.html'>⚫ Gomoku</a>"
           "<a class='nav' href='holdem_tournament_report.html'>🃏 Hold'em</a>"
           "<a class='nav active' href='kuhn_tournament_report.html'>🃏 Kuhn</a></nav>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Kuhn Poker</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🃏</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>{_STYLE}</style></head>
<body>{nav}<div class="wrap">
  <h1>🃏 AI Battle Arena — Kuhn Poker (poker light)</h1>
  <div class="sub">5 models · round-robin · {rep['episodes_per_pair']} seat-swapped hands/pair · deck {{J,Q,K}}, ~2 decisions/hand</div>

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
  </div>

  <div class="callout">Kuhn Poker is <a href="https://en.wikipedia.org/wiki/Kuhn_poker" target="_blank" rel="noopener"><b>fully solved</b></a>, so we judge play against the Nash
  equilibrium rather than chips alone. At only {rep['episodes_per_pair']} hands/pair the chip totals
  are <b>high-variance and directional</b> — the <i>fundamentals</i> and <i>betting-style</i> sections
  below are the real skill signal. The leaderboard is ranked by fewest blunders, then chips.</div>

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

  <h2>Betting style <span class="note">(bet % by card, first to act · GTO: K 100% / Q 0% / J ~33%)</span></h2>
  <table>
    <tr><th class='model'>model</th><th>bet K (value)</th><th>bet Q (trap)</th><th>bet J (bluff)</th></tr>
    {style_rows}
  </table>

  <h2>Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>net/hand</th><th>win%</th>
        <th>hands</th><th>decisions</th><th>invalid%</th><th>trunc%</th><th>avg tokens/dec</th></tr>
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
    options:{{plugins:{{legend:{{labels:{{color:'#9aa3b5'}}}}}},
      scales:{{y:{{beginAtZero:true,max:100,grid:{{color:'#20242e'}},
        ticks:{{color:'#9aa3b5',callback:v=>v+'%'}}}},
        x:{{grid:{{color:'#20242e'}},ticks:{{color:'#9aa3b5'}}}}}}}}
  }});
  </script>
</div></body></html>"""


def main():
    data = json.load(open(DATA))
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
