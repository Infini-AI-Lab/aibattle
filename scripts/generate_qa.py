"""Generate reports/qa.html — a plain-language metrics Q&A page.

A public benchmark lives or dies on whether its numbers can be trusted, so this
page defines the non-obvious metrics the reports use, in plain terms (aggression,
correlation, equity, bluffing, the gear-shift, win-rate vs Elo). Things a chart's
own caption already explains are intentionally left off.

    PYTHONPATH=src:scripts python scripts/generate_qa.py
"""

from __future__ import annotations

import os

from report_theme import BASE_CSS

REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")
NAV_HEAD = '<link rel="stylesheet" href="nav.css"><script defer src="nav.js"></script>'

def render() -> str:
    css = BASE_CSS + """
      .ok { color:var(--pos); font-weight:700; }
      .warn { color:#b45309; font-weight:700; }
      .soft { color:var(--dim); font-weight:700; }
      .lead { font-size:14px; color:var(--dim); margin:2px 0 22px; }
      h2 { margin-top:34px; }
      .formula { background:var(--faint); border:1px solid var(--line); border-left:3px solid var(--red);
        padding:10px 14px; margin:10px 0; font-size:13px; }
      ul.tight li { margin:5px 0; } ul.tight { margin:8px 0; }
      .qa { margin:16px 0; font-size:13px; line-height:1.6; }
      .qa > b:first-child { display:block; color:var(--red); margin-bottom:3px; font-size:14px; }
      .qa .formula { margin:7px 0; display:inline-block; }
    """
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Battle Arena — Metrics Q&amp;A</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📐</text></svg>">
{NAV_HEAD}<style>{css}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/q-and-a<span class="cursor"></span></h1>
  <div class="lead">Plain-language definitions for the metrics used across the reports.</div>

  <div class="qa"><b>1. What does “aggression” mean?</b>
  How often a model attacks instead of going along with the action — used identically on every page:
  <span class="formula">aggression = (bet + raise + all-in) ÷ (bet + raise + all-in + call + check)</span>
  Folds are not counted. 0% = never bets/raises (pure caller); 100% = always bets/raises.</div>

  <div class="qa"><b>2. What is “corr with win%” (correlation)?</b>
  Shown on some tables as e.g. <code>corr with win%: +0.61</code> — the Pearson correlation between a
  metric and match win rate across the models, from −1 to +1: <b>+0.9</b> ≈ move together almost
  perfectly, <b>+0.6</b> = clearly related, <b>0</b> = unrelated, <b>negative</b> = move in opposite
  directions. With only ~11 models, read values above ~+0.58 (or below −0.58) as a real signal and
  smaller ones as weak.</div>

  <div class="qa"><b>3. What is “equity”?</b>
  A hand's chance of winning if all the cards were dealt out, estimated by simulation (deal the opponent a
  random hand, finish the board, repeat thousands of times). Caveat: the opponent's real cards are never in
  the logs, so equity is computed <b>vs a random hand</b>, not their actual range — a rough, range-free
  proxy. That is why the equity-based “Decision quality” block is labelled <b>experimental</b>.</div>

  <div class="qa"><b>4. How is a “bluff” detected?</b>
  A bet/raise/all-in made with a weak hand — specifically when the model's equity (vs random) is below 40%,
  so it is betting a hand that is probably behind, to push the opponent off. <b>bluff success</b> = of those
  bluffs, how often the opponent actually folds. (Pure bluffs and semi-bluffs are lumped together, and 40%
  is a chosen cut-off.)</div>

  <div class="qa"><b>5. What is the “gear-shift” (ahead vs behind)?</b>
  How a model changes when it is losing on chips. Strong players, when behind, <b>raise more</b> and
  <b>fold less</b> (they fight for pots); weak players play the same whether ahead or behind. Each cell shows
  the value when ahead → when behind.</div>

  <div class="qa"><b>6. Win rate vs Elo — why are they different?</b>
  <b>Win rate</b> is just the share of matches won. <b>Elo</b> is opponent-adjusted — beating strong
  opponents is worth more than beating weak ones, and it accounts for who each model actually played. A model
  can have a decent raw win rate but a lower Elo if its wins came against weaker opposition, so the
  leaderboards are ordered by Elo as the fairer ranking.</div>

  <div class="qa"><b>7. How do you serve the open-weight models?</b>
  All open-weight models are served via <b>Fireworks AI</b> (serverless inference) through one
  OpenAI-compatible client; the closed models (Claude, GPT-5.x) run on their own provider APIs.</div>

  <div class="qa"><b>8. How is the overall ranking and Elo calculated?</b>
  The overview leaderboard shows two cross-game summaries over six head-to-head games: Connect Four,
  Gomoku, Hold'em 1-Hand, Hold'em Match, Colonel Blotto and Leduc Holdem. On that table, click a
  <b>score header</b> to sort by it, or the <b>ⓘ</b> for how each is computed. Coverage = games a model
  has entered (treat low coverage as provisional).</div>

</div></body></html>"""


def main():
    html = render()
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, "qa.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
