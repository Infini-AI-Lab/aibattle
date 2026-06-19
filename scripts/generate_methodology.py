"""Generate reports/methodology.html — the Methodology & Limitations page.

A public benchmark lives or dies on whether its numbers can be trusted, so this
page (a) defines every metric and (b) discloses the limitations up front. The
statistical-significance table is computed from the per-game analysis JSONs
(Elo + bootstrap SD), so the "is the top of this board actually decided?" verdict
stays honest as the data changes — rather than being hand-asserted.

Run after the analyzers (they write reports/*_analysis.json):
    PYTHONPATH=src:scripts python scripts/generate_methodology.py
"""

from __future__ import annotations

import json
import os

from model_names import display_name
from report_theme import BASE_CSS

REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=27"></script>'

# The six games that feed the cross-game ranking (must match
# analyze_board_tournament.ARENA_GAMES), with a friendly title and the sample-size
# field/label to read from each game's analysis JSON.
ARENA = [
    ("connect4_analysis.json", "🔴 Connect Four", "num_games", "games"),
    ("gomoku_analysis.json", "⚫ Gomoku-Lite", "num_games", "games"),
    ("holdem_tournament_analysis.json", "🃏 Hold'em 1-Hand", "num_games", "tables"),
    ("match_tournament_analysis.json", "🃏 Hold'em Match", "episodes_per_pair", "matches/pair"),
    ("repeated_colonel_blotto_analysis.json", "⚔️ Colonel Blotto", "num_games", "games"),
    ("leduc_poker_analysis.json", "🎴 Leduc Holdem", "num_games", "games"),
]


def _ratings_with_sd(d: dict):
    """Return [(model, elo, sd), ...] ordered by Elo desc, dropping unrated.

    Handles both shapes: a top-level ``elo_ci`` dict (board / new-games) or an
    ``elo_sd`` carried on each leaderboard row (match / kuhn)."""
    elo = d.get("elo") or {}
    sd = {}
    if isinstance(d.get("elo_ci"), dict):
        sd = {m: (d["elo_ci"].get(m) or {}).get("sd") for m in elo}
    elif isinstance(d.get("leaderboard"), list):
        sd = {r["model"]: r.get("elo_sd") for r in d["leaderboard"] if "model" in r}
    rated = [(m, elo[m], sd.get(m)) for m in elo if elo[m] is not None]
    rated.sort(key=lambda t: t[1], reverse=True)
    return rated


def _significance(rated):
    """Given Elo-ranked (model, elo, sd), judge whether the TOP is decided.

    Two models are 'separated' when their ±1-SD intervals do not overlap. The
    top is 'decided' if #1 is separated from #2; otherwise we report how many
    leaders sit in a statistical tie with #1 (their intervals overlap #1's)."""
    if len(rated) < 2:
        return {"verdict": "n/a", "tie": 1}
    (m1, e1, s1), (m2, e2, s2) = rated[0], rated[1]
    if s1 is None or s2 is None:
        return {"verdict": "no error bars", "tie": 1}
    lo1 = e1 - s1
    # how many leaders overlap #1's lower bound
    tie = 1
    for m, e, s in rated[1:]:
        if s is None:
            break
        if e + s >= lo1:           # interval overlaps #1's
            tie += 1
        else:
            break
    decided = (e2 + s2) < lo1
    return {"verdict": "decided" if decided else "tie", "tie": tie,
            "top": display_name(m1), "runner": display_name(m2),
            "gap": round(e1 - e2), "sd": round((s1 + s2) / 2)}


def build_rows():
    rows, notes = [], []
    for fname, title, size_key, size_label in ARENA:
        path = os.path.join(REPORT_DIR, fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        n = d.get(size_key, "?")
        rated = _ratings_with_sd(d)
        sig = _significance(rated)
        rows.append({"title": title, "n": n, "n_label": size_label,
                     "models": len(rated), **sig})
    return rows


def _verdict_cell(r):
    if r["verdict"] == "decided":
        return (f"<span class='ok'>#1 decided</span><div class='small'>{r['top']} "
                f"leads #2 by {r['gap']} (±{r['sd']})</div>")
    if r["verdict"] == "tie":
        n = r["tie"]
        if n >= 3:
            return (f"<span class='warn'>{n}-way tie</span>"
                    f"<div class='small'>top {n} statistically inseparable — order is noise</div>")
        return (f"<span class='soft'>#1–#2 photo finish</span>"
                f"<div class='small'>gold/silver within ±1 SD; field below is separated</div>")
    return f"<span class='small'>{r['verdict']}</span>"


def render(rows) -> str:
    trows = ""
    for r in rows:
        trows += (f"<tr><td class='model'>{r['title']}</td>"
                  f"<td>{r['n']} {r['n_label']}</td><td>{r['models']}</td>"
                  f"<td>{_verdict_cell(r)}</td></tr>")
    big = [r for r in rows if r["verdict"] == "tie" and r["tie"] >= 3]
    big_games = ", ".join(r["title"].split(" ", 1)[1] for r in big)
    css = BASE_CSS + """
      .ok { color:var(--pos); font-weight:700; }
      .warn { color:#b45309; font-weight:700; }
      .soft { color:var(--dim); font-weight:700; }
      .lead { font-size:14px; color:var(--dim); margin:2px 0 22px; }
      h2 { margin-top:34px; }
      .formula { background:var(--faint); border:1px solid var(--line); border-left:3px solid var(--red);
        padding:10px 14px; margin:10px 0; font-size:13px; }
      ul.tight li { margin:5px 0; } ul.tight { margin:8px 0; }
    """
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Battle Arena — Methodology &amp; Limitations</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📐</text></svg>">
{NAV_HEAD}<style>{css}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/methodology<span class="cursor"></span></h1>
  <div class="lead">How the numbers are computed — and where they should and shouldn't be trusted.</div>

  <h2>The two arenas</h2>
  <p class="note"><b>Model Arena</b> runs every model through one identical generic pipeline (same prompts,
  same harness, same scoring) so results are comparable. <b>Harness Arena</b> (coming soon) opens the same
  games to any model + any scaffolding. Every game is round-robin and seat-swapped, so first-mover or
  position advantages cancel out across a pair.</p>

  <h2>Per-game scores</h2>
  <p class="note">Each game is scored in its own natural unit, then rated by an opponent-adjusted Elo so
  models that faced different opponents stay comparable:</p>
  <ul class="tight note">
    <li><b>Board games &amp; Blotto</b> — Bradley-Terry / <b>Elo</b> fit from all head-to-head
      win/loss/draw results (field mean 1500). Board games also score <b>tactical accuracy</b>
      (taking immediate wins, blocking immediate threats).</li>
    <li><b>Hold'em 1-Hand &amp; Leduc</b> — <b>chip-weighted Elo</b>: the Bradley-Terry fit is fed gross
      chips won, so winning big pots counts more than winning many tiny ones. Raw bb/100 is shown too.</li>
    <li><b>Hold'em Match</b> — <b>win-based Elo</b>: a match is win-or-lose (chips don't carry past the
      match), so the rating is fit over match wins/losses only.</li>
    <li><b>Hold'em Table</b> — 5-handed ring game, ranked by <b>average finishing place</b> (no pairwise
      Elo in multi-way play).</li>
    <li><b>Blackjack</b> — vs the house dealer (no opponent), scored by <b>mean chips/hand</b>.</li>
  </ul>
  <div class="formula"><b>Error bars (± 1 SD)</b> come from a <b>bootstrap</b>: resample the games with
  replacement 300×, refit the Elo each time, and take the standard deviation. Two models whose ±1-SD
  intervals overlap are a <b>statistical tie</b> — their ordering is within noise.</div>

  <h2>Cross-game ranking</h2>
  <p class="note">Six head-to-head games feed the headline: Connect Four, Gomoku, Hold'em 1-Hand, Hold'em
  Match, Colonel Blotto and Leduc Holdem. (Blackjack is <b>excluded</b> — no opponent Elo and luck-dominated;
  Othello, Kuhn and the Table mode have their own pages but don't feed the headline.) Two summaries:</p>
  <div class="formula"><b>Arena Rank Score</b> — ordinal. In each game a finish becomes a 0–1 score, evenly
  spaced via <code>(N−1−rank)/(N−1)</code>; the score is the mean across games played, ×100. It ignores
  <i>margin</i> — winning by a mile or a hair both score 1.0.</div>
  <div class="formula"><b>Arena Elo</b> — margin-aware. Each game's Elo is standardized within its field,
  <code>z = (rating − mean)/SD</code>; a model's z is averaged across games and rescaled to
  <code>1500 + 150·z</code>. It rewards <i>how much</i> you win by, not just finishing order.</div>
  <p class="note"><b>Coverage</b> is how many of the six games a model entered; treat a low-coverage row as
  provisional (it's averaged over fewer games).</p>

  <h2>⚠ Statistical significance <span class="note">(is each board actually decided?)</span></h2>
  <p class="note">Sample size and bootstrap error bars per core game. Where the top two models' ±1-SD
  intervals overlap, the leader is <b>not</b> statistically distinguishable from the runner-up — the
  ordering there is within noise and should be read as a tie.</p>
  <table>
    <tr><th class='model'>game</th><th>sample</th><th>models</th><th>top of board</th></tr>
    {trows}
  </table>
  <p class="note"><b>Read this honestly:</b> {len(big)} of the six core games
  ({big_games or 'none'}) have a <b>large statistical tie</b> — many leaders are inseparable within
  error bars, so their per-game <i>order</i> is essentially noise even though it still feeds the Arena
  scores. The board games and Blotto have a clear hierarchy and only a close gold/silver, so they carry
  most of the real signal. Bottom line: trust the broad tiers, not the exact rank of two adjacent
  models — and especially not the order inside the poker games' top group.</p>

  <h2>Limitations</h2>
  <ul class="tight note">
    <li><b>Luck-heavy games.</b> Poker and Blackjack have high variance; short runs reshuffle standings.
      Blackjack is kept off the headline for this reason; Hold'em 1-Hand and Leduc are included but their
      top ranks are currently ties (above).</li>
    <li><b>Claude models came from a different harness.</b> The Claude Opus/Sonnet results were produced in
      a separate Fireworks-hosted run, not the identical main pipeline, and only cover four games (so they
      read as 4/6 coverage). Treat Claude's placement as indicative, not strictly apples-to-apples.</li>
    <li><b>Ordinal vs margin.</b> Arena Rank Score ignores how decisively a game was won; Arena Elo adds
      that back. When the two disagree, the model is winning consistently but narrowly (or vice-versa).</li>
    <li><b>Hidden reasoning.</b> Some providers hide chain-of-thought, so completion-token counts are not a
      valid proxy for "how much a model thought" — we only report observable behaviour.</li>
    <li><b>Provenance.</b> Exact model snapshot dates are not yet pinned on the page; treat results as a
      point-in-time snapshot of the listed model versions.</li>
  </ul>

  <h2>Reproducibility</h2>
  <p class="note">Each report embeds its own data; charts are computed at generation time from the raw
  per-episode logs. Model display names map to the Fireworks catalog (e.g. <code>glm-5p1</code> →
  GLM-5.1). Company logos are the trademarks of their respective owners, shown only to identify each
  model's maker.</p>
</div></body></html>"""


def main():
    rows = build_rows()
    html = render(rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, "methodology.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['title']:20} {r['n']} {r['n_label']:12} -> {r['verdict']}"
              + (f" ({r['tie']}-way)" if r['verdict'] == 'tie' else ""))


if __name__ == "__main__":
    main()
