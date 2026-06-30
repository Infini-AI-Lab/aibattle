"""Column glossaries ("legends") shown directly under each report's leaderboard.

These short, hand-authored `<div class="note collegend">` blocks were originally
pasted into the generated HTML by hand (PR #7). They are folded into the report
generators here so re-running the analysis pipeline reproduces them instead of
dropping them. Keyed by a semantic report family; several games share one legend.

Edit the prose here — every regenerated report picks it up automatically.
"""

from __future__ import annotations

LEGENDS = {
    # connect4, gomoku (board games with immediate-win/threat tactical columns)
    "board": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted rating · <b>net/game</b> mean result/game (+1/−1/0) · <b>win%</b> / <b>draw%</b> won / drawn · <b>1st-move win%</b> / <b>2nd-move win%</b> win rate moving first / second · <b>win-take</b> immediate wins converted · <b>block</b> opponent win-threats stopped · <b>miss/allow</b> missed wins / allowed losses · <b>think</b> avg sec/decision</div>""",

    # connect4, gomoku — slim "who won" leaderboard (v2 layout; tactical columns
    # move into the "why" section, so the results table keeps only outcomes)
    "board_results": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted rating · <b>net/game</b> mean result/game (+1/−1/0) · <b>win%</b> games won · <b>1st-move win%</b> / <b>2nd-move win%</b> win rate moving first / second · <b>games</b> games played (varies by wave)</div>""",

    # leduc, repeated_colonel_blotto, othello_lite_6x6 (round-robin versus games)
    "versus": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted rating · <b>net/game</b> mean result/game (+1/−1/0) · <b>win%</b> / <b>draw%</b> won / drawn · <b>1st-move win%</b> win rate acting first · <b>invalid%</b> illegal moves · <b>plies</b> avg game length · <b>think</b> avg sec/decision · <b>games</b> games played (varies by wave)</div>""",

    # independent_blackjack (model vs the built-in dealer)
    "blackjack": """<div class="note collegend">columns: <b>mean/hand</b> avg profit/hand (units) · <b>win%</b> / <b>push%</b> / <b>loss%</b> vs dealer · <b>bust%</b> went over 21 · <b>double%</b> doubled down · <b>natural%</b> dealt a blackjack · <b>invalid%</b> illegal actions · <b>hands</b> hands played · <b>think</b> avg sec/decision</div>""",

    # holdem_1hand (each hand scored independently, bb/100)
    "holdem": """<div class="note collegend">columns: <b>style</b> play-style archetype · <b>Elo</b> opponent-adjusted rating · <b>chips/hand</b> net chips/hand (normalized) · <b>bb/100</b> big blinds won per 100 hands · <b>win%</b> hands won · <b>tokens/dec</b> avg output tokens/decision · <b>hands</b> hands played (varies by wave)</div>""",

    # kuhn_poker (solved game, scored on GTO fundamentals)
    "kuhn": """<div class="note collegend">columns: <b>fold K vs bet</b> folds a King (best card) to a bet — a blunder · <b>call J vs bet</b> calls a Jack (worst card) to a bet — a blunder · <b>total blunders</b> sum of the two</div>""",

    # holdem_match (heads-up, stacks carried, win the match)
    "match": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted rating · <b>win%</b> matches won · <b>wins/matches</b> won / played · <b>draws</b> matches drawn · <b>bust-out%</b> matches losing all chips · <b>hands/match</b> avg hands/match · <b>avg win margin</b> avg chip margin in wins · <b>matches</b> matches played (varies by wave)</div>""",

    # holdem_table (5-handed ring, scored by finishing rank)
    "table": """<div class="note collegend">columns: <b>avg rank</b> avg finish (lower = better) · <b>top-1%</b> tables won · <b>avg final stack</b> avg ending stack · <b>bust%</b> tables busted out · <b>tables</b> tables played (varies by wave) · <b>#1–#5</b> finish-place distribution</div>""",
}


def legend(key: str) -> str:
    """Return the legend HTML for a report family, or "" if unknown."""
    return LEGENDS.get(key, "")
