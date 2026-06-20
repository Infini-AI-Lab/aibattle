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
    "board": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted skill rating (Bradley–Terry, 1500-centred; higher = stronger) · <b>net/game</b> mean result per game (+1 win, −1 loss, 0 draw) · <b>win%</b> / <b>draw%</b> games won / drawn · <b>1st-move win%</b> win rate when moving first · <b>win-take</b> share of immediate wins converted (offense) · <b>block</b> share of opponent immediate-win threats stopped (defense) · <b>miss/allow</b> missed wins / allowed losses (counts) · <b>invalid%</b> illegal or unparseable moves · <b>plies</b> average game length in moves · <b>think</b> average seconds per decision</div>""",

    # leduc, repeated_colonel_blotto, othello_lite_6x6 (round-robin versus games)
    "versus": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted skill rating (Bradley–Terry, 1500-centred; higher = stronger) · <b>net/game</b> mean result per game (+1 win, −1 loss, 0 draw) · <b>win%</b> / <b>draw%</b> games won / drawn · <b>1st-move win%</b> win rate for the player who acts first · <b>invalid%</b> illegal or unparseable moves · <b>plies</b> average game length in moves · <b>think</b> average seconds per decision</div>""",

    # independent_blackjack (model vs the built-in dealer)
    "blackjack": """<div class="note collegend">columns: <b>mean/hand</b> average profit per hand (betting units) · <b>win%</b> / <b>push%</b> / <b>loss%</b> hands won / tied / lost vs dealer · <b>bust%</b> hands that went over 21 · <b>double%</b> hands doubled down · <b>natural%</b> hands dealt a natural blackjack · <b>invalid%</b> illegal or unparseable actions · <b>hands</b> hands played · <b>think</b> average seconds per decision</div>""",

    # holdem_1hand (each hand scored independently, bb/100)
    "holdem": """<div class="note collegend">columns: <b>style</b> play-style archetype (from VPIP/PFR/aggression) · <b>Elo</b> opponent-adjusted skill rating · <b>chips</b> net chips won · <b>bb/100</b> big blinds won per 100 hands (standard poker win-rate) · <b>win%</b> hands won · <b>VPIP</b> voluntarily-put-money-in-pot % (looseness) · <b>PFR</b> pre-flop raise % · <b>aggr</b> aggression — bet+raise share of bet/raise/call · <b>fold→bet</b> how often it folds facing a bet · <b>all-in%</b> hands taken all-in · <b>bet size</b> average bet/raise as a multiple of the pot · <b>think</b> average seconds per decision · <b>tokens/dec</b> average output tokens per decision</div>""",

    # kuhn_poker (solved game, scored on GTO fundamentals)
    "kuhn": """<div class="note collegend">columns: <b>fold K vs bet</b> how often it folds a King — the best card — facing a bet (a clear blunder) · <b>call J vs bet</b> how often it calls with a Jack — the worst card — facing a bet (a clear blunder) · <b>total blunders</b> combined count of these two mistakes</div>""",

    # holdem_match (heads-up, stacks carried, win the match)
    "match": """<div class="note collegend">columns: <b>Elo</b> opponent-adjusted skill rating · <b>win%</b> matches won · <b>wins/matches</b> matches won / played · <b>draws</b> matches drawn · <b>bust-out%</b> share of matches where the model lost all its chips · <b>hands/match</b> average hands per match · <b>avg win margin</b> average chip margin in matches it won</div>""",

    # holdem_table (5-handed ring, scored by finishing rank)
    "table": """<div class="note collegend">columns: <b>avg rank</b> average finishing position across tables (lower = better) · <b>top-1%</b> share of tables finished 1st · <b>avg final stack</b> average ending chip stack · <b>bust%</b> share of tables where it busted out · <b>#1–#5</b> finishing-place distribution: how often it placed 1st through 5th</div>""",
}


def legend(key: str) -> str:
    """Return the legend HTML for a report family, or "" if unknown."""
    return LEGENDS.get(key, "")
