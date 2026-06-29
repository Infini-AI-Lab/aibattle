#!/usr/bin/env bash
#
# Rebuild the static site under reports/ from the committed analysis JSONs and
# the raw runs/ logs, in dependency order.
#
#   ./build.sh
#
# Prerequisites:
#   * the runs/ tree is present (the episode logs; gitignored, not shipped)
#   * the committed reports/*_analysis.json inputs exist (they do, in git)
#
# Deploy: publish the reports/ directory. reports/runs is a symlink to ../runs;
# the replay viewers fetch reports/runs/<game>/replays/.../manifest.json, so the
# host MUST follow/include that symlink (e.g. `cp -RL`, or rsync -L) or the
# replays + featured dropdowns will 404.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src:scripts

run() { echo "==> $*"; python3 "$@"; }

# 1. Heavy analyzers — only if their committed JSON is missing (the Monte-Carlo
#    decision-quality pass takes ~20 min; skip when already built).
[ -f reports/match_factors.json ]          || run scripts/analyze_match_factors.py
[ -f reports/match_decision_quality.json ] || run scripts/analyze_match_decision_quality.py

# 2. Render per-game reports. analyze_board_tournament.py builds the overview
#    (index.html) and reads the holdem/match/new-games outputs, so it runs last.
run scripts/analyze_tournament.py          # Hold'em 1-Hand
run scripts/analyze_match_tournament.py    # Hold'em Match
run scripts/analyze_kuhn_tournament.py     # Kuhn Poker
run scripts/analyze_table_tournament.py    # Hold'em Table
run scripts/analyze_new_games.py           # Leduc / Blotto / Othello / Blackjack (+ new_games_index.json)
run scripts/analyze_board_tournament.py    # Connect Four / Gomoku + overview index.html
run scripts/generate_qa.py                 # Q&A page

# 3. Full replay data (gitignored, ~3.4 GB) then the curated featured sets.
#    build_featured validates every pick against the manifests AND extracts the
#    featured episodes into committed slim copies (reports/replays/), so it runs
#    last and its output is what the deployed viewers actually read.
run scripts/build_replays.py               # connect4/gomoku/holdem/match/table/kuhn
run scripts/build_new_games_replays.py     # leduc/blotto/othello/blackjack
run scripts/build_featured_replays.py      # reports/*_featured.json + reports/replays/

echo "==> done. Serve reports/ (the featured replays ship inside it; no symlink needed)."
