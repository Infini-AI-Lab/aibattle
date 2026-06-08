#!/usr/bin/env bash
# Overnight AC-8 four-model experiment for the new games (ORIGINAL design).
#
# AC-8 requires the multi-model Fireworks comparison across all four new games:
#   - round-robin (model-vs-model, seat-swap) for othello/leduc/blotto
#   - independent model-vs-dealer for blackjack
#
# The long games (Blotto 20 rounds, Othello ~30 plies) with model-vs-model are
# throughput-bound (both seats on slow reasoning models), so this runs them
# SERIAL across model pairs with low concurrency, ONE episode per pair, and a
# HIGH per-call timeout — intended to run overnight. Per-episode resume means it
# can be (re-)run any time to continue.
#
# Verified (Round 5): a single decision is fast (7-12s) for every model and the
# path/API/code are correct; the constraint is aggregate wall-clock, not bugs.
#
# Usage:  nohup bash scripts/run_overnight_experiment.sh > /tmp/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/haizhonz/letianr/src
export MODEL_TIMEOUT_S=900     # long games: allow slow reasoning steps
export MAX_CONCURRENCY=1       # episode-level concurrency within a match
export SERIAL_PAIRS=1          # one model pair at a time (round-robin path)
# BASELINE_MODE is intentionally UNSET so the long games use the AC-8
# model-vs-model round-robin (run_versus_game), not the diagnostic baseline.

# 1) Short games (fast): a couple hands per pair/model.
EPISODES=2 python scripts/new_games_experiment.py --episodes 2 \
  --games independent_blackjack,leduc_poker

# 2) Long games (slow): EXACTLY ONE episode per pair, serial, high timeout.
EPISODES=1 OTHELLO_EPISODES=1 python scripts/new_games_experiment.py --episodes 1 \
  --games repeated_colonel_blotto,othello_lite_6x6

echo "OVERNIGHT AC-8 EXPERIMENT DONE"
echo "Diagnostic-only model-vs-baseline mode (NOT AC-8) can be run separately with:"
echo "  BASELINE_MODE=1 PYTHONPATH=src python scripts/new_games_experiment.py \\"
echo "    --games repeated_colonel_blotto,othello_lite_6x6"
