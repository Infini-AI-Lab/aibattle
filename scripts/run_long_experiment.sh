#!/usr/bin/env bash
# Overnight AC-8 four-model experiment for the new games (ORIGINAL design).
#
# AC-8 requires the multi-model Fireworks comparison across all four new games:
#   - round-robin (model-vs-model, seat-swap) for othello/leduc/blotto
#   - independent model-vs-dealer for blackjack
#
# The long games (Blotto 20 rounds, Othello ~30 plies) with model-vs-model are
# throughput-bound, so this uses a high shared concurrency limit. Each model pair
# runs 10 seat-swapped deals, i.e. 20 total episodes per pair, so both seat
# directions are represented. Per-episode resume means it can be (re-)run any
# time to continue.
#
# Verified (Round 5): a single decision is fast (7-12s) for every model and the
# path/API/code are correct; the constraint is aggregate wall-clock, not bugs.
#
# Usage:  nohup bash scripts/run_long_experiment.sh > /tmp/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_TIMEOUT_S=900     # long games: allow slow reasoning steps
export MAX_CONCURRENCY=512     # shared model-call concurrency across episodes/pairs
export SERIAL_PAIRS=0          # run model pairs concurrently under MAX_CONCURRENCY
# BASELINE_MODE is intentionally UNSET so the long games use the AC-8
# model-vs-model round-robin (run_versus_game), not the diagnostic baseline.

# 1) Short games: blackjack gets 10 hands/model; Leduc gets 10 seat-swapped
# deals per pair (20 total episodes per pair).
EPISODES=10 python scripts/new_games_experiment.py --episodes 10 \
  --games independent_blackjack
EPISODES=20 python scripts/new_games_experiment.py --episodes 20 \
  --games leduc_poker

# 2) Long games: 10 seat-swapped deals per pair (20 total episodes per pair).
EPISODES=20 OTHELLO_EPISODES=20 python scripts/new_games_experiment.py --episodes 20 \
  --games repeated_colonel_blotto,othello_lite_6x6

echo "OVERNIGHT AC-8 EXPERIMENT DONE"
echo "Diagnostic-only model-vs-baseline mode (NOT AC-8) can be run separately with:"
echo "  BASELINE_MODE=1 PYTHONPATH=src python scripts/new_games_experiment.py \\"
echo "    --games repeated_colonel_blotto,othello_lite_6x6"
