#!/usr/bin/env bash
# Overnight AC-8 four-model experiment for the new games (ORIGINAL design).
#
# AC-8 requires the multi-model Fireworks comparison across all four new games:
#   - round-robin (model-vs-model, seat-swap) for othello/leduc/blotto
#   - independent model-vs-dealer for blackjack
#
# All games run in parallel in a single process under one shared 128-call
# concurrency budget. Each versus pair runs 25 seat-swapped deals, i.e. 50 total
# episodes per pair, so both seat directions are represented; blackjack plays
# 100 hands per model. Per-episode resume means it can be (re-)run any time to
# continue.
#
# Verified (Round 5): a single decision is fast (7-12s) for every model and the
# path/API/code are correct; the constraint is aggregate wall-clock, not bugs.
#
# Usage:  nohup bash scripts/run_new_games_tournament.sh > /tmp/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_TIMEOUT_S=900     # long games: allow slow reasoning steps
export MAX_CONCURRENCY=128     # ONE global model-call budget across all games
export SERIAL_PAIRS=0          # run model pairs concurrently under MAX_CONCURRENCY
export PARALLEL_GAMES=1        # all games run concurrently in one process, so
                               # the 128-call budget is truly shared (separate
                               # processes would each get their own semaphore)
# BASELINE_MODE is intentionally UNSET so the long games use the AC-8
# model-vs-model round-robin (run_versus_game), not the diagnostic baseline.

# All four games in one invocation: blackjack 100 hands/model; Leduc, Blotto
# and Othello 25 seat-swapped deals per pair (50 total episodes per pair).
EPISODES=50 BLACKJACK_EPISODES=100 OTHELLO_EPISODES=50 \
python scripts/new_games_tournament.py --episodes 50 \
  --games independent_blackjack,leduc_poker,repeated_colonel_blotto,othello_lite_6x6

echo "OVERNIGHT AC-8 EXPERIMENT DONE"
echo "Diagnostic-only model-vs-baseline mode (NOT AC-8) can be run separately with:"
echo "  BASELINE_MODE=1 PYTHONPATH=src python scripts/new_games_tournament.py \\"
echo "    --games repeated_colonel_blotto,othello_lite_6x6"
