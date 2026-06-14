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
# All FIVE models — the python script's default MODELS list predates
# minimax-m2p7 becoming available and silently runs only four.
export MODELS=${MODELS:-kimi-k2p6,deepseek-v4-pro,glm-5p1,minimax-m2p7,gpt-oss-120b}
export MODEL_TIMEOUT_S=${MODEL_TIMEOUT_S:-900}   # long games: allow slow reasoning steps
# Full 131k output cap — an eval condition, NOT tunable for speed (per owner:
# capping reasoning changes model behavior). Long Blotto/Othello decisions from
# heavy reasoners (~6+ min each measured) mean the run takes a day or more;
# that is accepted. Stamped into data.json settings.
export MAX_TOKENS=${MAX_TOKENS:-131072}
# ONE global model-call budget across all games (per owner: 64). Note the
# account's observed limits: at 128 every request 429'd and the run wedged in a
# retry storm; at 64 the account sits at its concurrency ceiling (extra calls
# 429, established streams do progress).
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
export SERIAL_PAIRS=0          # run model pairs concurrently under MAX_CONCURRENCY
export PARALLEL_GAMES=1        # all games run concurrently in one process, so
                               # the 128-call budget is truly shared (separate
                               # processes would each get their own semaphore)
# BASELINE_MODE is intentionally UNSET so the long games use the AC-8
# model-vs-model round-robin (run_versus_game), not the diagnostic baseline.

# Blackjack (100 hands/model) and Leduc (50 eps/pair) completed 2026-06-11/12;
# they are NOT relaunched so their stored aggregates stay at full size. Only the
# long games run here. Per owner decision 2026-06-12: Blotto 20 episodes/pair,
# Othello 50 episodes/pair.
EPISODES=20 OTHELLO_EPISODES=50 \
python scripts/new_games_tournament.py --episodes 20 \
  --games repeated_colonel_blotto,othello_lite_6x6

echo "OVERNIGHT AC-8 EXPERIMENT DONE"
echo "Diagnostic-only model-vs-baseline mode (NOT AC-8) can be run separately with:"
echo "  BASELINE_MODE=1 PYTHONPATH=src python scripts/new_games_tournament.py \\"
echo "    --games repeated_colonel_blotto,othello_lite_6x6"
