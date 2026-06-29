#!/usr/bin/env bash
# Incremental evaluation of accounts/fireworks/models/minimax-m3 across 8 games
# (all old + new EXCEPT holdem_table and othello, which the owner skipped).
#
# Approach: append minimax-m3 to each script's EXISTING model list (preserving
# order, so itertools.combinations keeps the 10 incumbent pairs byte-identical
# and per-episode resume skips them). Only m3's 5 new pairs per round-robin game
# run live, plus 1 new model-vs-dealer run for blackjack. Coach mode only.
# Concurrency capped at 64 (128 triggers a Fireworks 429 storm on this account).
#
# Gated: waits until the in-progress new_games run (blotto+othello) has fully
# exited so the two never compete for the 64-call budget.
set -u
cd "$(dirname "$0")/.."
# Activate your environment (e.g. `conda activate aibattle`) before running.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export COACHED=1
export MAX_CONCURRENCY=64
# 900s per-call timeout for EVERY game (kuhn+board were at 300s; m3 occasionally
# loops at temp 0 and was hitting that, dropping ~26% of kuhn episodes). kuhn
# reads MODEL_TIMEOUT_S, board reads BOARD_TIMEOUT; holdem/match/new_games
# already default to 900.
export MODEL_TIMEOUT_S=900
export BOARD_TIMEOUT=900
export HOLDEM_TIMEOUT=900

# Exact model order each script used to create its on-disk data, + m3 appended.
OLD_MODELS="deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7,minimax-m3"
NG_MODELS="kimi-k2p6,deepseek-v4-pro,glm-5p1,minimax-m2p7,gpt-oss-120b,minimax-m3"

log() { echo "[$(date '+%a %H:%M:%S')] $*"; }

# ---- Gate: wait for the current othello run to reach 500/500 and exit ----
OTH="runs/new_games_experiment/othello_lite_6x6"
log "waiting for current new_games run to finish (othello 500/500 + process exit)..."
while :; do
  n=$(find "$OTH" -name 'ep*.json' 2>/dev/null | wc -l)
  alive=$(pgrep -f scripts/new_games_tournament.py | wc -l)
  if [ "$n" -ge 500 ] && [ "$alive" -eq 0 ]; then break; fi
  sleep 60
done
log "current run complete; starting m3 incremental eval"

# ---- 1) Kuhn (fastest) ----
log "=== kuhn (30/pair) ==="
MODELS="$OLD_MODELS" EPISODES=30 python scripts/kuhn_tournament.py

# ---- 2) Board games: connect4 + gomoku (10/pair) ----
log "=== connect4 + gomoku (10/pair) ==="
MODELS="$OLD_MODELS" BOARD_GAMES=connect4,gomoku EPISODES=10 python scripts/board_tournament.py

# ---- 3) Blackjack (vs dealer, 100 hands) + Leduc (50/pair) ----
log "=== blackjack (100 vs dealer) + leduc (50/pair) ==="
MODELS="$NG_MODELS" EPISODES=50 BLACKJACK_EPISODES=100 \
  python scripts/new_games_tournament.py --episodes 50 \
  --games independent_blackjack,leduc_poker

# ---- 4) Hold'em 1-hand (50/pair) ----
log "=== holdem_1hand (50/pair) ==="
MODELS="$OLD_MODELS" HANDS=50 python scripts/tournament.py

# ---- 5) Colonel Blotto (20/pair) ----
log "=== colonel_blotto (20/pair) ==="
MODELS="$NG_MODELS" EPISODES=20 \
  python scripts/new_games_tournament.py --episodes 20 \
  --games repeated_colonel_blotto

# ---- 6) Hold'em match (40/pair, up to 30 hands each — slowest, last) ----
log "=== holdem_match (40/pair) ==="
MODELS="$OLD_MODELS" EPISODES=40 python scripts/match_tournament.py

log "M3 EVAL DONE across kuhn, connect4, gomoku, blackjack, leduc, holdem_1hand, blotto, holdem_match"
