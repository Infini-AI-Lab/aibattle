#!/usr/bin/env bash
# Recovery pass for minimax-m3 episodes dropped during transient Fireworks
# endpoint slowdowns in the main run_m3_eval.sh run. Re-runs ONLY the 4 games
# left under target; per-episode resume skips every already-complete episode
# (all incumbent pairs + the m3 episodes that did land), so only the missing m3
# episodes are retried. IDENTICAL frozen settings to run_m3_eval.sh — nothing
# changed except which games run.
set -u
cd "$(dirname "$0")/.."
# Activate your environment (e.g. `conda activate aibattle`) before running.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export COACHED=1
export MAX_CONCURRENCY=64
export MODEL_TIMEOUT_S=900
export BOARD_TIMEOUT=900
export HOLDEM_TIMEOUT=900

OLD_MODELS="deepseek-v4-pro,gpt-oss-120b,kimi-k2p6,glm-5p1,minimax-m2p7,minimax-m3"
NG_MODELS="kimi-k2p6,deepseek-v4-pro,glm-5p1,minimax-m2p7,gpt-oss-120b,minimax-m3"

log() { echo "[$(date '+%a %H:%M:%S')] $*"; }

# ---- ALL FOUR under-target games TOGETHER, one shared 64 semaphore ----
# connect4 + gomoku + blotto + holdem_match all run in a single process sharing
# one asyncio.Semaphore(64). Total Fireworks concurrency stays <=64, no slot ever
# sits idle, and — crucially — the board games' intermittent hung-socket episodes
# can't block the others: blotto/holdem keep the budget saturated regardless.
log "=== connect4 + gomoku + blotto + holdem_match (combined all-4, shared 64) [recovery] ==="
NG_MODELS="$NG_MODELS" OLD_MODELS="$OLD_MODELS" \
  BOARD_EPISODES=10 BLOTTO_EPISODES=20 MATCH_EPISODES=40 \
  python scripts/combined_recovery.py

log "M3 RECOVERY DONE across connect4, gomoku, blotto, holdem_match"
