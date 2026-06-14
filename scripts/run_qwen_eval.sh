#!/usr/bin/env bash
# Evaluate qwen3p7-plus vs minimax-m3 / deepseek-v4-pro / kimi-k2p6 on
# holdem_1hand + holdem_match + independent_blackjack, frozen settings.
#
# GATED: waits until the in-progress m3 recovery has fully finished (blotto
# 100/100 AND no combined_recovery.py process) so the two never run two
# 64-cap processes at once (that would push Fireworks concurrency to ~128).
set -u
cd "$(dirname "$0")/.."
export PATH="/home/haizhonz/anaconda3/envs/aibattle/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export COACHED=1
export MAX_CONCURRENCY=64
export MAX_TOKENS=131072
export TEMPERATURE=0.6
export MODEL_TIMEOUT_S=900
export HOLDEM_TIMEOUT=900
export MODELS="deepseek-v4-pro,kimi-k2p6,minimax-m3,qwen3p7-plus"
export HANDS=50 MATCH_EPISODES=40 BLACKJACK_EPISODES=100

log() { echo "[$(date '+%a %H:%M:%S')] $*"; }

BL="runs/new_games_experiment/repeated_colonel_blotto"
log "waiting for m3 recovery to finish (blotto 100/100 + no combined_recovery proc)..."
while :; do
  n=$(for d in $BL/*minimax-m3*; do [ -d "$d" ] && ls "$d"/ep*.json 2>/dev/null; done | wc -l)
  alive=$(pgrep -f scripts/combined_recovery.py | wc -l)
  if [ "$n" -ge 100 ] && [ "$alive" -eq 0 ]; then break; fi
  sleep 30
done
log "m3 recovery complete; starting qwen3p7-plus eval"

log "=== holdem_1hand + holdem_match + blackjack (combined, shared 64) [qwen] ==="
python scripts/combined_qwen.py

log "QWEN EVAL DONE across holdem_1hand, holdem_match, independent_blackjack"
