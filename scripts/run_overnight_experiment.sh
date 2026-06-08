#!/usr/bin/env bash
# Overnight four-model experiment for the new games.
#
# Verified facts (Round 5): the path/API/code are all correct — gpt-oss runs
# Blotto end-to-end in ~117s and Othello in ~231s with 0 invalid; deepseek-v4-pro
# and glm-5p1 return a valid allocation in 7-12s per single call. The only issue
# is aggregate throughput on the long games (Blotto 20 rounds, Othello ~30 plies)
# across four models, which is fine to run overnight.
#
# This wrapper runs everything with generous settings and per-episode resume, so
# it can be re-run any time to continue. Blotto/Othello use model-vs-baseline
# (BASELINE_MODE=1) so only one seat calls the API per step.
#
# Usage:  nohup bash scripts/run_overnight_experiment.sh > /tmp/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/home/haizhonz/letianr/src
export BASELINE_MODE=1
export MAX_CONCURRENCY=4      # modest concurrency; long games are throughput-bound
export EPISODES=10            # hands per model for the short games
export OTHELLO_EPISODES=3     # Othello games are long; fewer per model
# Short games first (fast, all four models), then the long games.
python scripts/new_games_experiment.py --episodes 10 \
  --games independent_blackjack,leduc_poker,repeated_colonel_blotto,othello_lite_6x6
echo "OVERNIGHT EXPERIMENT DONE"
