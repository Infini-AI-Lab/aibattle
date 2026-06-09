# New-Games Four-Model Experiment — Run Notes

## Current AC-8 structure

The experiment compares the four verified Fireworks models:
`kimi-k2p6`, `deepseek-v4-pro`, `glm-5p1`, and `gpt-oss-120b`.
Unavailable ids (`minimax-m2p7`, `deepseek-flash`) are intentionally out of scope.

| Game | Structure | Completion rule |
|------|-----------|-----------------|
| independent_blackjack | model vs built-in dealer | all 4 models have player-seat hands |
| leduc_poker | model-vs-model round-robin, seat-swapped | all 6 model pairs have both seat directions |
| repeated_colonel_blotto | model-vs-model round-robin, seat-swapped | all 6 model pairs have both seat directions |
| othello_lite_6x6 | model-vs-model round-robin, seat-swapped | all 6 model pairs have both seat directions |

The report now marks a round-robin game COMPLETE only when each model pair has
both seat directions represented. A single episode with `seat_swap=True` is not
enough because Runner treats `episodes` as the total episode budget.

## Overnight run

Run:

```bash
nohup bash scripts/run_new_games_tournament.sh > /tmp/overnight.log 2>&1 &
```

The overnight script uses `MAX_CONCURRENCY=512`. For each model-vs-model pair it
runs 10 seat-swapped deals, encoded as 20 total episodes per pair. Per-episode
resume is enabled, so reruns continue from completed episode files.

Results stream into `runs/new_games_experiment/<game>/data.json`, and the
aggregated report refreshes at `reports/new_games_leaderboard.md` and
`reports/new_games_leaderboard.json`.

To refresh the report from whatever is stored so far:

```bash
PYTHONPATH=src python -c "import scripts.new_games_experiment as e; e.write_report({})"
```

## Diagnostic baseline mode

`BASELINE_MODE=1` is still available for quick model-vs-local-baseline diagnosis
on long games, but it is not the AC-8 round-robin benchmark.
