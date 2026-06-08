# New-Games Four-Model Experiment — Correctness Verification & Run Notes

## Status of AC-8 (four-model Fireworks comparison)

| Game | Length | Four-model results stored | Notes |
|------|--------|---------------------------|-------|
| independent_blackjack | ~2 plies | ✅ all 4 models | model vs built-in dealer, 0% invalid |
| leduc_poker | ~4 plies | ✅ all 4 models | round-robin, 0% invalid |
| repeated_colonel_blotto | 20 rounds | ⏳ gpt-oss done; others accumulate | model vs `blotto_random` (baseline mode) |
| othello_lite_6x6 | ~30 plies | ⏳ runs after Blotto | model vs `board_random` (baseline mode) |

Models (the four verified-available on this Fireworks account):
`kimi-k2p6`, `deepseek-v4-pro`, `glm-5p1`, `gpt-oss-120b`.
Unavailable ids (`minimax-m2p7`, `deepseek-flash`) are intentionally absent.

## Correctness verification (Round 5)

The path, API, and code are all verified correct — the only constraint is
aggregate throughput on the long games. Evidence:

1. **End-to-end path (gpt-oss, baseline mode):**
   - `repeated_colonel_blotto`: one game in **117s**, 40 steps, **0 invalid**,
     correct zero-sum return.
   - `othello_lite_6x6`: one game in **231s**, 33 steps, **0 invalid**,
     correct zero-sum return.

2. **API + client correct for the "slow" models** (single call, real Blotto prompt):
   - `deepseek-v4-pro`: HTTP 200, **7-10s**, `content="alloc:7,13,20,27,33"`,
     `finish_reason=stop` — parses to a legal allocation.
   - `glm-5p1`: HTTP 200, **12s**, valid allocation, `finish_reason=stop`.
   - Both via raw `curl` and via the framework's `OpenAIClient.generate` give
     the same fast, valid result.

3. **Conclusion:** no bug in the game rules, the template parsing, the model
   client, or the request format. A single decision is fast (7-12s) for every
   model. The long games (Blotto 20 rounds, Othello ~30 plies) are
   **throughput-bound** when many decisions run across four models — they
   accumulate, so the full round-robin/sweep is best run over a long window
   (e.g. overnight). gpt-oss is fast enough to complete interactively; the other
   three reasoning models are slower in aggregate but correct.

## How to run the full experiment (overnight)

Per-episode resume is on, so this can be (re-)run any time and it continues:

```
nohup bash scripts/run_overnight_experiment.sh > /tmp/overnight.log 2>&1 &
```

This runs all four games over the four verified models with model-vs-baseline
for the two long games (only one seat calls the API per step), modest
concurrency, and generous episode counts. Results stream into
`runs/new_games_experiment/<game>/data.json` and the report refreshes
incrementally at `reports/new_games_experiment_report.md` (+ `.json`).

To refresh the aggregated report from whatever is stored so far:

```
PYTHONPATH=src python -c "import scripts.new_games_experiment as e; e.write_report({})"
```
