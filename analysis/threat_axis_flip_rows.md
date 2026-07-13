# Experiment: flipped row labels vs the ↘/↙ diagonal miss-rate gap (Gomoku)

**Question.** The board reports show single blockable threats are missed most often on the
↙ (down-left) diagonal, more than ↘ (see "miss-rate by threat axis" in
`reports/gomoku_report.html`). Two candidate explanations:

- **E1 — perceptual layout:** models read the 2-D shape of the row-major text; ↙ runs
  against the left-to-right reading order.
- **E2 — coordinate arithmetic:** models track (row, col) index labels; ↙ is hard because
  the two counters move in opposite directions.

**Design.** Every decision in the Gomoku runs where the acting model faced exactly one
blockable immediate-loss threat was extracted as a probe (515 diagonal probes; double
threats excluded; extraction reproduces the report's pinned numbers exactly). Each probe
was replayed as a single-move query under two arms, same positions, same sampling
settings as the tournament (temperature 0.6, max_tokens 131072, Fireworks):

- `base` — the verbatim prompt stored in the run logs;
- `flip_rows` — identical text, except the printed row labels go 9→1 top-to-bottom
  instead of 1→9. The board lines stay in the same order, so the *layout* is unchanged
  while every physical ↙ line becomes label-lockstep (and ↘ becomes label-opposing).

Example — one real probe (`gomoku__claude-opus-4.8__vs__claude-sonnet-4.6/ep001.json`
step 27, ↙ threat: X has four on E5-F4-G3-H2 and wins at the top-right cell). Only the
leading digits differ; every board line stays on the same text row:

```
arm base (verbatim from log)      arm flip_rows (labels only)
   A B C D E F G H I                 A B C D E F G H I
 1 . . . . . . . . .               9 . . . . . . . . .
 2 X . X . . . O X .               8 X . X . . . O X .
 3 . O . X O . X . .               7 . O . X O . X . .
 4 . . O X X X X O .               6 . . O X X X X O .
 5 . . . O X O X . .               5 . . . O X O X . .
 6 . X O O O O X . .               4 . X O O O O X . .
 7 . . . X O X O . .               3 . . . X O X O . .
 8 . . . O . . . . .               2 . . . O . . . . .
 9 . . . . . . . . .               1 . . . . . . . . .
```

The blocking cell is the same physical square, named `I1` under base labels and `I9`
under flipped labels. Walking down this ↙ line: base labels have row increasing while
column decreases (opposing counters); flipped labels have both decreasing (lockstep).

E1 predicts the gap stays on physical ↙; E2 predicts it moves to physical ↘.
Answers are scored in the arm's label frame; answers that would only block under the
wrong frame are logged as `frame_slip` (observed: 2 of 4120 — frame confusion is not a
factor). Misses = neither blocked nor took an available own win.

**Results** (miss rate, n=284 ↘ / 231 ↙ per arm per model):

| model          | base ↘ / ↙      | flip ↘ / ↙      |
|----------------|-----------------|-----------------|
| deepseek-v4-pro| 4.9% / 7.4%     | 3.2% / 5.6%     |
| kimi-k2p6      | 0.4% / 3.5%     | 1.1% / 1.3%     |
| minimax-m2p7   | 18.0% / 34.2%   | 20.4% / 22.1%   |
| gpt-oss-120b   | 10.6% / 31.2%   | 15.8% / 18.6%   |
| qwen3p7-plus   | 7.7% / 11.3%    | 8.8% / 7.8%     |
| **pooled**     | 8.3% / 17.5%    | 9.9% / 11.1%    |

**Initial read (no more than the data supports):**

1. The ↘/↙ asymmetry replicates on identical replayed positions — not a full-game
   confound.
2. For 4 of 5 models the gap largely disappears when only the labels flip: ↙ improves
   (minimax, gpt-oss, kimi, qwen), ↘ degrades slightly (gpt-oss, qwen). Since the text
   layout is unchanged, this part of the asymmetry is tied to the coordinate labels
   (consistent with E2), not to reading order.
3. No model shows a full reversal, and physical ↙ stays directionally worse in the flip
   arm for four of the five models (qwen is the exception, at 8.8/7.8) — a layout-bound
   residual (E1) is possible but not established. deepseek's (small) gap did not move
   with the labels.
4. Mechanism appears to differ across models; pooled report numbers hide this.

Untested here: the mirror-columns arm (true visual flip), other axes as controls,
Connect Four (its actions have no row labels), Claude models (not on Fireworks).

## Reproduce

```bash
# 1) extract probes from run logs; validates against the report's pinned numbers
python3 scripts/threat_probes_extract.py gomoku
#    -> runs/threat_probes/probes_gomoku.jsonl

# 2) replay both arms (resumable; re-run the same command to fill API-error gaps)
python3 scripts/threat_probe_replay.py run \
    --models deepseek-v4-pro,kimi-k2p6,minimax-m2p7,gpt-oss-120b \
    --axes diag_dr,diag_dl --arms base,flip_rows --concurrency 32
python3 scripts/threat_probe_replay.py run \
    --models qwen3p7-plus \
    --axes diag_dr,diag_dl --arms base,flip_rows --concurrency 16  # tight rate limit
#    -> appends runs/threat_probes/replays_gomoku.jsonl

# 3) result tables
python3 scripts/threat_probe_replay.py analyze
```

Needs the `.fireworks` key file in the repo root. ~1030 calls/model at ~4.5k reasoning
tokens each. Raw results for the run above are in `runs/threat_probes/`.
