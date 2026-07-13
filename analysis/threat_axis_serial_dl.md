# Experiment: ↙-contiguous serialization (rotated board) vs the ↘/↙ miss-rate gap (Gomoku)

**Question.** In the base row-major text, a horizontal line is contiguous in the token
stream and is the best-defended axis; ↙ diagonals are the most scattered and the worst.
If in-text contiguity is what makes an axis easy, then re-serializing the board so that
**↙ lines are the contiguous ones** should make ↙ the easy diagonal and ↘ the hard one.

**Design.** Same 515 diagonal probes (284 ↘ / 231 ↙), replayed under two arms with the
tournament's sampling settings. The new arm:

- `serial_dl` — the board is shown **rotated 45°** with its own coordinate system:
  each of the 17 anti-diagonals becomes a text row (rows 1-17), the 17 visual columns
  become letters A-Q; cell (r,c) of the original board appears at row r+c+1,
  column letter index 8+r-c. Rules and instruction are minimally reworded to the new
  frame; everything else in the stored prompt is unchanged. The model answers in
  rotated coordinates, which are remapped to the original cell for scoring
  (blocked / took-own-win / missed, as in all arms). A diamond layout is used (not a
  left-aligned stack) so that no false straight lines appear: real ↙ lines are
  horizontal text runs, real ↘ lines are the same column letter every 2nd row, and
  original horizontal/vertical lines are the two 45° staircases.

Example (real probe; X threatens to complete four on the ↙ line H2-G3-F4-E5 by playing
I1, which is A9 in the rotated view — the first cell of text row 9):

```
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   X
 3              .   .   .
 4            .   X   O   .
 5          .   .   .   .   .
 6        .   .   X   O   .   .
 7      .   .   O   X   .   X   .
 8    .   O   .   X   O   O   .   .
 9  .   X   X   X   X   O   .   .   .
10    .   .   X   O   O   X   .   .
11      .   O   X   O   O   O   .
12        .   .   X   X   .   .
13          .   .   O   .   .
14            .   .   .   .
15              .   .   .
16                .   .
17                  .
```

The ↙ threat that was scattered across five text rows in the base format is the
contiguous run `. X X X X` at the start of row 9.

**Results** (miss rate, minimax-m2p7; serial_dl arm: 3 unparsed, 0 truncated, 0
frame slips — the model consistently answered in the rotated coordinates). The
comparison is organized by the axis's **textual role** in each view, since the same
board axis plays different roles in the two serializations:

Diagonal cells of the base-view column are the base-arm replays from
`threat_axis_flip_rows.md` (same convention, same 515 probes); the horizontal/vertical
cells have no replay and fall back to the model's original tournament outcomes on its
own faced probes (marked *orig.*, stricter not-blocked = miss convention, small n).

**minimax-m2p7:**

| textual role in the view | base view | serial_dl view (replay, 515 probes) |
|---|---|---|
| contiguous text row      | horizontal: 6/44 = 13.6% *(orig.)* | ↙ diag_dl: 30/231 = 13.0% |
| across-line stride       | vertical: 9/43 = 20.9% *(orig.)*   | ↘ diag_dr: 61/284 = 21.5% |
| scattered diagonals      | ↘ 51/284 = 18.0% / ↙ 79/231 = 34.2% | orig. horizontal/vertical: not yet measured |

Making ↙ the contiguous axis reversed the diagonal gap (34.2% → 13.0% while
↘ 18.0% → 21.5%) rather than closing it.

A second model, **kimi-k2p6**, ran partially (stopped early at 243 of 515 probes;
0 unparsed, 0 frame slips):

| textual role in the view | base view | serial_dl view (partial replay) |
|---|---|---|
| contiguous text row      | horizontal: 2/45 = 4.4% *(orig.)* | ↙ diag_dl: 1/110 = 0.9% |
| across-line stride       | vertical: 4/41 = 9.8% *(orig.)*   | ↘ diag_dr: 0/133 = 0.0% |
| scattered diagonals      | ↘ 1/284 = 0.4% / ↙ 8/231 = 3.5% | orig. horizontal/vertical: not measured |

kimi stays at floor in the rotated format regardless of textual role: within the 243
paired probes it broke nothing that base had blocked (0 newly-missed on either axis),
fixed one base miss per axis, and its single remaining ↙ miss is a probe it had also
missed in base — its rates are too close to zero to separate the roles the way
minimax's do.

A third model, **deepseek-v4-pro** (full 515 probes; 0 unparsed, 0 frame slips) —
notable because it was the one model whose ↘/↙ gap did *not* move in the flip-rows
experiment:

| textual role in the view | base view | serial_dl view (replay, 515 probes) |
|---|---|---|
| contiguous text row      | horizontal: 0/36 = 0.0% *(orig.)* | ↙ diag_dl: 1/231 = 0.4% |
| across-line stride       | vertical: 9/52 = 17.3% *(orig.)*  | ↘ diag_dr: 4/284 = 1.4% |
| scattered diagonals      | ↘ 14/284 = 4.9% / ↙ 17/231 = 7.4% | orig. horizontal/vertical: not measured |

Its ↙ deficit vanishes when ↙ is the contiguous axis (paired: all 17 of its base ↙
misses were blocked in the rotated view; its single remaining ↙ miss is a new one),
and unlike minimax, ↘ also improved (12 fixed / 2 new) — the rotated view helped
deepseek on both diagonals (31 base misses → 5 overall).

Factual notes, no more: for minimax, whichever axis plays the "contiguous row" role
lands near 13%, and whichever plays the "across-line stride" role lands near 21%,
regardless of which board axis it is. Caveats: the *(orig.)* cells use original
tournament outcomes (small n, stricter not-blocked = miss convention) because
horizontal/vertical probes have not been replayed; the "scattered" cell of the
serial_dl column (original horizontal/vertical threats, which become staircases in
the rotated view) has not been run either. Cost observation: the rotated format
roughly quintupled reasoning length for minimax (median ~22k completion tokens per
call vs ~4k in base), yet its overall diagonal misses still fell (91/515 vs 130/515).
Three models tested (one partial); cross-experiment comparison is deferred until all
independent results are collected.

## Reproduce

```bash
# probes (shared with the other threat-axis experiments)
python3 scripts/threat_probes_extract.py gomoku

# serial_dl arm (resumable; re-run to fill API-error gaps)
python3 scripts/threat_probe_replay.py run \
    --models minimax-m2p7 --arms serial_dl --axes diag_dr,diag_dl --concurrency 32
python3 scripts/threat_probe_replay.py run \
    --models kimi-k2p6 --arms serial_dl --axes diag_dr,diag_dl --concurrency 32
#   (kimi run was stopped early at 243/515; re-running the command resumes it)
python3 scripts/threat_probe_replay.py run \
    --models deepseek-v4-pro --arms serial_dl --axes diag_dr,diag_dl --concurrency 32
#   -> runs/threat_probes/serial_dl/replays_gomoku_serial_dl.jsonl

# result tables
python3 scripts/threat_probe_replay.py analyze
```
