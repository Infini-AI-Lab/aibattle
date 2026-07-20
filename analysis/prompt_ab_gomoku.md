# Experiment: prompt-modification A/B self-play (Gomoku, minimax-m2p7)

**Question.** The threat-axis experiments showed miss rates depend on how the board is
serialized. Does that convert into GAME performance — can a modified prompt beat the
original one in real games?

**Design.** Full Gomoku games between two seats of the SAME model (minimax-m2p7, the
fastest of the studied models): one seat gets the original coached tournament prompt
(`orig`), the other a modified rendering. 100 games per setting, seeds 0-99 shared
across settings (same random 2-ply openings), seat-balanced (variant is player_0 on
even seeds), tournament sampling settings (temperature 0.6, max_tokens 131072, same
retry/repair/fallback protocol; the `orig` prompt is byte-identical to the stored
tournament prompts except the Match line is fixed at "Hand 1 of 1"). Score = variant
wins + half its draws. Per-decision tactical stats (block rate, invalid moves) are
logged per seat. Harness: `scripts/prompt_ab_games.py`; data:
`runs/prompt_ab/minimax-m2p7__<setting>.jsonl`.

**Results** (sorted by variant score; blk% = single-threat block rate variant/orig;
inv = unparsed-move fallbacks variant/orig):

| setting                 | n   | score  | blk% v/o | inv v/o | what the variant seat sees                                        |
|-------------------------|-----|--------|----------|---------|-------------------------------------------------------------------|
| orig+stones             | 100 | 75.5%  | 93%/86%  | 0/11    | standard board plus a coordinate list of every stone              |
| orig+serial_dl          | 100 | 59.0%  | 88%/75%  | 7/7     | standard board plus the ↙-rotated diamond view                    |
| flip+serial_dl          | 99  | 55.1%  | 90%/72%  | 6/14    | flipped-label board plus the ↙-rotated view (no standard view)    |
| orig+flip+serial_dl     | 98  | 55.1%  | 83%/78%  | 8/5     | standard, flipped-label, and ↙-rotated views together             |
| orig+serial_dr          | 99  | 53.5%  | 75%/80%  | 9/10    | standard board plus the ↘-rotated diamond view                    |
| orig+transpose          | 100 | 53.5%  | 89%/75%  | 2/7     | standard board plus a column-major view                           |
| flip                    | 100 | 50.0%  | 79%/73%  | 3/3     | standard board with row labels renumbered 9→1 top-to-bottom       |
| line_coach              | 100 | 48.0%  | 71%/78%  | 10/16   | standard board plus an explicit scan-all-four-axes instruction    |
| four_views              | 100 | 47.5%  | 78%/78%  | 1/9     | standard, column-major, and both rotated views together           |
| orig+flip               | 99  | 41.9%  | 76%/83%  | 10/9    | standard board plus the flipped-label rendering of the same board |
| serial_dl               | 100 | 28.0%  | 51%/73%  | 10/7    | ONLY the ↙-rotated diamond, with its own A-Q/1-17 coordinates     |
| serial_dr               | 100 | 27.0%  | 63%/71%  | 12/9    | ONLY the ↘-rotated diamond, with its own A-Q/1-17 coordinates     |
| transpose               | 100 | 3.5%   | 26%/89%  | 45/2    | ONLY a column-major board (each text line is one column A-I)      |

**Column definitions:**

- **score** — the variant seat's game score against the orig seat over the n games:
  (variant wins + 0.5 × draws) / n. 50% = the modification makes no difference;
  above 50% = the modified prompt wins the head-to-head.
- **blk% v/o** — block rate, variant seat / orig seat: of the decisions where that
  seat faced exactly ONE blockable immediate-loss threat (and had no immediate win
  of its own), the share where it played the blocking cell. Same definition as the
  threat-axis probe experiments.
- **inv v/o** — invalid-move fallbacks, variant / orig: total moves (across all n
  games) where no legal coordinate could be parsed from the model's reply after 3
  attempts, so a random legal move was played instead (the tournament's fallback
  rule). High counts mean the seat could not reliably express moves in that
  prompt's coordinate frame.

Settings: `stones` = coordinate list of all stones; `flip` = row labels 9→1;
`serial_dl`/`serial_dr` = 45°-rotated board with its own A-Q/1-17 coordinates;
`transpose` = column-major view; `orig+X` = both views in one prompt (answer in
standard coordinates); `four_views` = orig+transpose+both rotations; `line_coach` =
original board plus an explicit scan-all-four-axes instruction.

## Prompt samples

All prompts share the coached tournament skeleton — rules, coaching line,
`Match: Hand 1 of 1. / You are X.`, board section, instruction — and differ only
in the board section, plus reworded rules/instruction for the rotated-coordinate
settings and an answer-frame sentence for multi-view settings. The full `orig`
prompt on a sample position, then each setting's changed sections:

```text
You are playing Gomoku-Lite (9x9). Place a stone on any empty cell; connect five in a row (horizontal, vertical, or diagonal) to win. Columns are A-I, rows 1-9; center is E5.

Before you move, check whether you can make five, whether you must block the opponent's line, and how your own stones connect.

Match: Hand 1 of 1.
You are X.
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer.
```

### `flip`

```text
Match: Hand 1 of 1.
You are X.
   A B C D E F G H I
 9 . . . . . . O . .
 8 . X . . . . . . .
 7 . O . . . . . . .
 6 . . . . . . . . .
 5 . . . . . X . . .
 4 . . . . . . . X .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . O . . . . . . .
```

### `serial_dl`

```text
[rules line, reworded]
You are playing Gomoku-Lite (9x9). Place a stone on any empty cell; connect five in a row (horizontal, vertical, or diagonal on the original board) to win. In the rotated view below, columns are A-Q, rows 1-17; the board center is I9.

Match: Hand 1 of 1.
You are X.
This view is rotated 45 degrees: each line is one down-left diagonal of the board (row +1, column -1 per step to the right).

    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              .   X   .
 4            .   .   O   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      O   .   .   .   .   .   .
 8    .   .   .   .   .   .   .   .
 9  .   .   .   .   .   .   .   .   .
10    .   .   .   X   .   .   .   O
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   X   .   .   .
14            .   .   .   .
15              .   .   .
16                .   .
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell in the rotated view, e.g. I9 (column letter A-Q, row number 1-17). Think privately before you answer.
```

### `serial_dr`

```text
[rules line, reworded]
You are playing Gomoku-Lite (9x9). Place a stone on any empty cell; connect five in a row (horizontal, vertical, or diagonal on the original board) to win. In the rotated view below, columns are A-Q, rows 1-17; the board center is I9.

Match: Hand 1 of 1.
You are X.
This view is rotated 45 degrees: each line is one down-right diagonal of the board (row +1, column +1 per step to the right).

    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              O   .   .
 4            .   .   .   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      .   .   .   .   .   X   .
 8    .   .   .   .   X   .   .   .
 9  .   X   .   .   .   .   .   .   .
10    .   O   .   .   .   .   .   .
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   .   .   .   .
14            .   .   .   .
15              .   .   .
16                .   O
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell in the rotated view, e.g. I9 (column letter A-Q, row number 1-17). Think privately before you answer.
```

### `transpose`

```text
Match: Hand 1 of 1.
You are X.
   1 2 3 4 5 6 7 8 9
 A . . . . . . . . .
 B . X O . . . . . O
 C . . . . . . . . .
 D . . . . . . . . .
 E . . . . . . . . .
 F . . . . X . . . .
 G O . . . . . . . .
 H . . . . . X . . .
 I . . . . . . . . .
```

### `orig+flip`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 2 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — same board with row labels flipped (9 at top, 1 at bottom):
   A B C D E F G H I
 9 . . . . . . O . .
 8 . X . . . . . . .
 7 . O . . . . . . .
 6 . . . . . . . . .
 5 . . . . . X . . .
 4 . . . . . . . X .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . O . . . . . . .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `orig+serial_dl`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 2 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — same board rotated 45 degrees (each line is one down-left diagonal; reading aid only):
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              .   X   .
 4            .   .   O   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      O   .   .   .   .   .   .
 8    .   .   .   .   .   .   .   .
 9  .   .   .   .   .   .   .   .   .
10    .   .   .   X   .   .   .   O
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   X   .   .   .
14            .   .   .   .
15              .   .   .
16                .   .
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `orig+serial_dr`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 2 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — same board rotated 45 degrees the other way (each line is one down-right diagonal; reading aid only):
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              O   .   .
 4            .   .   .   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      .   .   .   .   .   X   .
 8    .   .   .   .   X   .   .   .
 9  .   X   .   .   .   .   .   .   .
10    .   O   .   .   .   .   .   .
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   .   .   .   .
14            .   .   .   .
15              .   .   .
16                .   O
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `orig+transpose`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 2 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — same board column by column (each line is one column A-I):
   1 2 3 4 5 6 7 8 9
 A . . . . . . . . .
 B . X O . . . . . O
 C . . . . . . . . .
 D . . . . . . . . .
 E . . . . . . . . .
 F . . . . X . . . .
 G O . . . . . . . .
 H . . . . . X . . .
 I . . . . . . . . .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `orig+stones`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 2 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — the stones as coordinate lists:
X stones: B2, F5, H6
O stones: G1, B3, B9

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `orig+flip+serial_dl`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 3 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — same board with row labels flipped (9 at top, 1 at bottom):
   A B C D E F G H I
 9 . . . . . . O . .
 8 . X . . . . . . .
 7 . O . . . . . . .
 6 . . . . . . . . .
 5 . . . . . X . . .
 4 . . . . . . . X .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . O . . . . . . .

View 3 — same board rotated 45 degrees (each line is one down-left diagonal; reading aid only):
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              .   X   .
 4            .   .   O   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      O   .   .   .   .   .   .
 8    .   .   .   .   .   .   .   .
 9  .   .   .   .   .   .   .   .   .
10    .   .   .   X   .   .   .   O
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   X   .   .   .
14            .   .   .   .
15              .   .   .
16                .   .
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `four_views`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 4 views of the SAME board.

View 1 — standard board (row by row, rows 1-9 top to bottom):
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .

View 2 — same board column by column (each line is one column A-I):
   1 2 3 4 5 6 7 8 9
 A . . . . . . . . .
 B . X O . . . . . O
 C . . . . . . . . .
 D . . . . . . . . .
 E . . . . . . . . .
 F . . . . X . . . .
 G O . . . . . . . .
 H . . . . . X . . .
 I . . . . . . . . .

View 3 — same board rotated 45 degrees (each line is one down-left diagonal; reading aid only):
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              .   X   .
 4            .   .   O   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      O   .   .   .   .   .   .
 8    .   .   .   .   .   .   .   .
 9  .   .   .   .   .   .   .   .   .
10    .   .   .   X   .   .   .   O
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   X   .   .   .
14            .   .   .   .
15              .   .   .
16                .   .
17                  .

View 4 — same board rotated 45 degrees the other way (each line is one down-right diagonal; reading aid only):
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              O   .   .
 4            .   .   .   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      .   .   .   .   .   X   .
 8    .   .   .   .   X   .   .   .
 9  .   X   .   .   .   .   .   .   .
10    .   O   .   .   .   .   .   .
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   .   .   .   .
14            .   .   .   .
15              .   .   .
16                .   O
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the standard A-I / 1-9 coordinates.
```

### `flip+serial_dl`

```text
Match: Hand 1 of 1.
You are X.
The current position is shown in 2 views of the SAME board.

View 1 — same board with row labels flipped (9 at top, 1 at bottom):
   A B C D E F G H I
 9 . . . . . . O . .
 8 . X . . . . . . .
 7 . O . . . . . . .
 6 . . . . . . . . .
 5 . . . . . X . . .
 4 . . . . . . . X .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . O . . . . . . .

View 2 — same board rotated 45 degrees (each line is one down-left diagonal; reading aid only):
    A B C D E F G H I J K L M N O P Q
 1                  .
 2                .   .
 3              .   X   .
 4            .   .   O   .
 5          .   .   .   .   .
 6        .   .   .   .   .   .
 7      O   .   .   .   .   .   .
 8    .   .   .   .   .   .   .   .
 9  .   .   .   .   .   .   .   .   .
10    .   .   .   X   .   .   .   O
11      .   .   .   .   .   .   .
12        .   .   .   .   .   .
13          .   X   .   .   .
14            .   .   .   .
15              .   .   .
16                .   .
17                  .

[instruction, changed]
Respond with ONLY a coordinate for an empty cell, e.g. E5 (column letter A-I, row number 1-9). Think privately before you answer. Give your answer using the labels of View 1.
```

### `line_coach`

```text
[coaching line, extended]
Before you move, check whether you can make five, whether you must block the opponent's line, and how your own stones connect. Explicitly scan all four directions for lines of four — especially the two diagonals (down-right and down-left).

Match: Hand 1 of 1.
You are X.
   A B C D E F G H I
 1 . . . . . . O . .
 2 . X . . . . . . .
 3 . O . . . . . . .
 4 . . . . . . . . .
 5 . . . . . X . . .
 6 . . . . . . . X .
 7 . . . . . . . . .
 8 . . . . . . . . .
 9 . O . . . . . . .
```

**Factual notes:**

1. **Augmenting the original view helps; replacing it hurts.** Every single-view
   replacement loses (28.0%, 27.0%, 3.5%), while most `orig+X` combinations win.
   The win-rate ordering tracks the block-rate ordering, tying game outcomes to the
   defensive perception the probe experiments measured.
2. **`orig+stones` is the clear winner** (75.5%; wins from both seats: 39/50 as
   first player, 36.5/50 as second): the original board plus a plain coordinate
   list of the stones. It also had zero unparsed moves and the best block rate
   (93% vs 86%).
3. **`orig+serial_dl` confirms the probe prediction in real games** (59.0%, block
   88% vs 75%): adding the ↙-contiguous rotated view — the manipulation that
   reversed the diagonal miss gap in probes — wins games.
4. **Instruction-only coaching does nothing** (`line_coach` 48.0%): telling the
   model to scan all axes does not substitute for a representation it can read.
5. **Conflicting label systems hurt**: `orig+flip` (two contradictory row
   numberings side by side) scores 41.9%, worse than either rendering alone.
6. **`transpose` alone is catastrophic** (3.5%, 45 unparsed moves, block 26%):
   the model largely fails to read a column-major board.
7. **More views ≠ better**: `four_views` (47.5%) underperforms the best two-view
   prompts; view count seems to trade against clarity.

Caveats: one model (minimax-m2p7); 5 of 1,300 games lost to API errors (n column);
single game per seed per setting at temperature 0.6; win/loss in self-play compounds
all decision quality, not only threat defense; GLM-5.2 was attempted and abandoned
(request timeouts at 300s prevented any game from completing).

## Reproduce

```bash
python3 scripts/prompt_ab_games.py run --models minimax-m2p7 \
    --settings flip,serial_dl,serial_dr,transpose,orig+flip,orig+serial_dl,orig+serial_dr,orig+transpose,orig+stones,orig+flip+serial_dl,four_views,flip+serial_dl,line_coach \
    --games 100 --concurrency 64 --verbose      # resumable; re-run to fill gaps
python3 scripts/prompt_ab_games.py summary
```
