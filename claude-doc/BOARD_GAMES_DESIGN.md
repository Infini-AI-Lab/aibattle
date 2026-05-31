# Connect Four & Gomoku-Lite — Concrete Design Plan

Engineering companion (like `DESIGN.md` for Kuhn and `HOLDEM_DESIGN.md` for
Hold'em) for adding two perfect-information board games to AI Battle Arena.
Where this disagrees with the high-level write-ups, this document wins for v0.

These two games broaden the arena from *imperfect-information gambling* (Kuhn,
Hold'em) to *perfect-information planning*. They need **no protocol change** —
both are discrete, single-token actions with no amount — so they ride the
existing `Move(type=..., amount=None)` path. The only framework additions are a
game-defined **fallback hook** and an **entropy source for evaluation**
(perfect-information + deterministic models = identical replays without one).

---

## 0. Decisions locked for v0

| # | Question | Decision |
|---|----------|----------|
| 1 | Action space | **Discrete tokens, no amount.** Connect Four: a column id; Gomoku: a coordinate like `F5`. Reuses the existing action path; no `amount`, no new validation machinery. |
| 2 | Outcome / `returns` | **Win/loss/draw → +1 / −1 / 0** (zero-sum). Draws are real and common (esp. Connect Four); the runner already maps equal returns to `winner=None`. |
| 3 | Invalid-move fallback | **Center-biased, game-defined.** Generalize the runner's fixed fallback into `Game.fallback_action(state, player, legal)`. Board games override: Connect Four → center column else nearest-to-center legal; Gomoku → center cell else nearest empty. Poker games keep the old check→fold→call order. |
| 4 | Determinism / replay variety | **Seeded random openings.** Both games play `random_open` forced-random legal plies at the start of each episode (driven by the per-episode RNG). Without this, two temp-0 models replay the identical game. Default: Connect Four `random_open=2`, Gomoku `random_open=2`. |
| 5 | Board sizes | Connect Four **6×7**; Gomoku **9×9**. |
| 6 | Win conditions | Connect Four: **4 in a row**; Gomoku: **5 in a row** (H/V/both diagonals). |
| 7 | Gomoku forbidden-move (overline / 3-3 / 4-4) rules | **Disabled in v0.** |
| 8 | Coordinate system (Gomoku) | Columns `A–I` (left→right), rows `1–9` (top→bottom). A cell is `<col letter><row number>`, e.g. `E5` = center. |
| 9 | Observation | **Full board for both players** (perfect information) — rendered as an ASCII grid plus structured cells + legal moves. |

---

## 1. Framework fit — reused vs new

**Reused unchanged:** `Game`/`Agent` ABCs, the `Move` action protocol (amount
stays `None`), runner loop, seat-swap, parallel execution + shared semaphore,
invalid-action policy mechanics, JSONL logging, trajectories, transcripts,
per-decision seeding, model client, config loader, CLI (`run`/`eval`),
`--rerun`, per-run dirs, human play, terminal colors.

**New (small, general):**
1. **`Game.fallback_action(state, player, legal) -> str`** — default returns the
   current runner order (`check→fold→call→first legal`); the runner calls this
   instead of hard-coding the order. Board games override for center bias. This
   is the right home for "what to do on an illegal move" anyway.
2. **`random_open` game param** — N forced-random opening plies in
   `initial_state(rng)`, the entropy source for evaluation (see §6).
3. Two game modules, a shared line-detector, two templates, two baselines, and
   board-game eval metrics.

**No change to:** `AgentResponse`, `validate_action` (default membership check
suffices — a move is legal iff its token is in `legal_actions`), the runner's
core loop.

---

## 2. Shared board helpers (`games/board.py`)

```python
def connects(grid, r, c, player, need) -> bool:
    """True if the piece at (r,c) completes a line of >= `need` for `player`,
    scanning the 4 axes (horizontal, vertical, both diagonals) through (r,c)."""

def is_full(grid) -> bool: ...
def render_grid(grid, symbols) -> str:   # ASCII board, shared by both games
    ...
```

`connects` checks only lines through the last move (cheap, O(need)). Correctness
here is the bug-prone core, so it ships with an exhaustive test file: each axis,
both diagonals, edge/corner placements, lines that must *not* wrap across an
edge, and full-board draws.

---

## 3. Connect Four (`games/connect4.py`)

### Parameters
- Board **6 rows × 7 columns**; players `player_0` (e.g. `X`), `player_1` (`O`).
- `random_open` default 2; `version = "1.0.0"`.

### State (immutable)
```
Connect4State(
  grid: tuple of 6 rows x 7 cols (None | "player_0" | "player_1"),
  to_act: PlayerId,
  last: Optional[(r, c)],   # last move, for win check
  winner: Optional[PlayerId],
  done: bool,
)
```

### Contracts
- `legal_actions(state, _) -> [col]` — columns (as strings `"0".."6"`) whose top
  cell is empty.
- `step(state, move)` — drop into `move.type`'s column: piece lands at the lowest
  empty row; recompute `connects(...)` for the win; set `done` on win or full board.
- `validate_action` — default (token ∈ legal_actions).
- `fallback_action` — center column `"3"` if legal, else nearest legal column to
  center.
- `returns` — winner +1 / loser −1; draw 0/0.
- `observation` — own/opponent symbol, the rendered grid, legal columns, rule and
  output-format text. (Perfect info: board is fully shared.)

### Render
```
 0 1 2 3 4 5 6
 . . . . . . .
 . . . . . . .
 . . . . . . .
 . . . X . . .
 . . O X . . .
 . O X O . . .
You are X. Legal columns: 0,1,2,3,4,5,6. Drop a piece: reply with a column number.
```

---

## 4. Gomoku-Lite (`games/gomoku.py`)

### Parameters
- Board **9×9**; black `player_0`, white `player_1`; win = **5 in a row**.
- Forbidden-move rules off; `random_open` default 2; `version = "1.0.0"`.

### State (immutable)
```
GomokuState(grid: 9x9, to_act, last, winner, done)
```

### Contracts
- `legal_actions(state, _) -> [coord]` — every empty cell as a coordinate string
  (`"A1".."I9"`). ~81 entries early; large but fine to list.
- `step(state, move)` — parse `move.type` → (row, col), place stone, check
  `connects(..., need=5)`, set `done` on win or full board.
- `validate_action` — default membership (coordinate must be a legal empty cell).
- `fallback_action` — center `"E5"` if empty, else nearest empty cell to center.
- `returns` — +1 / −1 / 0.
- `observation` — symbol, rendered grid with row/col labels, legal-move *rule*
  ("any empty cell, e.g. F5") rather than dumping all 81 tokens into the render
  (the structured `legal_actions` list still carries them for non-LLM agents).

### Render
```
   A B C D E F G H I
 1 . . . . . . . . .
 ...
 5 . . . . X . . . .
 ...
You are X (black). Place a stone on any empty cell, e.g. E5. Win: 5 in a row.
```

Coordinate parsing is tolerant: case-insensitive, accepts `e5`, `E5`, `E-5`.

---

## 5. Templates (`agents/templates/{connect4,gomoku}.py`)

Generic model agent + per-game template, as for the poker games.
- `render_prompt`: rules + win condition + the rendered board + match context +
  legal-move description + strict output format (Connect Four: "reply with one
  column number"; Gomoku: "reply with one coordinate like F5"; "put the move on
  the last line if you reason first").
- `parse(raw, request)`: tolerant extraction →
  - Connect Four: first integer that is a legal column → `Move(type=str(col))`.
  - Gomoku: first `<letter><number>` token that maps to a legal empty cell →
    `Move(type="E5")`. Returns `None` (→ repair → fallback) if nothing legal.
- `repair_prompt`: restate the legal options / coordinate format.

Note: Gomoku's full legal list is long; the prompt states the *rule* and the
board, and validates the parsed coordinate against `legal_actions`, rather than
enumerating 81 tokens in the instruction text.

---

## 6. Determinism & the entropy source (the key eval decision)

Perfect information + `temperature: 0` ⇒ a model's move is a deterministic
function of the board, so **two such models replay the identical game every
time**. Reps add nothing. We need injected variety:

**v0 choice: seeded random openings.** `initial_state(rng)` plays `random_open`
uniformly-random legal plies before the agents take over. Because the per-episode
RNG is seeded from the deal seed:
- different episodes → different openings → a *distribution* of fair positions;
- `seat_swap` reuses the seed with swapped seats → the same opening played from
  both sides (duplicate-style fairness, and it neutralizes first-move advantage,
  which matters: Connect Four is a first-player forced win with perfect play).

`random_open` is a game param, so a match can also be run at `random_open=0` for
a single canonical game, or higher for more diversity. Temperature > 0 remains an
orthogonal option but is not required.

---

## 7. Evaluation metrics (`eval` extensions)

Poker's VPIP/aggression don't apply; board games stamp `episode_metadata` with
`reason` (`win`/`draw`) and the game name, enabling:

- **Win / draw / loss rate** per model (primary), with CI.
- **First-mover win rate** — did moving first (or exploiting Connect Four's
  first-player edge) convert?
- **Invalid-move rate** — expected to be higher for Gomoku (LLMs misread grids);
  a genuine capability signal, not noise.
- **Tactical accuracy** (computable from the board — the standout metric):
  - **win-take rate**: when an immediate winning move existed, did it play one?
  - **block rate**: when the opponent had an immediate winning threat (an open
    `need-1` line), did it block?
  These are objective "correct vs blunder" rates measuring threat detection.
- Avg game length (plies), center-preference, decision latency.

The evaluator computes win-take/block by replaying each decision point: scan
legal moves for an immediate win for the mover (was one available? taken?) and
for the opponent on the prior ply (was a block needed? made?).

---

## 8. Baselines (`agents/`)

- **RandomBoardAgent** — uniform over legal moves (per-decision seeded). Works
  for both games.
- **Connect4HeuristicAgent / GomokuHeuristicAgent** — a tactical rule set: (1)
  take an immediate win if available; (2) else block an immediate opponent win;
  (3) else prefer center / extend own longest line. Non-trivial calibration
  baselines and a sanity check for the win-take/block metrics.

---

## 9. Config examples

```yaml
# Connect Four: model vs heuristic
game:
  name: connect4
  params: { random_open: 2 }
players:
  player_0: { agent: { type: model, name: gpt-oss-120b, model: {...} } }
  player_1: { agent: { type: builtin, name: connect4_heuristic } }
run: { episodes: 50, seed: 7, seat_swap: true, max_concurrency: 16 }
output: { dir: ./runs/c4_oss_vs_heur, save_full_log: true, save_summary: true,
          save_transcripts: true }
```

Gomoku is identical with `name: gomoku`. Human play works out of the box
(`type: human`) — the numbered menu lists legal moves; for Gomoku the human types
a coordinate (the menu may be summarized for large boards).

---

## 10. Build order

1. `games/board.py` (`connects`, render) + exhaustive line-detection tests.
2. **`Game.fallback_action` hook** + retrofit the runner to call it (poker
   behavior unchanged).
3. `games/connect4.py` + registry + template + parser; random/heuristic
   baselines → first no-API end-to-end match. **Milestone: a logged Connect Four
   game with center-fallback and win/draw outcomes.**
4. `random_open` entropy + board-game eval metrics (win/draw, win-take, block).
5. `games/gomoku.py` + template + coordinate parsing + baselines.
6. Small Fireworks model smoke runs for each; then optional multi-model board-game
   tournament reusing `scripts/tournament.py` (swap the game name).

Connect Four first (smallest, fastest, validates the whole perfect-info path);
Gomoku second (coordinate parsing, bigger board, harder spatial reasoning).

---

## 11. Risk register

- **Win detection** (`connects`) — diagonals and edge non-wrap are the classic
  bugs; covered by tests.
- **Determinism** — without `random_open` the eval is a single game per ordering;
  make sure it defaults on for tournaments.
- **Gomoku grid parsing by models** — expect elevated invalid-move rates; the
  center-fallback keeps games valid, and the rate itself is a reported metric.
- **Coordinate ambiguity** — fix the system (col=letter A–I, row=number 1–9, top
  row = 1) and state it explicitly in the prompt and render.

---

## 12. Deferred past v0

Larger Gomoku (15×15), forbidden-move rules (overline / 3-3 / 4-4), swap/swap2
openings, draw-by-agreement, >2 players, and any search/solver baselines beyond
the simple tactical heuristic.
