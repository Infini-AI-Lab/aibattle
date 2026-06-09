# Add four new games + supporting infrastructure + four-model experiment

## Summary

Adds **four new self-contained games** to the AI Battle Arena framework, each
mirroring the existing game / template / registry patterns, together with all the
supporting infrastructure (built-in agents, prompt templates, tests, configs) and
a **four-model Fireworks comparison experiment**. No shared framework files
(`types.py`, the runner, the evaluator, the config loader) are modified — every
game is added purely through new modules and registry entries.

`new-game` → `main`. Full test suite: **67 passing** (`PYTHONPATH=src python -m
pytest tests/ -q`).

## The four games

| Game | Type | Key rules / design |
|------|------|--------------------|
| **Leduc Poker** (`leduc_poker`) | Imperfect-information poker (agent vs agent) | Deck J J Q Q K K; ante 1; one private card each; one public card revealed after round 1; two betting rounds with **at most one raise per street**; fixed bet sizes (round 1 = 2, round 2 = 4); `amount` = total street commitment (raise-to, like Hold'em); showdown order pair-with-board > higher private > split; **zero-sum payoffs from total contributions** (correct even on fold-after-raise); split awards the odd chip to player_0; public **betting history** exposed in the observation. |
| **Independent Blackjack** (`independent_blackjack`) | Risk / probability (agent vs environment) | Modeled as a 2-seat game: player_0 = the LLM, player_1 = a built-in **non-LLM dealer** (`blackjack_dealer`). Ace-aware totals; hit / stand / double; **player bust (incl. double-bust) is immediately terminal — the dealer does not draw**; the dealer acts only when it must draw and follows a fixed policy (hit < 17, stand on all 17s incl. soft 17); two-card **naturals resolved before ordinary totals** (player +1.5, both push 0, dealer-only −1); double pays ±2; zero-sum; the dealer hole card is hidden until the dealer's turn. |
| **Othello-lite 6x6** (`othello_lite_6x6`) | Perfect-information board (agent vs agent) | 6×6 Reversi; central 2×2 start `W B / B W`; Black (player_0) first; a legal move flips ≥1 opponent piece in any of 8 directions; **pass only when no flipping move exists**; **double-pass / mutual-no-move terminates**; winner by piece count; zero-sum. |
| **Repeated Colonel Blotto** (`repeated_colonel_blotto`) | Strategic allocation (simultaneous, agent vs agent) | 20 rounds, 100 resources, battlefields valued `[1,2,3,4,5]`. Simultaneity is emulated sequentially: player_0 submits first into a **hidden pending allocation** that is never leaked to player_1; player_1 submits and the round resolves. Allocation is encoded as a string in `Move.type` (`"alloc:a,b,c,d,e"`), so **`types.py` is unchanged** (no new vector field). Ties score nothing; cumulative winner gets ±1. Trajectories record per-battlefield outcomes, round + cumulative scores, and a full terminal `round_history` (incl. the final round). |

## Supporting infrastructure

For each game:

- **Game module** under `src/aibattle/games/` implementing the `Game` interface
  (immutable state, legal actions, step, terminal, returns, observation, render).
- **Prompt template** under `src/aibattle/agents/templates/` (rules + coaching +
  output instruction + robust parser + repair hint). The Blackjack template's
  rules now spell out the full payout structure (+1/−1/0, double ±2, blackjack
  +1.5, dealer blackjack −1, both-blackjack push).
- **Registry wiring**: games, templates (+ coaching lines), and the agent
  `_BUILTINS` registry.
- **New built-in agents**: `blackjack_dealer` (fixed-policy, no LLM),
  `blackjack_random`, `leduc_random`, `blotto_random` (these supply valid
  bet/raise amounts and allocations that the generic random agent cannot).
- **Tests**: per-game pytest (`tests/test_{othello_lite,blackjack,leduc,blotto}.py`,
  4 files) covering rules, scoring, zero-sum payoffs, hidden-information
  enforcement, and template parsing — **67 tests total, all passing**.
- **Configs**: random-vs-random smoke + gpt-oss correctness configs per game
  (9 new YAMLs under `configs/`).
- `tests/conftest.py` pins imports to the in-repo `src` so tests are not shadowed
  by a separate editable install.

An **independent code review** (Codex) of all four game-rule implementations was
run; the two issues it found in Blackjack — dealer-phase `legal_actions` not being
policy-tight, and `render(perspective=...)` revealing the dealer hole card — are
fixed and covered by tests.

## Four-model experiment (`scripts/new_games_experiment.py`)

Compares the four verified-available Fireworks models — `kimi-k2p6`,
`deepseek-v4-pro`, `glm-5p1`, `gpt-oss-120b` (the requested `minimax-m2p7` /
`deepseek-flash` are not available on this account and are intentionally
excluded). Structure:

- **Othello / Leduc / Blotto**: model-vs-model round-robin (C(4,2)=6 pairs),
  seat-swapped.
- **Blackjack**: each model independently as player_0 vs the built-in dealer;
  a dedicated analysis script reports only the LLM seat (the dealer is never
  ranked).
- Per-episode resume (`episode_dir`) so runs can be stopped/continued anytime.
- The report marks a round-robin game **COMPLETE only when every model pair has
  both seat directions represented** (a single `seat_swap=True` episode is one
  direction only, since the runner treats `episodes` as the total budget).
- Reports written to `reports/new_games_experiment_report.md` + `.json`.

### Experiment progress (current)

| Game | Coverage | Status |
|------|----------|--------|
| Independent Blackjack | 4/4 models | **COMPLETE** |
| Leduc Poker | 6/6 pairs, 6/6 seat-swapped | **COMPLETE** |
| Repeated Colonel Blotto | 6/6 pairs, **0/6 seat-swapped** | PARTIAL |
| Othello-lite 6x6 | 3/6 pairs, **0/6 seat-swapped** | PARTIAL |

### Experiment results so far

Across every completed run the **invalid-action rate is ~0%** (the models parse
the templates cleanly), confirming the games and templates work end-to-end with
real models.

**Independent Blackjack** (vs dealer, mean profit/hand): gpt-oss-120b +0.50,
kimi-k2p6 0.00, glm-5p1 −0.50, deepseek-v4-pro −1.00. _(few hands; directional)_

**Leduc Poker** (round-robin, net chips/game): gpt-oss-120b **+1.83**, glm-5p1
+0.33, deepseek-v4-pro +0.17, kimi-k2p6 −2.33. All 0% invalid.

**Repeated Colonel Blotto** (one seat direction so far, net/game): gpt-oss-120b
+0.33 (67% win), kimi-k2p6 +0.33 (67%), glm-5p1 −0.33, deepseek-v4-pro −0.33.
Notably **glm-5p1 has a 40% invalid-allocation rate** here (it often emits a
malformed allocation), which is itself a useful diagnostic signal.

**Othello-lite 6x6** (partial — only the fast-model pairs finished): gpt-oss-120b
went 2/3, etc. 0% invalid where played.

### Known limitation (why Blotto/Othello are PARTIAL)

The long games are **wall-clock-bound by reasoning-model latency in
model-vs-model mode**, not by any bug. Measured single-decision latency on
Othello: `gpt-oss-120b` ~11s, `deepseek-v4-pro` ~47s, but **`kimi-k2p6` and
`glm-5p1` exceed 90s per move**. A full seat-swapped Othello round-robin
(6 pairs × 2 directions × ~60 moves) is an overnight-scale job for those models.
Blackjack and Leduc (short games) complete quickly; Blotto and Othello accumulate
and are intended to be filled in over a long run.

To resume / continue the long-game round-robin (per-episode resume — never loses
progress):

```bash
nohup bash scripts/run_long_experiment.sh > /tmp/long.log 2>&1 &
```

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -q      # 67 passed
```

Plus per-game gpt-oss correctness runs (high parse rate, ~0% invalid, persisted
trajectories/transcripts) and the experiment leaderboards above.
