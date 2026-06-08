# Implementation Plan: Four New Games for AI Battle Arena (Leduc Poker, Independent Blackjack, Othello-lite 6x6, Repeated Colonel Blotto)

## Goal Description

Add four new playable game environments to the AI Battle Arena framework, each implemented as an independent module that mirrors the existing game/template/registry patterns, then validate correctness with `gpt-oss-120b` and run a multi-model comparison experiment over the Fireworks API, storing all trajectories and summaries.

The four games:
1. **Leduc Poker** (`leduc_poker`) — imperfect-information 2-player poker; collects bluffing / belief-update / multi-round betting trajectories.
2. **Independent Blackjack** (`independent_blackjack`) — one LLM agent vs a fixed mechanical dealer; collects risk-calibration / basic-strategy trajectories. Agent-vs-environment, modeled as a 2-seat game where seat `player_1` is a built-in (non-LLM) dealer agent.
3. **Othello-lite 6x6** (`othello_lite_6x6`) — perfect-information 2-player board game; collects legal-move / mobility / long-horizon-planning trajectories.
4. **Repeated Colonel Blotto** (`repeated_colonel_blotto`) — simultaneous-allocation 2-player game over 20 rounds; collects resource-allocation / opponent-modeling trajectories. Simultaneity is emulated sequentially with hidden pending allocations.

Each game is self-contained: a new game module under `src/aibattle/games/`, a new prompt template under `src/aibattle/agents/templates/`, registry entries, optional new built-in agents, per-game tests, and per-game configs/scripts. The existing shared framework (`types.py`, the runner, the evaluator) is **not** modified.

A design constraint discovered during planning: the runner is strictly sequential turn-based (exactly one seat acts per step, both seats are agents, exactly two seats for `run_match`). All four designs are shaped to fit this constraint without touching shared framework code.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. "PASS" tests describe behavior that must hold; "FAIL" tests describe inputs/behaviors that must be rejected or must not occur.

- AC-1: **Each of the four games is registered and instantiable through the public factory.**
  - Positive Tests (expected to PASS):
    - `make_game("leduc_poker")`, `make_game("independent_blackjack")`, `make_game("othello_lite_6x6")`, `make_game("repeated_colonel_blotto")` each return a `Game` instance whose `name` matches the key.
    - `available_games()` includes all four new names.
    - `make_template(<name>)` returns a `GameTemplate` for each of the four names; `make_template(<name>, coached=True)` injects a non-empty coaching line.
  - Negative Tests (expected to FAIL):
    - `make_game("leduc")` (wrong key) raises `ValueError`.
    - `make_template("repeated_blotto")` (wrong key) raises `ValueError`.

- AC-2: **Othello-lite 6x6 implements correct Reversi rules with pass and double-pass termination.**
  - AC-2.1: Initial board and turn order.
    - Positive: `initial_state` yields the 6x6 board with the four center pieces `W B / B W` on the central 2x2 (rows 3-4, cols C-D), all other cells empty; `current_player` is Black and Black moves first.
    - Negative: An initial board with five or more occupied cells, or with White to move first, is rejected by the corresponding test.
  - AC-2.2: A legal move flips at least one opponent piece in one or more of the eight directions.
    - Positive: From the initial state, each legal move returned by `legal_actions` flips ≥1 opponent piece when applied by `step`, and the flipped pieces change owner.
    - Negative: A move onto an occupied cell, or onto an empty cell that flips zero opponent pieces, is rejected by `validate_action`.
  - AC-2.3: Pass is legal only when the current player has no flipping move.
    - Positive: When the current player has at least one real move, `legal_actions` does not include `pass`; when the current player has zero real moves, `legal_actions` is exactly `["pass"]` and `step` applied to `pass` switches the turn without changing the board.
    - Negative: `validate_action` rejects `pass` when at least one real move exists.
  - AC-2.4: The game terminates when both players have no legal move (two consecutive passes / full board), and the winner is the player with more pieces.
    - Positive: A constructed end position where both players must pass is `is_terminal`; `returns` gives `+1`/`-1` by piece count, or `0`/`0` on an equal count.
    - Negative: A position where the current player still has a real move is not `is_terminal`.

- AC-3: **Independent Blackjack implements correct hit/stand/double play against a fixed dealer policy with correct scoring.**
  - AC-3.1: Seats and turn flow.
    - Positive: `players == ["player_0", "player_1"]`; `player_0` (the LLM seat) acts until it stands, busts, or doubles; `current_player` returns `player_1` (the dealer) only when the dealer must draw; the dealer's hidden card is revealed only at the dealer's turn.
    - Negative: After the player busts (including a hit-bust or a double-bust), the hand is already terminal and `current_player` does NOT advance to the dealer; a test asserting the dealer drew after a player bust fails.
  - AC-3.2: Dealer policy (built-in agent).
    - Positive: A `blackjack_dealer` built-in agent, registered in the agent registry, returns `hit` while its ace-aware total is below 17 and `stand` at any total ≥ 17, including standing on soft 17; it never calls an LLM.
    - Negative: A dealer that hits on soft 17, or that consults a model client, fails the policy test.
  - AC-3.3: Scoring, with naturals resolved before ordinary totals (two-card naturals only).
    - Positive: Normal win `+1`, normal loss `-1`, push `0`, double win `+2`, double loss `-2`, player natural (two-card 21) only `+1.5`, player-and-dealer both natural `0` (push), dealer natural only `-1`. `returns` is zero-sum: `player_1 == -player_0`.
    - Negative: A three-card 21 scored as a natural (`+1.5`), or `sum(returns) != 0`, or a double-then-bust scored as anything other than `-2`/`+2`, fails the scoring test.
  - AC-3.4: Blackjack configs disable seat swap and rely on a dedicated analysis script (not plain `aibattle eval`) for standings.
    - Positive: The shipped blackjack configs set `seat_swap: false`; a dedicated blackjack analysis script reports only the `player_0` (LLM) profit and excludes the dealer seat from rankings.
    - Negative: A blackjack config with `seat_swap: true` is flagged by the plan as incorrect usage; the dealer agent appearing in a model ranking table produced by the dedicated script fails the test.

- AC-4: **Leduc Poker implements correct two-round betting and showdown with zero-sum payoffs.**
  - AC-4.1: Deck, deal, public card, and betting structure.
    - Positive: Deck is `J J Q Q K K` (K>Q>J); each player antes 1 and receives one private card; after the first betting round one public card is revealed; a second betting round follows; legal actions are drawn from `fold/check/call/bet/raise` with at most one raise per round; bet sizes are fixed per street (round 1 = 2, round 2 = 4); `amount` means total street commitment (raise-to semantics, matching Hold'em).
    - Negative: A second raise within the same betting round is rejected by `validate_action`; a `bet`/`raise` whose `amount` is below the fixed street minimum or above the stack is rejected.
  - AC-4.2: Showdown ordering.
    - Positive: A private card pairing the public card beats any non-pair hand; otherwise the higher private card wins; equal strength splits the pot.
    - Negative: A non-pair higher card scored as beating a pair-with-board hand fails the showdown test.
  - AC-4.3: Zero-sum integer payoffs including split pots.
    - Positive: `returns` always satisfies `sum(returns) == 0` and integer chip accounting; on a split, the deterministic odd-chip rule awards any single odd chip to `player_0` while preserving zero-sum.
    - Negative: A terminal state whose `returns` do not sum to zero, or a split that loses or duplicates a chip, fails the payoff test.

- AC-5: **Repeated Colonel Blotto implements 20 rounds of simultaneous allocation, emulated sequentially with hidden pending allocations.**
  - AC-5.1: Setup and round flow.
    - Positive: 20 rounds, 100 resources per round, 5 battlefields valued `[1,2,3,4,5]`; within a round `player_0` submits an allocation first, then `player_1`; the round resolves on the second submission; per battlefield the higher allocation wins that battlefield's value; ties score nothing; scores accumulate; the higher cumulative score after all rounds wins.
    - Negative: A round that resolves before both allocations are submitted, or a battlefield value list other than `[1,2,3,4,5]`, fails the flow test.
  - AC-5.2: Allocation encoding and validation (no `Move`/framework change).
    - Positive: An allocation is encoded in `Move.type` as the string `"alloc:a,b,c,d,e"` (five non-negative integers summing to 100), with `amount` left `None`; `validate_action` accepts any well-formed allocation string, not only the default; `legal_actions` returns a list whose entries are all themselves valid `Move`s (it returns the valid default `["alloc:20,20,20,20,20"]`).
    - Negative: An allocation summing to ≠100, or containing a negative number, or with other than five components, is rejected by `validate_action`; `types.py`'s `Move` definition is unchanged (no new vector field).
  - AC-5.3: Hidden-information leakage is prevented before round resolution.
    - Positive: `player_1`'s observation (`private`, `public`, `history`, `rendered`) does not contain `player_0`'s pending allocation for the current unresolved round; only resolved prior-round allocations appear; the persisted trajectory and transcript for the pre-resolution step likewise do not leak the pending allocation.
    - Negative: A test that finds `player_0`'s current-round allocation anywhere in `player_1`'s observation or in the pre-resolution persisted record fails.
  - AC-5.4: A dedicated `blotto_random` built-in agent enables random-vs-random smoke runs.
    - Positive: A registered `blotto_random` agent samples a valid random allocation (five non-negative integers summing to 100) seeded deterministically from `decision_seed`, producing varied allocations across decisions; a random-vs-random match runs to completion with a low invalid-action rate.
    - Negative: An agent that can only emit the single default allocation, or that produces an allocation not summing to 100, fails the smoke test.

- AC-6: **Each game runs end-to-end through the runner and produces clean, recordable trajectories.**
  - Positive Tests (expected to PASS):
    - For each of the four games, a random-vs-random (or LLM-vs-dealer, for Blackjack) match runs to completion via the runner and writes a JSONL log plus the configured trajectory/transcript artifacts.
    - The recorded trajectory for each game contains the draft-mandated content: for Leduc — private observations, public-state transitions, betting history, legal actions, raw + parsed actions, fallback events, showdown result, chip movement; for Blackjack — player hand, dealer upcard, revealed hidden dealer card, legal actions, decisions, card draws, terminal outcome, hand profit; for Othello — full board per turn, current player, legal moves, chosen move, flipped pieces, pass events, final board, winner; for Blotto — round number, battlefield values, both allocations, per-battlefield outcomes, round scores, cumulative scores, invalid/fallback events, final winner.
  - Negative Tests (expected to FAIL):
    - A run that crashes, hangs, or yields a trajectory missing any of the mandated fields for its game fails.
    - An illegal action that is neither resolved by `fallback` nor recorded as an invalid/fallback event fails.

- AC-7: **`gpt-oss-120b` plays each game correctly through its template.**
  - Positive Tests (expected to PASS):
    - For each game, a small `gpt-oss-120b` run (e.g. 10 episodes; LLM-vs-dealer for Blackjack) completes with the model's outputs parsed into legal moves at a high rate and a low invalid-action rate, with logs/transcripts written.
  - Negative Tests (expected to FAIL):
    - A template whose `parse` cannot recover a legal move from typical `gpt-oss-120b` output (high invalid rate) fails this criterion.

- AC-8: **A multi-model Fireworks comparison experiment runs across the four available models and stores results.**
  - Positive Tests (expected to PASS):
    - For the agent-vs-agent games (Othello, Leduc, Blotto), a round-robin across the four verified Fireworks models (`kimi-k2p6`, `deepseek-v4-pro`, `glm-5p1`, `gpt-oss-120b`) runs with per-episode resume, writing trajectories + summaries under `runs/` and an aggregated report under `reports/`.
    - For Blackjack, each of the four models independently plays N hands against the dealer; a dedicated analysis script produces a per-model profit comparison.
  - Negative Tests (expected to FAIL):
    - An experiment configuration that references an unavailable model id (`minimax-m2p7` or `deepseek-flash`) fails, since only the four verified models are in scope per the resolved decision.
    - A tournament run that cannot resume after interruption (loses completed episodes) fails.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

All four games fully implemented as independent modules with: correct rules per the draft (and the clarifications in this plan), prompt templates with parse + repair + coaching, registry wiring, the two new built-in agents (`blackjack_dealer`, `blotto_random`), per-game pytest coverage (initial state, legal actions, step transitions, terminal detection, zero-sum/sign-correct returns, and a hidden-information leakage test for Blackjack and Blotto), random-vs-random smoke configs, `gpt-oss-120b` correctness configs, per-game tournament scripts with per-episode resume, a dedicated Blackjack analysis script, and a stored four-model comparison experiment with aggregated reports. No shared framework files are modified.

### Lower Bound (Minimum Acceptable Scope)

All four games implemented with correct core rules satisfying AC-1 through AC-6, each registered and runnable via the runner, with random-vs-random (or LLM-vs-dealer) smoke validation and the minimum tests that prove rule correctness, zero-sum payoffs, and hidden-information enforcement. `gpt-oss-120b` correctness validation (AC-7) is included. The large four-model comparison (AC-8) may be reduced in episode count but must still run and store results for all four games.

### Allowed Choices

- Can use: the existing `Game` / `GameTemplate` / `Agent` interfaces and registries; the existing runner, logger, evaluator, model client, and config loader unchanged; Hold'em-style `amount = total street commitment` semantics for Leduc; encoding the Blotto allocation vector inside `Move.type` as a string; new built-in agents registered in `_BUILTINS`; `pytest`; new YAML configs and new tournament/analysis scripts mirroring existing ones.
- Cannot use: any modification to `src/aibattle/types.py` (the `Move` dataclass keeps `type` + optional `amount` only — no vector field); any modification to the runner's turn loop, the evaluator, or the config loader; any LLM call inside a built-in dealer/random agent; the unavailable Fireworks model ids `minimax-m2p7` and `deepseek-flash`.

> **Note on Deterministic Designs**: The draft fixes most game rules exactly (deck composition, dealer policy, scoring values, battlefield values, round/resource counts, board size, initial position). For these, the path boundaries converge — the implementation must match the draft and the clarifications in this plan, with no design latitude. Latitude remains only in non-rule areas (prompt wording, parser robustness, test breadth, experiment episode counts).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

**Othello-lite 6x6** (cleanest fit; build first to establish the pattern):
- State: a 6x6 immutable board (tuple of tuples), current player, and a `last_was_pass` flag.
- `legal_actions(state, player)`: scan empty cells, compute flips in eight directions; return coordinate strings (`"A1".."F6"`); if none, return exactly `["pass"]`.
- `step`: place the piece and flip captured runs; on `pass`, switch player and set the pass flag; clear the flag on a real move.
- `is_terminal`: both players have no real move (equivalently two consecutive passes / full board).
- `returns`: piece count → `+1`/`-1`/`0`.
- `fallback_action`: prefer a corner, then the first legal move.

**Independent Blackjack** (asymmetric, modeled as two seats):
- Built-in `blackjack_dealer` agent reads the observation and returns `hit`/`stand` from an ace-aware total (ace counts 11 unless it busts; stand on all 17s including soft 17). No model client.
- `current_player`: `player_0` until it stands/busts/doubles; then `player_1` only while the dealer must draw.
- Resolve naturals (two-card 21) before ordinary totals; double draws exactly once then stands; double-then-bust is immediately terminal at `-2`/`+2` (dealer does not draw).
- `returns` zero-sum: `player_1 = -player_0`. Configs set `seat_swap: false`. A dedicated analysis script reports only the LLM seat.

**Leduc Poker** (reuse Hold'em betting idioms):
- `amount` = total street commitment (raise-to). Fixed bet size per street (round 1 = 2, round 2 = 4); ante 1; at most one raise per round.
- `validate_action` mirrors Hold'em amount checks (minimum / stack bounds) but with the simpler Leduc structure.
- Showdown: pair-with-board > non-pair; else higher private; equal → split. Compute `returns` from contributed chips so they always sum to zero; on a split, award any odd chip deterministically to `player_0`.

**Repeated Colonel Blotto** (simultaneous via sequential hidden submission):
- Within a round, `player_0` submits first; store the allocation as a PRIVATE pending value invisible to `player_1`. `player_1` submits; `step` resolves the round.
- Encode allocations in `Move.type` as `"alloc:a,b,c,d,e"`. `legal_actions` returns the valid default `["alloc:20,20,20,20,20"]` (every entry is a valid move); `validate_action` parses and accepts any well-formed allocation (five non-negative ints summing to 100).
- `blotto_random` built-in samples a valid random allocation seeded from `decision_seed` so generic random play explores the space.
- Observation for `player_1` exposes only resolved prior-round allocations (for opponent modeling), never the current pending one. Verify no leakage into persisted records.

### Relevant References

- `src/aibattle/games/base.py` — the `Game` interface to implement.
- `src/aibattle/games/kuhn.py` — minimal discrete-game reference (state, legal actions, returns, observation).
- `src/aibattle/games/connect4.py`, `src/aibattle/games/board.py` — board-game patterns and fallback bias (Othello reference).
- `src/aibattle/games/holdem.py`, `src/aibattle/agents/templates/holdem.py` — amount/raise-to semantics, validate_action amount checks, split-pot returns, action+amount parsing (Leduc reference).
- `src/aibattle/games/registry.py`, `src/aibattle/agents/templates/registry.py` — `_GAMES`, `_TEMPLATES`, `_COACHING` registration.
- `src/aibattle/agents/base.py`, `src/aibattle/agents/random_agent.py`, `src/aibattle/agents/heuristic_agent.py`, `src/aibattle/agents/board_agents.py`, `src/aibattle/agents/registry.py` — built-in agent interface (`async act`, `decision_seed` RNG) and `_BUILTINS` registration (dealer / blotto_random reference).
- `src/aibattle/agents/templates/base.py`, `src/aibattle/agents/templates/connect4.py` — template structure and last-line-then-whole-text parsing.
- `src/aibattle/runner/runner.py` — sequential turn loop, `seat_swap`, standings aggregation.
- `src/aibattle/eval/evaluator.py` — name-based aggregation (and why Blackjack needs a dedicated script instead).
- `src/aibattle/config/loader.py` — accepted YAML keys for configs.
- `scripts/kuhn_tournament.py`, `scripts/board_tournament.py`, `scripts/analyze_*` — tournament + analysis script patterns with per-episode resume.
- `configs/deepseek_vs_oss.yaml` — model-vs-model config template (Fireworks).

### Conceptual Approach — Experiment

- Verified Fireworks model ids (this account): `accounts/fireworks/models/kimi-k2p6`, `accounts/fireworks/models/deepseek-v4-pro`, `accounts/fireworks/models/glm-5p1`, `accounts/fireworks/models/gpt-oss-120b`.
- Agent-vs-agent games: round-robin over the four models (C(4,2)=6 pairs), `seat_swap: true`, per-episode resume, store under `runs/<game>_tournament/` and aggregate into `reports/`.
- Blackjack: each model independently plays N hands vs the dealer; the dedicated analysis script produces a per-model profit table.

## Dependencies and Sequence

### Milestones

1. **Milestone A — Othello-lite 6x6** (establishes the end-to-end pattern; least coupling).
   - Phase A: Game module + template + registry entries + coaching line.
   - Phase B: Tests (rules, pass/double-pass, returns) + random-vs-random smoke config + `gpt-oss-120b` config.
   - Review gate before proceeding (per the resolved per-game review cadence).

2. **Milestone B — Independent Blackjack** (introduces a built-in environment agent).
   - Phase A: Game module (hands, dealer turn flow, scoring) + `blackjack_dealer` built-in agent + registry entries.
   - Phase B: Template + tests (scoring incl. naturals/double/soft-17, dealer policy) + LLM-vs-dealer smoke + `gpt-oss-120b` config + dedicated analysis script.
   - Review gate.

3. **Milestone C — Leduc Poker** (reuses Hold'em betting idioms).
   - Phase A: Game module (deal, two-round betting, public card, showdown, zero-sum returns) + registry entries.
   - Phase B: Template (action + amount parse/repair) + tests (betting limits, showdown ordering, split-pot zero-sum) + smoke + `gpt-oss-120b` config.
   - Review gate.

4. **Milestone D — Repeated Colonel Blotto** (the only design with a non-standard action encoding; build last).
   - Phase A: Game module (sequential hidden submission, allocation encoding, round resolution, cumulative scoring) + `blotto_random` built-in agent + registry entries.
   - Phase B: Template (allocation parse/repair) + tests (validation, hidden-info leakage, cumulative scoring) + random-vs-random smoke + `gpt-oss-120b` config.
   - Review gate.

5. **Milestone E — Multi-model experiment and storage** (after all four games validated).
   - Step 1: Per-game tournament/analysis scripts (mirroring existing ones, with per-episode resume).
   - Step 2: Run the four-model round-robin for Othello/Leduc/Blotto and the four-model independent runs for Blackjack.
   - Step 3: Aggregate reports; store all trajectories + summaries under `runs/` and `reports/`.

Dependencies: Milestones A–D are independent of each other in code (separate files) but are sequenced for review cadence; A is built first to establish the pattern. Each game's tournament/analysis (Milestone E) depends on that game passing its own validation (Milestones A–D) and on `gpt-oss-120b` correctness (AC-7).

## Task Breakdown

Each task includes exactly one routing tag: `coding` (implemented by Claude) or `analyze` (executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Implement Othello-lite 6x6 game module (board, legal moves/flips, pass, double-pass termination, returns, fallback) | AC-2 | coding | - |
| task2 | Implement Othello template (coord/pass parse, repair, coaching) + register game & template | AC-1, AC-2 | coding | task1 |
| task3 | Othello tests + random-vs-random smoke config + gpt-oss config; run smoke and gpt-oss validation | AC-2, AC-6, AC-7 | coding | task2 |
| task4 | Implement Independent Blackjack game module (hands, dealer turn flow, naturals/double/soft-17 scoring, zero-sum returns) | AC-3 | coding | - |
| task5 | Implement `blackjack_dealer` built-in agent (ace-aware policy, no LLM) + register agent, game, template | AC-3, AC-1 | coding | task4 |
| task6 | Blackjack template + dedicated analysis script (LLM-seat-only) + tests + LLM-vs-dealer & gpt-oss configs; run validation | AC-3, AC-6, AC-7 | coding | task5 |
| task7 | Implement Leduc Poker game module (deal, two-round betting w/ one raise, public card, showdown, zero-sum split-pot returns) | AC-4 | coding | - |
| task8 | Implement Leduc template (action+amount parse/repair, raise-to wording) + register game & template | AC-4, AC-1 | coding | task7 |
| task9 | Leduc tests (betting limits, showdown ordering, zero-sum split) + smoke + gpt-oss config; run validation | AC-4, AC-6, AC-7 | coding | task8 |
| task10 | Implement Repeated Colonel Blotto game module (sequential hidden submission, alloc encoding in Move.type, round resolution, cumulative scoring, hidden-info enforcement) | AC-5 | coding | - |
| task11 | Implement `blotto_random` built-in agent + Blotto template (alloc parse/repair) + register agent, game, template | AC-5, AC-1 | coding | task10 |
| task12 | Blotto tests (validation, hidden-info leakage, cumulative scoring) + random-vs-random smoke + gpt-oss config; run validation | AC-5, AC-6, AC-7 | coding | task11 |
| task13 | Per-game tournament scripts (round-robin for Othello/Leduc/Blotto; independent runs for Blackjack) with per-episode resume | AC-8 | coding | task3, task6, task9, task12 |
| task14 | Run four-model Fireworks experiment; aggregate reports; store trajectories + summaries under runs/ and reports/ | AC-8 | coding | task13 |
| task15 | Independent expert review of the four game-rule implementations (showdown/scoring/flip/allocation correctness and hidden-info) before the large experiment | AC-2, AC-3, AC-4, AC-5 | analyze | task3, task6, task9, task12 |

## Claude-Codex Deliberation

### Agreements

- Othello and Leduc fit the existing `Move(type, amount)` model cleanly and stay fully isolated in new files.
- Blackjack fits as a two-seat zero-sum game with a scripted built-in dealer, with no framework change.
- Blotto can avoid all framework changes by encoding the allocation vector in `Move.type` as a string; a vector field on `Move` is not required.
- No shared framework changes (`types.py`, runner, evaluator, config loader) are needed for any of the four games.
- Each game should ship random-vs-random smoke tests, illegal-action validation tests, terminal-return tests, and (for hidden-information games) a transcript/observation leakage test.

### Resolved Disagreements

- **Blotto `legal_actions` contents**: Codex flagged that returning a bare `["allocate"]` token would make `validate_action` reject every pick by `random_agent` and harness agents. Resolution: `legal_actions` returns the valid default `["alloc:20,20,20,20,20"]` (every entry is itself a valid move) while `validate_action` accepts any well-formed allocation string; additionally a dedicated `blotto_random` built-in samples valid random allocations so random-vs-random smoke runs explore the space. Rationale: keeps the "pick from legal_actions" contract sound without changing `Move`.
- **Blackjack double-then-bust**: Codex required explicit handling. Resolution: a double-then-bust (or any player bust) makes the hand immediately terminal at `-2`/`+2` (or `-1`/`+1` for a non-double bust); the dealer does not draw and `current_player` does not advance. Rationale: avoids an incorrect extra dealer turn and matches casino rules.
- **Blackjack naturals**: Resolution: two-card naturals are resolved before ordinary totals — player+dealer natural is a push (`0`), player-only natural is `+1.5`, dealer-only natural is `-1`. Rationale: correct precedence prevents mis-scoring a natural against an ordinary 21.
- **Blackjack soft 17**: Resolution: dealer total is ace-aware (ace = 11 unless it busts) and the dealer stands on all 17s including soft 17. Rationale: matches the draft's stated dealer policy.
- **Leduc split-pot zero-sum**: Codex required integer-safe zero-sum. Resolution: compute `returns` from contributed chips so they always sum to zero; on a split, award any single odd chip deterministically to `player_0`. Rationale: guarantees `sum(returns) == 0` with integer accounting and a single deterministic rule (not alternatives).
- **Othello termination**: Resolution: terminal when both players have no real move (two consecutive passes / full board); `pass` is legal only when the current player has zero flips. Rationale: covers all end conditions in a runner that always asks `current_player` to act.
- **Blackjack standings**: Codex noted plain `aibattle eval` would aggregate the dealer as an agent and mislead. Resolution (per user decision): Blackjack uses a dedicated analysis script reporting only the LLM seat; plain `aibattle eval` is documented as not meaningful for Blackjack standings; no evaluator framework change. Rationale: keeps the framework untouched while producing correct per-model results.

### Convergence Status

- Convergence rounds executed: 2 (second-pass review + revised-plan re-review).
- Final Status: `converged` — after the revisions, Codex reported no remaining `REQUIRED_CHANGES` and no high-impact `DISAGREE`. The two items Codex left as user decisions were resolved by the user (see Pending User Decisions, all `RESOLVED`).
- Note on confidence: cross-review was performed with Codex (`gpt-5.5`, high effort) over two rounds; the first Codex attempt failed on auth and was retried successfully after the user fixed it. Full cross-review confidence applies.

## Pending User Decisions

All decisions below were resolved during planning; none remain `PENDING`.

- DEC-1: Handling of the two requested-but-unavailable Fireworks models (`minimax-m2p7`, `deepseek-flash`).
  - Claude Position: Use only the four verified-available models; do not reference unavailable ids.
  - Codex Position: User decision — substitute (e.g. `kimi-k2p5`) or reduce to the verified four.
  - Tradeoff Summary: Substituting keeps the model count higher but mixes in a model the user did not request; reducing keeps the roster faithful to what is actually available.
  - Decision Status: **RESOLVED — use only the four available models (`kimi-k2p6`, `deepseek-v4-pro`, `glm-5p1`, `gpt-oss-120b`); no substitution.**

- DEC-2: Whether plain `aibattle eval` should be made dealer-aware for Blackjack.
  - Claude Position: Keep the framework untouched; ship a dedicated Blackjack analysis script and document that plain `aibattle eval` standings do not apply to Blackjack.
  - Codex Position: User decision — dedicated-script-only is acceptable if documented; making plain eval dealer-aware is a framework change.
  - Tradeoff Summary: A dedicated script avoids touching shared code (lower risk) but means plain eval is misleading for Blackjack; a framework change is more general but affects all games.
  - Decision Status: **RESOLVED — dedicated analysis script, no framework change.**

- DEC-3: Implementation / review cadence after this plan.
  - Claude Position: Either per-game review gates or one combined review.
  - Codex Position: N/A — process question.
  - Tradeoff Summary: Per-game review catches issues earlier; combined review is faster overall.
  - Decision Status: **RESOLVED — per-game stop-and-review in the order Othello → Blackjack → Leduc → Blotto.**

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Phase", "Step", "task<N>", or similar workflow markers. These belong in this plan document only.
- Use descriptive, domain-appropriate naming in code (e.g. `dealer_should_hit`, `resolve_round`, `flip_in_direction`, `allocation_is_valid`).
- Mirror the style, docstring conventions, and immutability discipline of the existing game/template/agent modules. State objects are immutable frozen dataclasses; `step` returns new state and never mutates its argument.
- Keep each game self-contained; do not modify `src/aibattle/types.py`, the runner, the evaluator, or the config loader.

### Branch

- Development proceeds on the `new-game` branch (already checked out from the latest `main`).

--- Original Design Draft Start ---

# Additional Game Designs for AI Battle Arena

This document summarizes four additional games proposed for AI Battle Arena: Leduc Poker, Independent Blackjack, Othello-lite 6x6, and Repeated Colonel Blotto.

The current goal is to define playable environments and collect clean trajectories. Detailed game-specific analysis and metrics can be added later after enough trajectories are collected.

## 1. Leduc Poker

### Type

Imperfect-information poker game. Two players.

### Purpose

Leduc Poker is an intermediate poker game between Kuhn Poker and Texas Hold'em. It is useful for collecting trajectories involving bluffing, hidden-information reasoning, public-card belief updates, and multi-round betting, while remaining much simpler than full Hold'em.

### Rules

- Deck: J, J, Q, Q, K, K.
- Card strength: K > Q > J.
- Each player antes and receives one private card.
- First betting round.
- One public card is revealed.
- Second betting round.
- If no player folds, showdown happens.

### Actions

The game supports fold, check, call, bet, and raise. For v0, each betting round can be limited to at most one raise to keep the betting logic simple.

### Showdown

- A private card that pairs with the public card beats any non-pair hand.
- If neither player has a pair, the higher private card wins.
- If both players have equivalent hand strength, the pot is split.

### Trajectory Content

A trajectory should record private observations, public state transitions, betting history, legal actions, raw agent outputs, parsed actions, fallback events if any, showdown results, and final chip movement.

## 2. Independent Blackjack

### Type

Risk and probability game. One agent plays independently against the dealer in each evaluation run.

### Purpose

Independent Blackjack evaluates risk calibration, probability reasoning, rule following, and basic strategy. Agents do not directly compete with one another. Each agent independently plays many hands against a fixed dealer policy, and the trajectories can later be analyzed across agents.

### Rules

- Each hand starts with a player hand and a dealer upcard.
- The agent acts until it stands, busts, or doubles.
- The dealer follows a fixed policy.
- Profit is recorded for the hand.
- Each new hand is independent.

### Actions

The v0 action set supports hit, stand, and double. Split, surrender, and insurance are out of scope for the first version.

### Dealer Policy

Dealer hits until 17 and stands on both hard 17 and soft 17.

### Scoring

- Normal win: +1.
- Normal loss: -1.
- Push: 0.
- Double win or loss: +/-2.
- Blackjack: +1.5.

### Trajectory Content

A trajectory should record the player hand, dealer upcard, hidden dealer card when revealed, legal actions, agent decisions, card draws, terminal outcome, and hand profit. Future analysis can be performed from these trajectories without requiring metrics to be built into the first implementation.

## 3. Othello-lite 6x6

### Type

Perfect-information board game. Two players.

### Purpose

Othello-lite is a 6x6 version of Othello/Reversi. It is intended to collect trajectories involving legal move generation, board evaluation, mobility, corner and edge control, and long-horizon planning. It is more strategic than Connect Four but simpler than Chess or Go.

### Board

The game uses a 6x6 board. Rows are numbered 1 to 6 from top to bottom, and columns are labeled A to F from left to right.

### Initial Board

```
. . . . . .
. . . . . .
. . W B . .
. . B W . .
. . . . . .
. . . . . .
```

Black moves first.

### Rules

- A player places one piece on a legal empty cell.
- A legal move must flip at least one opponent piece.
- Flips can happen in eight directions: horizontal, vertical, and diagonal.
- If a player has no legal move, that player passes.
- If both players cannot move, the game ends.
- The player with more pieces at the end wins.

### Trajectory Content

A trajectory should record the full board state at each turn, current player, legal moves, chosen move, flipped pieces, pass events, final board, and winner.

## 4. Repeated Colonel Blotto

### Type

Strategic allocation game. Two players. Simultaneous actions.

### Purpose

Repeated Colonel Blotto tests resource allocation, numerical reasoning, opponent modeling, adaptation, memory, and risk diversification. The repeated format is preferred because it allows agents to observe and respond to opponent allocation patterns over time.

### Default Setup

- Rounds: 20.
- Resources per round: 100.
- Battlefields: 5.
- Battlefield values: [1, 2, 3, 4, 5].

### Round Flow

- Each agent secretly allocates 100 resources across 5 battlefields.
- Allocations must use non-negative integers and sum to 100.
- For each battlefield, the higher allocation wins that battlefield value.
- Ties give no score in the v0 version.
- Scores accumulate across rounds.
- The player with the higher cumulative score after all rounds wins.

### Trajectory Content

A trajectory should record round number, battlefield values, both submitted allocations, battlefield-level outcomes, round scores, cumulative scores, invalid allocation or fallback events, and final winner.

## Summary

| Game | Category | Main Purpose | Interaction |
| --- | --- | --- | --- |
| Leduc Poker | Imperfect information | Bluffing, betting, belief update | Agent-vs-agent |
| Independent Blackjack | Risk / probability | Basic strategy and risk calibration | Agent-vs-environment |
| Othello-lite 6x6 | Perfect information | Board planning and mobility | Agent-vs-agent |
| Repeated Colonel Blotto | Strategic allocation | Resource allocation and adaptation | Simultaneous agent-vs-agent |

These games should first be implemented as clean trajectory-collection environments. Later analysis can use the collected trajectories to derive outcome, validity, behavior, and diagnostic metrics.

--- Original Design Draft End ---
