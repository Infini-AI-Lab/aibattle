# Hold'em Evaluation Modes — Implementation Plan

**Date:** 2026-05-31
**Scope:** Add two new Hold'em evaluation formats to AI Battle Arena —
*Heads-Up Match Mode* and *Multi-Agent Table Mode* — on top of the existing
single-hand engine.

---

## Summary

We will add two new poker evaluation modes. **Heads-Up Match Mode** groups many
hands into a stack-carryover match decided by match win rate; it is a small,
low-risk increment on the current engine. **Multi-Agent Table Mode** seats 4–6
agents at one table with side pots and a final ranking; it is a much larger lift
that requires generalizing the 2-player assumptions baked across the runner,
config, and eval layers, plus a new multi-way betting/side-pot engine.

The central design decision is to **model a match (and a table session) as a
`Game`**, not as a new orchestration layer above the runner. This reuses the
runner, logger, and evaluator unchanged for the heads-up case and keeps the
clean "the runner runs games" layering intact.

This plan reflects a full read of the codebase and an engine-correctness
verification (5000 fuzzed hands + scripted scenarios) completed on 2026-05-31.

---

## Background / Context

### What exists today

- The architecture is layered: **Game** (pure, immutable rules) → **Runner**
  (orchestrates the loop, seat-swap, invalid-action policy) → **Agent** →
  **Model**, with a pure **Eval** function over JSONL logs. See
  `claude-doc/DESIGN.md`.
- The load-bearing invariant: **one episode = one hand**, and **episodes are
  independent and pure**. The runner builds the full episode plan up front,
  draws deal seeds deterministically, and runs episodes **concurrently** under a
  semaphore (`src/aibattle/runner/runner.py`).
- `HoldemPoker` (`src/aibattle/games/holdem.py`) plays exactly one hand per
  episode; **stacks reset every hand** (`STARTING_STACK = 50`, blinds 1/2).
- `MatchContext` (`src/aibattle/types.py`) already carries cross-hand cumulative
  `standing` to agents — it was designed with multi-hand awareness in mind.
- The evaluator (`src/aibattle/eval/evaluator.py`) already aggregates poker
  metrics (bb/100, showdown/fold wins, CIs) by agent identity across seats.

### Engine verification (prerequisite, already done)

We verified the single-hand engine before planning anything on top of it:

- **No bugs found** in preflop/postflop action order, min-raise/min-bet
  validation, all-in reopening (short all-in does *not* reopen; full all-in
  does), or showdown/evaluator (category ordering, wheel, kickers, split pots).
- Zero-sum and chip conservation hold across 5000 randomized full hands.
- **Key finding:** `_refund_excess` (the uncalled-bet / all-in-for-less refund)
  is effectively *dead code in single-hand heads-up mode* — with equal 50-chip
  starts and per-street refunds, both stacks are always equal at the start of
  every street, so all-in-for-less can never occur. It **will fire constantly in
  Match Mode** once stacks carry over and effective stacks become unequal. The
  logic is already correct and verified on unequal-stack states, so the one
  piece of engine logic match mode most depends on is already in place.
- **Caveat:** the repo currently has **zero tests**. Engine correctness is
  unguarded against regression — addressed in Phase 0 below.

### Why match mode (an honest reframing)

The source design doc motivates Match Mode as *variance reduction* over
single-hand chip-delta. We should reframe this:

- Match win rate is **statistically less efficient** than mean chip delta
  (bb/100) for skill ranking — collapsing a match to win/loss/draw discards
  magnitude information that bb/100 uses. The evaluator already reports bb/100
  with confidence intervals.
- A single all-in still flips a *match* outcome, so match mode does not remove
  all-in variance; it changes the **objective** to stack-aware, survival/
  tournament play.
- Therefore Match Mode's real value is a **different, more realistic objective**
  (and watchability), not a cheaper skill estimator. We build it for that
  reason, accepting the statistical-efficiency trade-off.

This reframing drives a concrete parameter decision (see Phase 1, blinds/stacks).

---

## Guiding Design Decisions

1. **Match / table = a `Game`, not a new orchestration layer.** One episode
   becomes one whole match (or one whole table session). The existing runner's
   concurrency then applies at the match/session level — many matches in
   parallel — which is exactly what we want. Per-step logs still capture every
   hand's every decision, so hand-level metrics survive.

2. **Refactor the hand engine into a reusable core.** Extract the single-hand
   Hold'em state machine (betting rounds, all-in logic, showdown) from
   `holdem.py` into a hand-core that *both* the single-hand game and the
   match/table games drive. Avoids duplicating subtle betting logic.

3. **Tests before changes.** Land a regression suite around the current engine
   first (Phase 0), since the betting/all-in surface is exactly what match/table
   work will perturb.

4. **Heads-up first, table second.** Table mode depends on (a) generalizing the
   stack to N players everywhere and (b) a correct side-pot engine. Sequence it
   strictly after match mode.

5. **New modes are strictly additive — existing modes must not change
   (non-negotiable).** `kuhn_poker`, single-hand `holdem`, `connect4`, and
   `gomoku` must remain behaviorally identical (same outputs, same CLI, same
   logs). New modes are new `Game` classes under new registry names. The two
   places that touch shared code are isolated as follows:
   - **Hand-core refactor (Phase 1.1):** see the decision in Open Questions —
     either a shared core guarded by Phase 0 regression tests proving identical
     single-hand behavior, or full duplication so existing files are untouched.
   - **N-player generalization (Phase 2):** strictly backward-compatible. The
     2-player path stays a special case with identical behavior; the
     table-session `Game` handles multi-way forfeit/ranking internally so the
     runner's existing logic is unchanged for old games.
   - **Eval gating:** match/table metrics are emitted only for the new modes;
     single-hand `holdem`/Kuhn summaries are byte-for-byte unchanged.
   Phase 0's regression suite is the enforcement mechanism for this principle.

---

## Plan / Approach

### Phase 0 — Regression test harness (prerequisite)

Land the verification harness as the project's first tests so the engine is
guarded before we touch it.

- Create `tests/test_holdem.py` and `tests/test_poker_eval.py`.
- Port from the verification harness: evaluator category ordering + tiebreaks +
  wheel; preflop/postflop action order; BB option after a limp; min-raise/bet
  validation reason codes; **short vs full all-in reopening**; uncalled-bet
  refund on an unequal-stack state; and a fuzz loop asserting zero-sum, chip
  conservation, non-negative stacks, and self-generated-move validity.
- Wire `pytest` (already in the `dev` extra) into the workflow.

**Exit criteria:** `pytest` green; fuzz of ≥5000 hands clean.

### Phase 1 — Heads-Up Match Mode (`holdem_match` game)

Implement match mode as a new `Game` whose episode spans up to *N* hands with
carried-over stacks.

#### 1.1 Hand-core refactor
- Extract the hand state machine into a reusable component callable with an
  arbitrary `(stacks, button)` rather than always `STARTING_STACK`.
- Single-hand `HoldemPoker` becomes a thin wrapper over the core (behavior
  unchanged — Phase 0 tests must still pass).
- **Isolation note (per guiding decision #5):** if we want *zero* edits to
  existing files, the alternative is to put the engine in a new
  `holdem_core.py` that `holdem_match` uses while leaving `holdem.py` entirely
  untouched (accepting duplication). Decision tracked in Open Questions #5.

#### 1.2 `holdem_match` game state
State spans hands: per-player match stacks, `hand_index`, `max_hands`, button
schedule (alternates each hand), and the embedded current hand-core state.
- `initial_state`: equal starting stacks; derive each hand's deal from the
  episode seed deterministically (so a match is reproducible).
- `step`: delegates to the hand-core; when a hand ends, applies the chip delta to
  match stacks, rotates the button, and starts the next hand.
- `is_terminal`: true when a player busts **or** `max_hands` is reached.
- `returns`: match outcome. **Open question** below — chip differential vs.
  ±1/0 win/loss/draw.
- `current_player` / `observation` / `legal_actions`: delegate to the hand-core,
  but the observation's `rendered` text and `public` dict gain match context
  (hand number, max hands, both match stacks, chip lead/deficit, button).

#### 1.3 Agent context
Match-level context already has a home: extend `MatchContext` usage and the
Hold'em template (`src/aibattle/agents/templates/holdem.py`) to surface "this is
an N-hand match, stacks carry over, goal = finish ahead," current hand number,
and stack lead. No new cross-layer plumbing needed.

#### 1.4 Parameters (decision required — see Open Questions)
Defaults must make matches **decisive**, or stack-aware play and bust metrics
never trigger. Options: shorter effective stacks, more hands, and/or **blind
escalation**. A 100-chip stack at fixed 1/2 blinds over 20 hands will almost
always hit the hand cap near even — high per-match variance and no busts.

#### 1.5 Eval + metrics
- The evaluator already keys by agent identity and computes action/showdown
  stats from steps — those keep working because per-hand steps are still logged.
- Add **match-level** aggregation: match win rate (primary), matches
  won/lost/drawn, avg final stack, avg final stack diff, bust-out rate, avg hands
  per match. These derive from the episode (= match) summary records.
- Reuse the existing per-step metrics for hand-level stats (all-in/fold/raise
  frequency, invalid rates, showdown win rate).

**Exit criteria:** a `configs/holdem_match_*.yaml` runs end-to-end; match win
rate + hand-level metrics appear in `summary.json`; matches are decisive (busts
or clear stack leads occur at a reasonable rate).

### Phase 2 — Multi-player generalization (engine + framework)

This is the structural work that table mode needs **before** side pots. The
2-player assumption is hardcoded in several places that must be generalized —
**all changes strictly backward-compatible so the existing 2-player games are
unaffected** (guiding decision #5). Each item below keeps the 2-player path as a
behaviorally identical special case:

- **Config** (`src/aibattle/config/loader.py`): `_VALID_PLAYERS` is fixed to
  `player_0/player_1`. Support N seats.
- **CLI** (`src/aibattle/cli.py:119`): builds exactly two agents. Generalize to a
  list of players.
- **Runner forfeit logic** (`src/aibattle/runner/runner.py:244`): currently
  `opponent = the other player` and assigns `{forfeiter:-1, other:+1}` — a hard
  2-player notion. In a 4-handed pot a fold ≠ "everyone else +1". Forfeit must
  fold the player within the current hand, not award the whole match.
- **Winner vs. ranking** (`runner.py:250`): the runner emits a single `winner`.
  Multi-player needs a **ranking** derived from `returns`. Either extend the
  episode summary with a `ranking` field or compute it in eval.
- **Multi-way betting:** action order by seat position, button/blind rotation
  among non-busted players, correct round-closing with >2 players, and player
  statuses (active / folded / all-in / busted).

### Phase 3 — Side-pot engine (highest correctness risk)

Build side pots as a **standalone, pure, heavily unit-tested function** before
wiring to agents — a model agent will never tell you the chips landed in the
wrong stack.

- Signature (conceptual): `(contributions: dict[player,int], folded: set,
  showdown_rank: dict[player,key]) -> awards: dict[player,int]`.
- Logic: partition total contributions into contestable layers; each layer is a
  pot with an amount and an eligible set (contributors not folded); resolve main
  pot (smallest layer) first, then each side pot; split ties within a pot; return
  uncontested over-contribution to its owner.
- Test matrix: 3-way all-ins for different amounts; folded contributor still owes
  dead money to the main pot; ties across a pot; an all-in that wins only the
  main pot while a larger side pot goes to a deeper stack; odd-chip splits.

**Exit criteria:** dedicated `tests/test_side_pots.py` covering the matrix above;
property test that total awarded == total contributed (chip conservation).

### Phase 4 — Multi-Agent Table Mode (`holdem_table` game)

With Phases 2–3 in place, assemble the table session as a `Game` (episode = one
session).

- Fixed-hands table (not full sit-and-go) for predictable runtime; play until
  the hand cap or one player remains.
- Deal into non-busted players only; rotate button among them.
- Ranking: active players by final stack desc, busted players below (bust order
  as tiebreaker). If only one player has chips, they rank first.
- **Evaluation:** support *repeated table trials* with rotated/randomized seat
  assignments across seeds to reduce positional bias. Primary metric: average
  final rank; plus top-1 rate, avg final stack, bust-out rate.
- **Cost note:** hands within a table are sequential (cannot parallelize) and
  cost 4–6 model calls per decision. Parallelize **across** tables using the
  global-semaphore pattern already in `scripts/board_tournament.py`, and budget
  API spend explicitly.

---

## Open Questions

1. **Match-mode objective / `returns`.** Is the goal leaderboard skill ranking or
   realistic tournament play? If the former, reconsider whether match mode earns
   its complexity over running more hands and reporting bb/100 (already
   supported). If we proceed, should `returns` be ±1/0 (win/loss/draw) to make
   "match win rate" the literal payoff, or the chip differential (preserves
   magnitude, more efficient)? Recommendation: chip differential in the log,
   win/loss/draw derived in eval — keep both.
2. **Match parameters.** Fixed blinds or **blind escalation**? Stack depth and
   `max_hands`? Needs to be set so matches are decisive (busts/clear leads
   actually happen). This is a required design parameter, not "configurable
   later."
3. **Table size and elimination semantics.** Start at 4-handed before 6. Pure
   fixed-hands-then-rank for v0, or do we need real elimination/sit-and-go
   dynamics? (Busted-player ranking rules only matter if busts actually occur —
   loops back to the blind/stack question.)
5. **Hand-core: shared vs. duplicated.** To honor "new modes must not influence
   old modes," do we (a) extract a shared `holdem_core` and rewire single-hand
   `HoldemPoker` to it, guarded by Phase 0 tests that prove identical behavior
   (DRY, but edits a shared file); or (b) duplicate the engine into a new module
   so existing files are untouched (zero risk, some duplication)?
   Recommendation: (a) — the regression suite makes "identical behavior"
   verifiable, and duplication of subtle betting logic is its own long-term risk.
6. **Seat-swap generalization.** For matches, "duplicate" seat-swap means
   replaying the whole deal sequence with seats mirrored (variance reduction). Do
   we want this, and how does it interact with carryover stacks? For tables,
   prefer seat rotation across trials instead.

---

## Action Items

1. **Phase 0:** Land `tests/test_holdem.py` + `tests/test_poker_eval.py` from the
   verification harness; get `pytest` green. *(low risk, do first)*
2. **Decide Open Questions 1 & 2** (match objective + parameters) before building
   Phase 1, since they shape `returns` and defaults.
3. **Phase 1:** Refactor the hand-core; implement `holdem_match`; add match-level
   eval metrics; ship a `configs/holdem_match_*.yaml`.
4. **Phase 2:** Generalize config/CLI/runner/eval beyond two players.
5. **Phase 3:** Build + unit-test the standalone side-pot resolver.
6. **Phase 4:** Implement `holdem_table`, repeated-trial evaluation, and a
   parallel-across-tables tournament script.

---

## Roadmap at a glance

| Phase | Deliverable | Risk | Depends on |
|------|-------------|------|------------|
| 0 | Regression tests around current engine | Low | — |
| 1 | Heads-Up Match Mode (`holdem_match`) | Low–Med | 0 |
| 2 | N-player generalization (config/CLI/runner/eval) | Med | 0 |
| 3 | Side-pot engine (standalone + tested) | **High** | 0 |
| 4 | Multi-Agent Table Mode (`holdem_table`) | Med–High | 2, 3 |
