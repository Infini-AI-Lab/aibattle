# Hold'em Evaluation Modes — Implementation Plan

**Date:** 2026-05-31

## Summary

We are adding two new Texas Hold'em evaluation modes to AI Battle Arena:
**Heads-Up Match Mode** (two players, multi-hand match, match win rate) and
**Multi-Agent Table Mode** (N players at one table, ranking output with side
pots). The single load-bearing decision, now locked: **a match/table session is
modeled as one episode**, which is also the atomic saved/resumable unit. With
that framing the existing runner, concurrency, determinism, and per-episode
resume infrastructure are reused unchanged; all new work lives in the game
engine. Match Mode is a small, high-ROI addition to build first; Table Mode is a
genuine multi-player engine extension whose correctness hinges on side pots.

## Background / Context

The original single-hand Hold'em format treats each hand as an independent
episode (stacks reset every hand). This makes evaluation noisy: a single all-in
hand can dominate the aggregate chip delta. We want match-level and table-level
formats that reduce this variance and better resemble real poker competition.

A design doc (`claude-doc/HOLDEM_MODES_IMPLEMENTATION_PLAN.md`) specified the two
modes in detail but left one architectural question open: how a multi-hand match
maps onto the framework's `Game`/episode/runner model. This plan resolves that
and translates the design into concrete, sequenced engineering work.

## Locked Decision: match/table = one episode = the atomic saved unit

The framework's core invariant is that **episodes are independent and pure**, so
the runner plays them concurrently in any order, with per-decision deterministic
seeds, and persists/resumes one file per episode
(`runner/runner.py`, `episode_dir` → `ep<NNN>.json`, atomic temp+rename).

Carrying stacks across hands makes hands *sequential and stateful*, which is
incompatible with treating a hand as an episode. The resolution:

- **A match (Heads-Up) or a table session (Multi-Agent) is ONE episode.** The
  game's state spans all hands internally (stacks, hand counter, button, pots).
  `returns()` is computed at match/table granularity (win/loss or final-stack
  rank), not per hand.
- **The smallest saved/resumable unit is one full multi-hand game.** This maps
  directly onto the existing per-episode persistence: one `ep<NNN>.json` file =
  one complete match/table session. No new persistence code.
- **Tradeoff accepted (no per-hand checkpointing):** if a match is interrupted
  at hand 12 of 20, the whole match replays from hand 1. We lose at most one
  in-progress match per pairing/table, never a completed one. Hands within a
  match are stateful and cheap to redo together; persisting partial mid-match
  state would be far more complex for little benefit.

Consequences of this framing:
- Runner concurrency, `seat_swap`, deterministic seeding, per-episode resume,
  and the analysis pipeline all keep working for Match Mode with no changes.
- In-match context (current stacks, chip lead, hand number) is **legitimately
  deterministic** to expose in the prompt, because hands within a match run
  sequentially — unlike the cross-episode `standing` we had to suppress under
  parallel execution.

## Part I — Heads-Up Match Mode (build first)

A small, additive change that reuses the existing single-hand engine.

### Engine work
- New game `holdem_match` (e.g. `games/holdem_match.py`) implementing the `Game`
  ABC. Its state wraps the existing per-hand Hold'em logic as a sub-step and adds:
  carried-over stacks, hand counter, max-hands, alternating button.
- `initial_state(rng)` seeds both stacks equally; derives all per-hand deals from
  the single episode `rng` (determinism preserved).
- `is_terminal` when one player busts or max hands reached; `returns` = match
  result (win = +1 / loss = −1, or final stack diff — decide; recommend
  win/loss for the leaderboard metric, keep stack diff as secondary).
- Reuse existing `validate_action`, `fallback_action`, showdown/eval.
- Register in `games/registry.py`.

### Template work
- New `agents/templates/holdem_match.py` (subclass of the Hold'em template) that
  injects match context: hand number, max hands, own/opponent stack, chip lead,
  button. Reuse the existing tolerant parser (last-line-first, action-anchored
  amount). Register in `agents/templates/registry.py`.

### Tournament/metrics
- Tournament variant: round-robin pairs, each pair plays many matches
  (seat/role-rotated across matches for blind-order fairness). Primary metric:
  **match win rate**; secondaries per the design doc (final-stack diff, bust
  rate, hands/match, action frequencies, showdown win rate, invalid rate).
- Per-episode resume comes for free (episode = match).

### Honest framing note
Match Mode **reduces** all-in variance but does not eliminate it: a single big
all-in can still decide who leads at the final hand. What it fixes is
**magnitude → outcome** — a win counts the same regardless of margin, so one
lucky large pot can't dominate an aggregate. Consider pairing with an all-in EV
adjustment as an independent variance lever.

## Part II — Multi-Agent Table Mode (larger, sequence after Match Mode)

The bulk of the work and nearly all the correctness risk.

### Runner generalization (touches shared core — regression risk)
- The runner is currently hardwired to two agents: `run_match(agent_a, agent_b)`,
  `agents = {"player_0": p0, "player_1": p1}`, two-name `standing`, two-way
  `seat_swap`. Table Mode needs a path that takes a **list of N agents**.
- Generalize without breaking Kuhn/Hold'em/board (regression-test all games).

### Multi-player betting engine (rewrite, not extension)
- Current betting state (`last_raise_size`, `aggressor`, `acted_since`) is
  heads-up logic. N-player needs: table-position action order, correct
  round-ending with multiple players, re-opening rules when players are all-in
  for different amounts, per-player stacks, min-bet/min-raise enforcement.
- We already fixed a *heads-up* short-all-in reopen bug; the N-player version is
  materially harder.

### Side pots (mandatory; the #1 correctness hotspot)
- Track each player's total contribution; partition into contestable layers
  (main pot + side pots), each with an amount and an eligible-player set (folded
  excluded; all-in players eligible only up to their contribution).
- At showdown, award each pot independently among its eligible players; split on
  ties; return unmatched over-contribution.

### Ranking + context
- `returns()` yields per-player results for N; add a **ranking aggregator**
  (avg rank, top-1 rate, bust rate, avg stack) alongside the existing pairwise
  aggregation.
- N-agent prompt context: seat, button/blinds, all visible stacks/statuses, pot
  and side-pot info, amount to call, **per-decision min/max legal raise for this
  player's stack**, legal actions, public history. Never leak hole cards / undealt
  cards / seed.

## Milestones

1. **Single-hand stability** — already in place (hand engine, betting, showdown,
   logging).
2. **Heads-Up Match Mode** — `holdem_match` game + template + tournament variant +
   match-win-rate metrics. *(Small; do first.)*
3. **Multi-player hand engine** — N seats, button/blind rotation, multi-player
   action order, player statuses, all-in handling. *(Includes runner
   generalization.)*
4. **Side-pot engine** — contribution layers, eligibility, independent awards,
   splits, with a dedicated fuzz/property test. *(Correctness-critical.)*
5. **Multi-Agent Table Mode** — fixed-hands table, bust handling, ranking,
   repeated table trials, table-level metrics.

## Risks

- **Side-pot correctness (highest):** code that looks right and silently
  misallocates chips. Mitigate with property tests (below).
- **Multi-player reopen logic:** subtle with mixed all-in amounts.
- **Runner generalization regressions:** the N-agent path touches the loop every
  game uses; regression-test Kuhn/Hold'em/board.
- **Resume granularity:** a match/table is the atom; an interrupt loses the
  in-progress one (accepted).

## Testing

- **Match Mode:** determinism (same seed → same match outcome), stack carry-over,
  termination (bust vs max-hands), button alternation, resume (run twice → second
  loads completed matches, no rewrite).
- **Side pots (property/fuzz, first-class deliverable):**
  - *Chip conservation:* Σ pot awards == Σ contributions every hand.
  - *Eligibility:* folded players never win; all-in players eligible only up to
    their contribution; over-contribution returned.
  - *Splits:* tie remainders handled deterministically.
- **Multi-player betting fuzz:** random legal action sequences never deadlock or
  produce illegal states; round ends correctly with all-ins.

## Open Questions

- Match result for `returns()`: pure win/loss vs final-stack margin (recommend
  win/loss primary, margin secondary).
- Table size for v0: start 4-player before 6.
- Whether to combine Match Mode with an all-in EV adjustment now or later.
- Default match length (20 hands) and table length (20–30) — confirm.

## Action Items

1. Update the design doc to state the locked framing: **atomic persisted unit =
   one full multi-hand game (match = episode)**.
2. Implement Heads-Up Match Mode (Milestone 2) once the running board tournament
   completes, so the live run is not disrupted.
3. Add Match Mode tests (determinism, carry-over, termination, resume).
4. Scope Table Mode as its own project with side-pot fuzz testing as a
   first-class deliverable; sequence Milestones 3 → 4 → 5.
