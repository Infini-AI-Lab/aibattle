# Heads-Up Texas Hold'em Lite — Concrete Design Plan

This is the engineering companion to the Hold'em Lite high-level doc, in the
same spirit as `DESIGN.md` for Kuhn Poker. It pins down the one framework change
the game requires (typed actions with amounts + engine-side validation), the
game module's internal contracts, the hand evaluator, the agent template, the
evaluation metrics, and the build order. Where this disagrees with the
high-level doc, this document wins for v0 implementation details.

The headline: Kuhn validated the framework on a *discrete* action space. Hold'em
is the first game with a *numeric* action space (agent-chosen bet/raise
amounts), so it forces one deliberate, backward-compatible extension to the
agent/game protocol. Everything else reuses the existing layers unchanged.

---

## 0. Decisions locked for v0

Answers to the high-level doc's open questions (§23 there), plus the new ones
this game raises. Settled so implementation can proceed.

| # | Question | Decision |
|---|----------|----------|
| 1 | How are bet/raise amounts represented? | **`amount` = the player's TOTAL committed chips for the current street** (raise-to, not raise-by). Integer chips. |
| 2 | Protocol change for amounts? | Add **`amount: Optional[int]`** to `AgentResponse` and to the structured action passed to the game. Fold/check/call leave it `None`. |
| 3 | Who validates amounts? | **The game**, via a new `Game.validate_action(state, player, action)`. The runner calls it instead of string membership. Kuhn reimplements it trivially. |
| 4 | Invalid action / amount handling? | Reuse the existing runner policy. Fallback order: **check → fold → call → first legal type**. All fallbacks are amount-free. Every invalid event logged with raw output + reason. |
| 5 | All-in a separate action? | **Yes.** `all_in` is its own action type (no amount needed; commits the whole stack). |
| 6 | Fold when no bet is facing? | **Illegal.** Folding a free check is strictly dominated; map such output to fallback (check). |
| 7 | Raise caps? | **Uncapped re-raises up to stack.** With a 50bb stack heads-up it self-limits; no per-street cap in v0. |
| 8 | Min-raise rule? | A raise-to must be **≥ current bet + last raise increment** (≥ current bet + big blind if no prior raise this street). An all-in *below* a full raise does **not** reopen betting for the player who already acted. |
| 9 | Button assignment? | **Alternate deterministically** by episode parity. Composes with `seat_swap` (same deal, swapped seats = duplicate poker). |
| 10 | Hand evaluator? | **Internal 7-card evaluator**, validated by an exhaustive test suite. No new dependency. |
| 11 | Observation style? | **Natural-language render + structured numeric fields** (call amount, min/max raise), as Kuhn does. |
| 12 | Uncalled bets? | The **uncalled portion of a bet/all-in is returned** to the bettor so chip deltas sum to zero. |

---

## 1. The protocol change (the only framework-level change)

### Before (Kuhn): discrete, stringly-typed
- `legal_actions(state, player) -> list[Action]` returns concrete actions.
- `AgentResponse.action: str`.
- runner validates with `action in legal_actions`.
- `Game.step(state, action: str)`.

### After: typed actions + amounts + engine-side validation
- `legal_actions(state, player) -> list[str]` returns legal **action types**
  (`"fold"`, `"check"`, `"call"`, `"bet"`, `"raise"`, `"all_in"`). (Kuhn already
  effectively does this.)
- **`AgentResponse` gains `amount: Optional[int] = None`.** This is a public
  protocol change: the external HTTP agent contract gains an optional
  `"amount"` field in its JSON response. Backward compatible — existing agents
  that omit it are unaffected.
- **New `Game.validate_action(state, player, action) -> (ok: bool, reason: Optional[str])`.**
  Full validation including amount range/integrality. The runner's
  `resolve_action` calls this rather than membership-testing a string.
- **`Game.step` takes a structured action** (a small `Action` dataclass with
  `type: str` and `amount: Optional[int]`) instead of a bare string.

### `types.py` additions
```python
@dataclass(frozen=True)
class Action:
    type: str                 # "fold"|"check"|"call"|"bet"|"raise"|"all_in"
    amount: Optional[int] = None   # total street commitment for bet/raise

@dataclass
class AgentResponse:
    action: str                    # action TYPE (unchanged field name/role)
    amount: Optional[int] = None   # NEW: required for bet/raise
    message: Optional[str] = None
    raw_output: Optional[str] = None
    metadata: dict = ...
```

The runner assembles an `Action(type=response.action, amount=response.amount)`,
asks the game to validate it, applies the invalid-action policy on failure, and
passes the (possibly fallback) `Action` to `step`. Kuhn's `validate_action`
ignores `amount`; its `step` accepts an `Action` and reads only `.type`.

This keeps poker specifics entirely inside the game module — the runner stays
game-agnostic.

---

## 2. Game module (`games/holdem.py`)

### 2.1 Parameters (v0 defaults, later configurable)
- Players: 2 (`player_0`, `player_1`)
- Starting stack: **50** chips each (reset every episode)
- Small blind: **1**, Big blind: **2**
- Standard 52-card deck; 2 hole cards each; flop 3 / turn 1 / river 1
- One hand per episode; integer chips
- `version = "1.0.0"`

### 2.2 State (opaque dataclass)
```
HoldemState(
  button: PlayerId,                 # SB/button this hand
  deck: tuple,                      # remaining undealt cards (hidden)
  hole: {player: (card, card)},     # hidden per player
  board: tuple,                     # revealed community cards (0,3,4,5)
  street: "preflop"|"flop"|"turn"|"river"|"showdown"|"done",
  stacks: {player: int},            # chips behind
  street_commit: {player: int},     # chips in THIS street's pot
  pot_settled: int,                 # chips from previous streets
  to_act: PlayerId,
  last_raise_size: int,             # increment of the last raise (for min-raise)
  aggressor: Optional[PlayerId],    # last player to bet/raise this street
  acted_since_aggressor: set,       # who has acted since the last aggression
  all_in: {player: bool},
  result: Optional[dict],           # filled at terminal: winner, reason, deltas
)
```
State is immutable; `step` returns a new state. Cards are dealt from a
per-episode shuffled deck seeded by the deal seed (reproducible).

### 2.3 Positions & action order
- Button posts SB (1); other player posts BB (2).
- **Preflop**: button/SB acts first. BB has the option (can raise after a call).
- **Postflop**: BB acts first.
- Button alternates by episode parity; with `seat_swap` the same deal is
  replayed with seats swapped (duplicate poker).

### 2.4 Legal action types (`legal_actions`)
Let `to_call = max(street_commit.values()) - street_commit[player]`.
- **No bet facing** (`to_call == 0`): `["check", "bet", "all_in"]`
  (bet only if stack ≥ min bet; all_in if stack > 0).
- **Bet facing** (`to_call > 0`): `["fold", "call", "raise", "all_in"]`
  (raise only if stack allows ≥ a min-legal raise; call becomes an all-in call
  if `to_call ≥ stack`).
- A player with 0 chips (all-in) cannot act; the engine skips them.

### 2.5 Amount semantics & validation (`validate_action`)
`amount` = player's **total** `street_commit` after the action.
- **bet**: legal only if `to_call == 0`. Requires
  `min_bet ≤ amount ≤ street_commit[player] + stack`, where
  `min_bet = street_commit[player] + big_blind` (i.e. bet ≥ one BB).
- **raise**: legal only if `to_call > 0`. Requires
  `amount ≥ current_bet + last_raise_size` (min-raise) and
  `amount ≤ street_commit[player] + stack` (max = all-in).
- **call**: no amount; moves `min(to_call, stack)` (all-in if short).
- **check**: no amount; legal only if `to_call == 0`.
- **fold**: no amount; legal only if `to_call > 0`.
- **all_in**: no amount; commits entire remaining stack.

Invalid examples (each → recorded invalid, then fallback): missing/non-integer/
negative amount, below min, above stack, raise without a facing bet, bet facing
a bet, check facing a bet, call with nothing to call, fold when checkable.

The engine **never silently clamps** an illegal amount — it records it invalid
and applies the fallback policy (§4). Exception: a *legal* call/all-in that
exceeds stack is the normal short-stack case, not an error.

### 2.6 Betting-round termination
A street's betting closes when either:
- a player folds (hand ends), or
- both players are all-in (run out the board), or
- the action returns to the last aggressor with all bets matched, **or** (no
  aggression) both players have acted and `to_call == 0` — with the preflop
  caveat that the **BB's option** must have been offered.

Implementation tracks `aggressor` + `acted_since_aggressor`: the round ends when
every non-all-in active player has acted since the last aggression and all
street commitments are equal.

### 2.7 Showdown & payouts (`returns`)
- If a player folds: the other wins the pot; uncalled chips returned first.
- If the river round closes with both active: **showdown** — best 5-of-7 per
  player via the hand evaluator (§3); higher wins, equal splits the pot.
- `returns` = net chip delta vs the 50-chip start, keyed by player, **summing
  to zero**. The terminal `result` records `winner`, `reason`
  (`"fold"`/`"showdown"`/`"all_in_showdown"`), and per-player delta.

### 2.8 Observation (`observation`) — hidden info preserved
Structured fields + a natural-language `rendered` string. Includes: player,
position (button/BB), street, **own hole cards**, visible board, pot
(`pot_settled + sum(street_commit)`), own & opponent stacks, own & opponent
street contributions, `to_call`, legal action **types**, **min/max legal
amount** for bet/raise when applicable, public betting history, all-in flags,
and output-format requirements. **Never** exposes opponent hole cards, the deck,
or undealt board cards.

---

## 3. Hand evaluator (`games/poker_eval.py`)

Internal 7-card evaluator. Input: 7 cards (2 hole + 5 board). Output: a
comparable rank key such that stronger hands compare greater, with correct
tie-breaking.

- Categories (high→low): straight flush, quads, full house, flush, straight,
  trips, two pair, pair, high card. (Royal flush is just the top straight
  flush.)
- Returns `(category_rank, tiebreak_tuple)` so Python tuple comparison resolves
  ties by kickers.
- Handles the **wheel** (A-2-3-4-5 as the low straight) and picks the best 5 of
  7 by evaluating all C(7,5)=21 combinations (simple, fast enough for v0).

**Correctness is non-negotiable**, so this ships with an exhaustive test file:
every category, flush-over-straight, quads kicker, full-house ranking, wheel
straight, split pots, and a set of hand-picked head-to-head comparisons with
known winners.

---

## 4. Invalid-action handling (reuses the runner policy)

No new mechanism — the existing `on_invalid_action` policy applies. For Hold'em
the fallback order is **check → fold → call → first legal type** (conservative;
all amount-free). The runner records, per the existing `InvalidInfo`: the raw
agent output, the parsed `(type, amount)`, the reason, and the fallback chosen.

Two counters feed the summary: **invalid action rate** (illegal action type)
and **invalid amount rate** (legal type, bad amount). Both are per-agent.

---

## 5. Agent template (`agents/templates/holdem.py`)

The default model agent stays generic; only the template is poker-specific.
- `render_prompt`: rules + the observation's natural-language render + the legal
  action types + the min/max amount range + a strict output-format spec. It
  states explicitly that an amount is an **integer total street commitment**
  within `[min, max]`.
- **Output format**: one action type, plus an integer amount for bet/raise.
  Recommended response grammar: a single line like `raise 8`, `bet 6`, `call`,
  `check`, `fold`, `all_in`. The parser is tolerant (case-insensitive, extracts
  the type token and the first integer) but rejects anything outside the legal
  set/range → `None` → repair prompt → fallback.
- `repair_prompt`: restates the legal types and the amount range.

Baselines for fast, no-API testing:
- **RandomHoldemAgent**: picks a legal type uniformly; for bet/raise picks a
  random legal integer (or a few canonical fractions of pot).
- **HoldemHeuristicAgent**: a simple hand-strength rule (e.g. raise strong made
  hands / high pairs, call/check medium, fold weak to big bets) — a non-trivial
  calibration baseline.

---

## 6. Evaluation metrics (`eval/evaluator.py` extensions)

Poker is high-variance, so the headline is **average chip delta per hand** over
many hands, never single-hand win rate. The episode summary record gains a few
game-stamped fields (`reason`, per-player action-type tallies, all-in flag) so
the evaluator can compute:

- Total & **average chip delta per hand** (primary), with 95% CI.
- **bb/100** (big blinds won per 100 hands) = avg_delta / big_blind × 100.
- Hand win rate; **showdown win rate**; **fold win rate**.
- **Invalid action rate** and **invalid amount rate**.
- Action-frequency breakdown: fold / check / call / bet / raise / all-in.
- Aggression frequency (bet+raise) / (call), all-in frequency.
- Average bet size; average episode length (streets reached).

Metrics are aggregated by agent identity across both button positions
(seat-swapped), so reported skill is position-neutral.

---

## 7. Config & CLI (no new mechanism)

Reuses the YAML schema and `aibattle` CLI. A Hold'em match is just
`game.name: holdem`. Example `configs/holdem_oss_vs_heuristic.yaml`:
```yaml
game:
  name: holdem
  version: "1.0.0"
  params: {}            # later: stack/blinds overrides

players:
  player_0: { agent: { type: model, name: gpt-oss-120b, model: {...} } }
  player_1: { agent: { type: builtin, name: holdem_heuristic } }

run:
  episodes: 200
  seed: 7
  seat_swap: true       # duplicate poker: same deal, swapped buttons
  on_invalid_action: fallback
  max_concurrency: 32

output:
  dir: ./runs/holdem_oss_vs_heuristic
  save_full_log: true
  save_summary: true
  save_trajectories: true
  save_transcripts: true
```
Human play works too (`type: human`) — the interactive observer already renders
moves; the human prompt will show the legal types and the amount range.

---

## 8. What is reused vs new

**Reused unchanged**: `Game`/`Agent` ABCs (with the §1 additions), runner loop,
seat-swap, invalid-action policy, JSONL logging, trajectories + transcripts,
deterministic per-decision seeding, model client (incl. reasoning capture),
config loader, CLI (`run`/`eval`), `--rerun`, per-run output dirs, progress bars.

**New**: `games/holdem.py`, `games/poker_eval.py`, `agents/templates/holdem.py`,
`agents/holdem baselines`, the `amount` protocol field + `validate_action` (and
Kuhn retrofit), and the Hold'em eval-metric extensions.

---

## 9. Build order

1. **Protocol change** — `Action` dataclass, `AgentResponse.amount`,
   `Game.validate_action`; retrofit Kuhn so all current behavior/tests are
   unchanged. (Smallest, riskiest-to-the-existing-system step; do it first.)
2. **Hand evaluator** + exhaustive ranking tests (independently testable).
3. **`holdem.py`**: state, blinds/positions, dealing, **betting rounds** (the
   correctness-critical part), all-in, uncalled-bet return, showdown — tested
   with scripted hands against known payoffs and zero-sum assertions.
4. **Baselines** (random + heuristic) → first end-to-end Hold'em match with no
   model dependency. **Milestone: a fully logged Hold'em hand.**
5. **Template + amount parsing**; eval metric extensions; example config.
6. Small **Fireworks model-vs-model** smoke run to confirm the numeric action
   loop works against a real model.

Each step is independently testable; the framework is exercisable end-to-end
after step 4 with zero model dependencies.

---

## 10. Risk register (where bugs hide)

- **Betting-round termination** — heads-up order (SB first preflop, BB first
  postflop), the **BB option** preflop, and "everyone acted since last
  aggression." Most likely source of subtle bugs → most tests.
- **Min-raise & short-stack all-ins** — all-in under a full raise not reopening
  action; uncalled bet returned so deltas sum to zero.
- **Hand evaluator correctness** — wheel straight, kickers, split pots.
- **Amount semantics** — enforce "total street commitment" everywhere; a
  raise-to vs raise-by slip corrupts pots silently.
- **Reproducibility under parallelism** — deck shuffled from the per-episode
  deal seed; model stochasticity remains non-reproducible by design.

---

## 11. Deferred past v0 (explicitly out of scope)

Multi-player & side pots, running cash-game bankroll, tournaments, antes,
increasing blinds, configurable params beyond the defaults, per-street raise
caps (unless round logic forces one), and any RL/leaderboard tooling.
