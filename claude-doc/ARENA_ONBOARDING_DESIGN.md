# Arena-Style Offline Model Onboarding — Design

This document specifies the offline system that places a **new model** into an
existing pool and reports its rank in every game, using the Chatbot Arena
(LMArena) **match mechanism**. The contribution is the matchmaking, not a new
rating: active sampling automatically picks informative opponents and reaches a
stable rating in far fewer games than a full round-robin.

**Scoring is deliberately reused, not reinvented.** Ratings come from the
repo's canonical `scripts/elo_util.py` (Bradley-Terry → Elo, field mean 1500,
chip-weighted for poker, bootstrap CIs) — the exact fit the existing HTML
reports use — so an onboarded model's number is directly comparable to them.

The pipeline, per game:

> **active-sampling opponent selection → battles → `elo_util` Bradley-Terry
> scoring → bootstrap error bars → ranking.**

PvP ("versus") games run battles and fit Elo; PvE ("environment") games place
the model by mean score against the pool. Everything is offline: the pool's
historical results seed the existing ratings; only the new model's edges are
played.

---

## 1. The unified data interface (the contract)

Every game — historical pool data **and** the new model's freshly played
episodes — is normalized into **one JSON object per game**:

```jsonc
{
  "game":  "leduc_poker",
  "kind":  "versus",            // "versus" (PvP) | "environment" (PvE)
  "elo_basis": "chips",         // optional, versus only: "wins" (default) | "chips"
  "models": ["m1", "m2", ...],  // optional; inferred from episodes if absent
  "episodes": [ <episode>, ... ]
}
```

Two episode shapes — the only fork in the system:

- **versus**: `{"scores": {"m1": 1.0, "m2": -1.0}, "seed": 123}`
  - Records **each participant's scalar payoff**, not "who won" — identical for
    heads-up and multi-seat tables (`scores` simply has more keys).
- **environment**: `{"model": "m1", "score": 3.5, "seed": 123}`
  - One model's scalar vs the environment. **Higher is better** (transform the
    metric beforehand if a game is lower-is-better).

`elo_basis` mirrors the analyzers: `wins` rates by win/loss/draw; `chips` rates
by chip-weighted Elo (poker, where margin matters).

### Derived records (what the rater consumes)

`episodes_to_records(episodes)` decomposes versus episodes into the two flat
lists `elo_util` already understands:

- **wld**: `(a, b, +1 | −1 | 0)` — higher payoff wins, equal = draw.
- **chip**: `(a, b, payoff_a, payoff_b)` — for the chip-weighted fit.

A `k`-player episode yields `C(k, 2)` records of each kind. `GameData.env_scores()`
gives `{model: [scalars]}` for environment games.

This is the **abstraction boundary**: above it is any game's raw data; below it
the rater sees only records/scalars. To onboard against a new game, convert its
data into one of the two episode shapes and drop the file into `--pool <dir>`.

---

## 2. Execution workflow (`scripts/onboard_model.py`)

### Step 0 — Load pool, fit baseline Elo (`_build_game_run` / `_refit_versus`)

For each versus game, `episodes_to_records` builds the pool's wld/chip records;
`elo_util.bradley_terry` over them gives the pool's baseline Elo (used as
opponent strengths for proximity, §3.2). One `UnitState(game, opponent)` is
created per pool opponent (`n = wins = losses = 0`).

### Step 1 — One global active-sampling scheduler (`run_versus_pool`)

All `(game, opponent)` units across **all** versus games share a single
`--parallel` budget; the loop continuously refills in-flight slots:

```
while active units remain:
    while in_flight < parallel:
        pick = draw_one()         # sample 1 unit WITH replacement
        launch play(pick)
    await FIRST_COMPLETED          # a battle finishes → refill
```

`draw_one()` weights each live unit by `info_gain(unit) ×
proximity_weight(elo_new, elo_opp)` (current Elo from the latest fit), floors at
`1e-3` for coverage, normalizes, and draws one. Because units are drawn **with
replacement** and re-weighted on every completion, the three required parallel
patterns emerge naturally:

- distinct `(game, opp)` → **different pairs / different games** in parallel;
- the same unit repeatedly → **the same pair repeating the same game**;
- one opponent across games → **the same pair across games**.

### Step 2 — One battle → records → re-score (`play`)

```python
pairs = await battle_fn(game, opponent, seed)     # [(new_payoff, opp_payoff), ...]
for np_, op_ in pairs:
    unit.record(1.0 if np_ > op_ else 0.0 if np_ < op_ else 0.5)
    gr.new_wld.append((NEW, opponent, +1/−1/0))
    gr.new_chip.append((NEW, opponent, np_, op_))
gr.result = _refit_versus(...)                     # elo_util over pool + new records
if _stop_versus(...): gr.done = True
```

### Step 3 — Stop condition (`_stop_versus`)

Per game: `battles ≥ max_battles`, **or** (`battles ≥ min_battles` **and** the
new model's bootstrap Elo **SD ≤ `--sd-target`**). Until met, the report flags
the model **Preliminary**. This is the mechanism's payoff: it stops as soon as
the rating is tight, rather than completing a fixed schedule.

### Step 4 — Environment games (`run_env`, concurrent with versus)

No opponent, no sampling: play `env_episodes` solo episodes vs the environment,
then place by mean score against the pool with a bootstrap CI (`_place_env`).

---

## 3. Mathematical principles

### 3.1 Scoring — reused from `elo_util` (the report rater)

The same fit the HTML reports use, so numbers are comparable:

- **Bradley-Terry** strengths `p_m` by the MM (Zermelo) iteration
  `p_i ← W_i / Σ_j N_ij/(p_i+p_j)`, geometric-mean normalized; draws count as
  half a win to each side.
- **Elo** = `400·log₁₀(p_m)`, recentred so the rated field averages **1500**.
- **Separable records** (a model with no win-or-draw, or no loss-or-draw) have
  no finite rating: they are **excluded** from the fit and shown as "—" (rather
  than regularized) — `elo_util`'s convention.
- **Chip basis** (poker): `gross_from_records` feeds chips-won per matchup into
  the same fit, so magnitude matters; `wins` basis feeds win/loss/draw counts.
- **Bootstrap** (`bootstrap_elo`): resample the per-game records with
  replacement `n_boot` times, refit each time; the spread is the model's SD and
  central interval. The onboarding loop reads the new model's SD as its stop
  signal.

The onboarding code adds **no rating math** — `score_versus()` is a thin wrapper
selecting the wld/chip path and calling `bradley_terry` + `bootstrap_elo`.

### 3.2 Active sampling — the match mechanism (`arena.py`)

This is the PR's actual contribution: deciding which opponent the new model
plays next so its rating stabilises in as few games as possible.

**What is tracked.** Per `(game, opponent)` unit we keep only `(n, wins)` — the
count of games and the new model's win total against that one opponent. From
these comes the **per-pair** win-rate variance `p(1−p)` with `p = wins/n`. This
is the **diagonal** of the win-matrix covariance, stored as a vector of
independent per-pair numbers. We do **not** maintain a covariance *matrix*: no
off-diagonal pair-to-pair correlations, and no BT Fisher-information matrix.
This matches what LMArena actually deploys (Eq. 9 uses only the diagonal); it is
**not** a formal A/D/E-optimal design (which would need the full matrix).

> Per-pair vs per-model: the sampler uses **per-pair** variance (which opponent
> to play). The new model's **per-model** variance — the bootstrap SD of its Elo
> — is a separate quantity, used only as the stop signal (§3.3, Step 3).

**Info gain.** The marginal reduction in a pair's standard error from one more
game (Chatbot Arena Eq. 9, the diagonal-of-covariance rule):

```
info_gain(unit) = √( p(1−p) / n )  −  √( p(1−p) / (n+1) )      (n > 0)
                = 1.0                                           (n = 0, cold start)
```

Fewer games and a closer matchup (`p ≈ 0.5`) ⇒ larger gain. It is a **myopic,
per-coordinate** greedy step: it shrinks one diagonal entry at a time rather
than solving a global optimal design.

**Proximity (an addition beyond Eq. 9 — see caveat).** A factor that focuses
games on opponents near the new model's current Elo:

```
proximity_weight = exp( −|elo_new − elo_opp| / scale )           (scale default 200)
```

The per-unit sampling weight is `max(info_gain · proximity, floor)`; the floor
(`1e-3`) keeps every unit reachable (coverage, anti-collapse).

**`draw_one()`** builds the candidate list (every opponent of every game not yet
`done`), computes those weights from the *current* fit, normalises to a
distribution, and draws **one** unit **probabilistically** (not arg-max — argmax
would starve all but one opponent). It is called once per freed scheduler slot;
across calls this is sampling with replacement, which is what lets the same pair
repeat or several opponents/games run concurrently. A unit stays in the
candidate pool while a battle on it is in flight, so it can be re-drawn.

> **Caveat — proximity is not literally Eq. 9.** Chatbot Arena's rule has no
> separate proximity term; "prefer close opponents" falls out of the `p(1−p)`
> variance (a close match has `p ≈ 0.5` ⇒ max variance ⇒ max gain). We added an
> explicit proximity factor only because our `info_gain` is hard-coded to `1.0`
> at `n = 0`, which is flat across opponents and so gives no cold-start
> preference. A more faithful single-formula variant drops proximity and uses
> the **model-predicted** win rate `p̂ = σ((elo_new − elo_opp)/400)` for the
> variance `p̂(1−p̂)`, which makes a mismatched opponent low-priority even at
> `n = 0` — so the new model would *not* touch every opponent first. This is the
> "faithful-Arena" option in §7.

> **Note — no IPW.** LMArena de-biases active sampling with inverse-probability
> weights, but `elo_util`'s MLE is unweighted. A well-specified Bradley-Terry
> MLE is consistent under any pairing design, so the point estimate is
> unbiased; the bootstrap CI reflects exactly the games actually played. We
> therefore reuse `elo_util` as-is rather than fork it for weights (§7).

### 3.3 Uncertainty and ranking

- **Bootstrap CI / SD** per model from `elo_util.bootstrap_elo`.
- **Rank** = position of the new model in the Elo ordering (`elo_util.elo_key`
  sinks unrated models to the bottom).
- **Rank spread** (`arena.rank_spread`, LMArena definition) from the CIs:
  ```
  best(M)  = 1 + #{ x ≠ M : lo[x] > hi[M] }
  worst(M) = 1 + #{ x ≠ M : hi[x] > lo[M] }
  ```
- **Environment placement**: bootstrap the new model's mean score; rank by mean
  against the pool's means.

---

## 4. Injection boundary & knobs

`onboard(...)` is agnostic to *how* a battle is played; it calls two injected
coroutines, so the whole loop is testable without API calls:

```python
battle_fn(game, opponent, seed) -> list[(new_payoff, opp_payoff)]
env_fn(game, seed)              -> list[new_payoff]
```

- **Real**: `_make_real_fns` uses `Runner.run_match(new vs opponent,
  seat_swap=True)`, returning each episode's `(new_return, opp_return)`.
  Concurrency is governed by the scheduler's `parallel`; each battle runs its
  own episodes at `max_concurrency = ep_per_battle`.
- **Smoke/tests**: a synthetic function (Bernoulli from latent skills) drives
  the entire scheduler.

| Flag | Controls | Default |
|---|---|---|
| `--parallel` | concurrent in-flight battles (global budget) | 10 |
| `--min-battles` / `--max-battles` | per-game battle floor / ceiling | 20 / 200 |
| `--sd-target` | stop a game once the new model's bootstrap Elo SD ≤ this | 15 |
| `--env-episodes` | episodes per environment game | 40 |
| `--ep-per-battle` | episodes per scheduled battle (seat-swapped) | 2 |

---

## 5. End-to-end summary

> **interface** (versus/environment episodes) → `episodes_to_records` →
> **matching** (global continuous-fill scheduler: `info_gain × proximity`,
> with-replacement sampling) → **battles** (injected `battle_fn`) → **scoring**
> (`elo_util` Bradley-Terry, mean 1500, chip-weighted, bootstrap SD) → **rank +
> rank spread** → drop the *Preliminary* flag once the SD tightens.

---

## 6. Validation

`pytest tests/test_arena.py` (offline, no API):

- record decomposition (incl. multi-player `C(k,2)`) and model/basis inference;
- `info_gain` monotonic in `n` + maximal at cold start; proximity prefers
  close opponents; with-replacement sampling honors `k` and is weight-proportional;
- rank spread on separated CIs;
- end-to-end synthetic onboarding lands the model at the expected rank within
  the `parallel` cap and with fewer games than the pool's round-robin depth;
- the environment path ranks a clearly-best model first.

End-to-end smoke against the real (coached) pool — converted to the unified
format, synthetic battles, no API — yields 1500-centred, chip-weighted Elo with
bootstrap SD across all four games, matching the report scale.

## 7. Open items

- **Sampler fidelity (the main choice).** Three levels, increasing fidelity/cost:
  - *Current* — `info_gain` (empirical `p`, `1.0` at `n=0`) × explicit
    proximity. Tends to play every opponent once early.
  - *Faithful-Arena* — drop proximity; use the model-predicted `p̂(1−p̂)` for the
    per-pair variance, so mismatched opponents are low-priority even at `n=0`
    and the new model is *not* forced to face everyone (§3.2 caveat). Small
    change in `arena.py`.
  - *A-optimal* — maintain the BT Fisher-information matrix `I(β)` and pick the
    pair minimising `trace(I⁻¹)`, using network coupling to localise with the
    fewest games. Larger change; this is the full "covariance matrix" design.
  A `--sampler {arena,aopt}` switch could expose these.
- **IPW**: `elo_util` is unweighted (§3.2). If the active sampler's skew ever
  needs explicit correction, add inverse-probability weights to a forked rater;
  today it is intentionally omitted to keep one rating source of truth.
- **Termination**: stop is on the new model's bootstrap Elo SD (§3.3, Step 3).
  A rank-stability criterion (rank_spread collapses to one rank, or K stable
  rounds) could be added when the ordinal placement matters more than the SD.
- The `convert_to_unified` adapter currently reads the `pairs`/`games`-style
  aggregates and environment `ep*.json`; a fully general version would walk
  `ep*.json` directly to cover every tournament family (kuhn/board/match/table).
- Environment placement assumes higher-is-better; transform the metric for any
  lower-is-better game.
