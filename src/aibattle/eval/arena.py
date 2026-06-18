"""Chatbot-Arena-style rating core for AI Battle Arena.

This module is **pure** (no I/O, no API calls) so it can be unit-tested with
synthetic data. It implements the pieces of the LMArena / Chatbot Arena
methodology that we need to place a *new* model into an existing pool:

  * weighted Bradley-Terry maximum-likelihood scoring (with inverse-sampling
    weights for de-biasing the active sampler, and an L2 "anchor" prior that
    keeps the existing pool semi-frozen while allowing slight fine-tuning),
  * bootstrap confidence intervals,
  * the rank-spread (best/worst rank) computation,
  * the active-sampling info-gain rule and a with-replacement unit sampler,
  * a scalar "environment" placement path for PvE games (no opponent).

------------------------------------------------------------------------------
UNIFIED RAW-DATA INTERFACE  (the one format all 10 games are converted into)
------------------------------------------------------------------------------
Every game's raw data — the historical pool data AND the new model's freshly
played episodes — is a single JSON object of this shape::

    {
      "game":  "leduc_poker",          # game id
      "kind":  "versus",               # "versus" (PvP) | "environment" (PvE)
      "models": ["m1", "m2", ...],     # optional; inferred from episodes if absent
      "episodes": [ <episode>, ... ]
    }

A **versus** episode lists every participating model and its scalar payoff for
that episode (works for heads-up *and* multi-seat table games)::

    {"scores": {"m1": 1.0, "m2": -1.0}, "seed": 123, "weight": 1.0}

  - Pairwise comparisons are derived from `scores`: for each unordered pair the
    higher score wins, equal scores are a draw.
  - `weight` is optional (default 1.0). The onboarding scheduler stores
    `1 / P_sample` here so BT fitting can de-bias the active sampling (IPW).

An **environment** episode is one model's scalar result vs the environment::

    {"model": "m1", "score": 3.5, "seed": 123}

  - Higher score == better (document/transform per game before feeding in).

`Comparison` is the internal pairwise record BT consumes; `load_unified` /
`episodes_to_comparisons` turn the interface above into it.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Elo mapping constants (same convention LMArena uses).
SCALE = 400.0
BASE = 10.0
INIT_RATING = 1000.0
_LOG_BASE = math.log(BASE)


# ---------------------------------------------------------------------------
# Unified interface helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    """One pairwise observation: `winner` beat `loser` (or a draw)."""

    a: str
    b: str
    # outcome from a's perspective: 1.0 = a wins, 0.0 = b wins, 0.5 = draw
    y: float
    weight: float = 1.0


def episodes_to_comparisons(episodes: list) -> list:
    """Derive pairwise `Comparison`s from unified *versus* episodes.

    Each episode's `scores` map is decomposed into all unordered pairs; the
    higher scalar payoff wins, ties are draws. Multi-seat games therefore
    contribute C(k,2) comparisons per episode.
    """
    comps: list = []
    for ep in episodes:
        scores = ep.get("scores")
        if not scores:
            continue
        w = float(ep.get("weight", 1.0))
        models = list(scores.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                sa, sb = scores[ma], scores[mb]
                if sa > sb:
                    y = 1.0
                elif sa < sb:
                    y = 0.0
                else:
                    y = 0.5
                comps.append(Comparison(ma, mb, y, w))
    return comps


@dataclass
class GameData:
    game: str
    kind: str  # "versus" | "environment"
    models: list
    episodes: list

    @property
    def comparisons(self) -> list:
        return episodes_to_comparisons(self.episodes)

    def env_scores(self) -> dict:
        """model -> list[float] of per-episode scalars (environment games)."""
        out: dict = {}
        for ep in self.episodes:
            m = ep.get("model")
            if m is None:
                continue
            out.setdefault(m, []).append(float(ep["score"]))
        return out


def load_unified(path: str) -> GameData:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return from_dict(d)


def from_dict(d: dict) -> GameData:
    kind = d.get("kind", "versus")
    episodes = d.get("episodes", [])
    models = d.get("models")
    if not models:
        ms: set = set()
        for ep in episodes:
            if "scores" in ep:
                ms.update(ep["scores"].keys())
            elif "model" in ep:
                ms.add(ep["model"])
        models = sorted(ms)
    return GameData(game=d.get("game", ""), kind=kind,
                    models=list(models), episodes=episodes)


# ---------------------------------------------------------------------------
# Weighted Bradley-Terry MLE
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # clip the linear term so a transiently large beta gap can't overflow exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class BTResult:
    models: list
    beta: dict          # model -> BT strength
    elo: dict           # model -> Elo-scaled rating

    def ranking(self) -> list:
        return sorted(self.models, key=lambda m: self.elo[m], reverse=True)


def weighted_bt_mle(
    comparisons: list,
    models: list,
    *,
    anchor: Optional[dict] = None,
    anchor_strength: float = 0.0,
    free_prior: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    step_cap: float = 2.0,
    beta_cap: float = 20.0,
) -> BTResult:
    """Fit Bradley-Terry strengths by weighted MLE (Newton's method).

    Maximises  sum_t w_t * [ y_t log p_t + (1-y_t) log(1-p_t) ]
    where p_t = sigmoid(beta_a - beta_b), minus an L2 prior.

    - `comparisons` carry per-observation weights (use 1/P_sample for IPW
      de-biasing of the active sampler).
    - `anchor` / `anchor_strength`: pull listed models' beta toward
      `anchor[m]` with strength `anchor_strength`. High strength == the pool is
      effectively frozen; low strength == "reuse but allow fine-tuning".
    - `free_prior`: tiny ridge on every model (toward anchor, or 0) so the
      Hessian is positive-definite and the gauge is fixed even for a brand-new,
      weakly-connected model.
    """
    idx = {m: k for k, m in enumerate(models)}
    M = len(models)
    if M == 0:
        return BTResult(models=[], beta={}, elo={})

    a_idx = np.array([idx[c.a] for c in comparisons], dtype=int)
    b_idx = np.array([idx[c.b] for c in comparisons], dtype=int)
    y = np.array([c.y for c in comparisons], dtype=float)
    w = np.array([c.weight for c in comparisons], dtype=float)

    # Per-model L2 prior strength and target.
    lam = np.full(M, float(free_prior))
    target = np.zeros(M)
    if anchor:
        for m, val in anchor.items():
            if m in idx:
                lam[idx[m]] += float(anchor_strength)
                target[idx[m]] = float(val)

    beta = np.zeros(M)
    if anchor:  # warm-start anchored models at their prior
        for m, val in anchor.items():
            if m in idx:
                beta[idx[m]] = float(val)

    for _ in range(max_iter):
        if len(comparisons):
            d = beta[a_idx] - beta[b_idx]
            p = _sigmoid(d)
            r = w * (y - p)               # residual per comparison
            grad = np.zeros(M)
            np.add.at(grad, a_idx, r)
            np.add.at(grad, b_idx, -r)
            wv = w * p * (1.0 - p)        # info weight per comparison
            H = np.zeros((M, M))
            np.add.at(H, (a_idx, a_idx), wv)
            np.add.at(H, (b_idx, b_idx), wv)
            np.add.at(H, (a_idx, b_idx), -wv)
            np.add.at(H, (b_idx, a_idx), -wv)
        else:
            grad = np.zeros(M)
            H = np.zeros((M, M))
        # L2 prior contributions (we maximise LL - 0.5*lam*(beta-target)^2).
        grad -= lam * (beta - target)
        H[np.diag_indices(M)] += lam
        # Newton step: maximise -> beta += H^{-1} grad  (H is the neg-Hessian, PD).
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # Damp: with separable data the unregularised Newton step can overshoot
        # massively. Cap per-step and overall magnitude so the fit stays finite
        # (the L2 ridge above is what ultimately bounds an undefeated model).
        step = np.clip(step, -step_cap, step_cap)
        beta = np.clip(beta + step, -beta_cap, beta_cap)
        if np.max(np.abs(step)) < tol:
            break

    beta_d = {m: float(beta[idx[m]]) for m in models}
    elo_d = {m: INIT_RATING + SCALE * beta_d[m] / _LOG_BASE for m in models}
    return BTResult(models=list(models), beta=beta_d, elo=elo_d)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(
    comparisons: list,
    models: list,
    *,
    rounds: int = 200,
    ci: float = 0.95,
    rng: Optional[random.Random] = None,
    **bt_kwargs,
) -> dict:
    """Bootstrap the weighted BT Elo to get per-model CIs.

    Resamples comparisons with replacement `rounds` times, refits each time,
    and returns {model: (lo, hi)} percentile intervals on the Elo scale.
    """
    rng = rng or random.Random(0)
    n = len(comparisons)
    if n == 0:
        return {m: (INIT_RATING, INIT_RATING) for m in models}
    samples = {m: [] for m in models}
    for _ in range(rounds):
        idxs = [rng.randrange(n) for _ in range(n)]
        boot = [comparisons[i] for i in idxs]
        res = weighted_bt_mle(boot, models, **bt_kwargs)
        for m in models:
            samples[m].append(res.elo[m])
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    return {m: (float(np.percentile(samples[m], lo_q)),
                float(np.percentile(samples[m], hi_q))) for m in models}


# ---------------------------------------------------------------------------
# Rank spread (best/worst rank from overlapping CIs) — LMArena definition
# ---------------------------------------------------------------------------
def rank_spread(elo: dict, ci: dict) -> dict:
    """best/worst rank per model from confidence intervals.

    best(M) = 1 + #{x : lo[x] > hi[M]}
    worst(M) = 1 + #{x : hi[x] > lo[M]}   (x != M)
    """
    out = {}
    for m in elo:
        lo_m, hi_m = ci[m]
        best = 1 + sum(1 for x in elo if x != m and ci[x][0] > hi_m)
        worst = 1 + sum(1 for x in elo if x != m and ci[x][1] > lo_m)
        out[m] = (best, worst)
    return out


# ---------------------------------------------------------------------------
# Active sampling (Chatbot Arena Eq. 9) + with-replacement unit sampler
# ---------------------------------------------------------------------------
@dataclass
class UnitState:
    """Accumulated battle stats for one (game, opponent) sampling unit."""

    game: str
    opponent: str
    n: int = 0
    wins: float = 0.0     # new model wins (draws count 0.5 here and in losses)
    losses: float = 0.0

    def record(self, y: float) -> None:
        """y is the new model's outcome: 1 win / 0 loss / 0.5 draw."""
        self.n += 1
        self.wins += y
        self.losses += (1.0 - y)

    @property
    def p(self) -> float:
        return 0.5 if self.n == 0 else self.wins / self.n


def info_gain(unit: UnitState) -> float:
    """Marginal reduction in the pair's standard error from one more vote
    (Chatbot Arena Eq. 9 diagonal rule)."""
    if unit.n == 0:
        return 1.0  # cold-start: maximal priority (capped/normalised by caller)
    var = unit.p * (1.0 - unit.p)
    var = max(var, 1e-6)  # avoid a zero gain when p has saturated at 0/1
    return math.sqrt(var / unit.n) - math.sqrt(var / (unit.n + 1))


def proximity_weight(elo_new: float, elo_opp: float, scale: float = 200.0) -> float:
    """Up-weight opponents close to the new model's current estimate — the
    score/ranking-estimation regime (similar strength == most informative)."""
    return math.exp(-abs(elo_new - elo_opp) / max(scale, 1e-6))


def unit_weights(
    units: list,
    elo_new: float,
    opp_elo: dict,
    *,
    proximity_scale: float = 200.0,
    floor: float = 1e-3,
) -> np.ndarray:
    """Sampling weight per unit = info_gain * proximity, floored for coverage."""
    w = np.array([
        info_gain(u) * proximity_weight(elo_new, opp_elo.get(u.opponent, elo_new),
                                        proximity_scale)
        for u in units
    ], dtype=float)
    w = np.maximum(w, floor)
    return w


def sample_units(
    units: list,
    weights: np.ndarray,
    k: int,
    rng: random.Random,
) -> list:
    """Draw `k` units WITH replacement ∝ weights.

    With-replacement is deliberate: it lets the scheduler launch the same
    (game, opponent) pair multiple times concurrently (repeat battles), pick
    several opponents for one game, or the same opponent across games — exactly
    the parallelism the onboarding loop needs. Returns (units, probs) so the
    caller can store P_sample for IPW de-biasing.
    """
    total = float(weights.sum())
    probs = weights / total if total > 0 else np.full(len(units), 1.0 / len(units))
    chosen_idx = rng.choices(range(len(units)), weights=list(probs), k=k)
    return [(units[i], float(probs[i])) for i in chosen_idx]


# ---------------------------------------------------------------------------
# Environment (PvE) placement — no opponent, rank by scalar score
# ---------------------------------------------------------------------------
@dataclass
class EnvPlacement:
    model: str
    mean: float
    ci: tuple
    rank: int
    rank_spread: tuple
    leaderboard: list  # [(model, mean), ...] sorted desc, incl. the new model


def environment_placement(
    new_scores: list,
    pool_scores: dict,
    *,
    rounds: int = 200,
    ci: float = 0.95,
    rng: Optional[random.Random] = None,
    higher_is_better: bool = True,
) -> EnvPlacement:
    """Place a new model among a pool by mean scalar score (PvE games).

    Bootstraps the new model's mean to a CI and derives a rank spread by
    comparing it against each pool model's bootstrapped mean CI.
    """
    rng = rng or random.Random(0)
    pool_means = {m: float(np.mean(s)) for m, s in pool_scores.items() if s}
    new_mean = float(np.mean(new_scores)) if new_scores else 0.0

    def boot_ci(scores: list) -> tuple:
        if not scores:
            return (0.0, 0.0)
        n = len(scores)
        means = [float(np.mean([scores[rng.randrange(n)] for _ in range(n)]))
                 for _ in range(rounds)]
        lo = float(np.percentile(means, (1 - ci) / 2 * 100))
        hi = float(np.percentile(means, (1 + ci) / 2 * 100))
        return (lo, hi)

    new_ci = boot_ci(new_scores)
    pool_ci = {m: boot_ci(s) for m, s in pool_scores.items() if s}

    sign = 1.0 if higher_is_better else -1.0
    lb = sorted([(self_m, mu) for self_m, mu in {**pool_means,
                 "__new__": new_mean}.items()],
                key=lambda kv: sign * kv[1], reverse=True)
    lb = [(m if m != "__new__" else "<new>", mu) for m, mu in lb]
    rank = 1 + sum(1 for m, mu in pool_means.items() if sign * mu > sign * new_mean)

    # rank spread from CI overlap (orientation-aware).
    if higher_is_better:
        best = 1 + sum(1 for m, c in pool_ci.items() if c[0] > new_ci[1])
        worst = 1 + sum(1 for m, c in pool_ci.items() if c[1] > new_ci[0])
    else:
        best = 1 + sum(1 for m, c in pool_ci.items() if c[1] < new_ci[0])
        worst = 1 + sum(1 for m, c in pool_ci.items() if c[0] < new_ci[1])
    return EnvPlacement(model="<new>", mean=new_mean, ci=new_ci, rank=rank,
                        rank_spread=(best, worst), leaderboard=lb)
