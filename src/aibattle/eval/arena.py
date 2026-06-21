"""Unified data interface + Chatbot-Arena active-sampling MATCH mechanism.

This module owns the two things the onboarding PR actually contributes:

  1. the **unified raw-data interface** every game is normalised into, and
  2. the **active-sampling match mechanism** — choosing which opponents a new
     model should play so it reaches a stable rating in as few games as
     possible, and automatically focusing on informative (close-strength)
     opponents.

**Scoring lives elsewhere on purpose.** Ratings reuse the repo's canonical
`scripts/elo_util.py` (Bradley-Terry → Elo, field mean 1500, chip-weighted for
poker, bootstrap CIs) so an onboarded model's number is directly comparable to
the existing reports. This module is pure (no I/O) and never fits ratings; it
only decides *who plays whom*.

------------------------------------------------------------------------------
UNIFIED RAW-DATA INTERFACE  (one JSON object per game)
------------------------------------------------------------------------------
    {
      "game":  "leduc_poker",
      "kind":  "versus",            # "versus" (PvP) | "environment" (PvE)
      "elo_basis": "chips",         # optional, versus only: "wins" (default) | "chips"
      "models": ["m1", "m2", ...],  # optional; inferred from episodes if absent
      "episodes": [ <episode> ]
    }

versus episode — each participant's scalar payoff for that game (works for
heads-up and multi-seat tables; pairwise records are derived from it)::

    {"scores": {"m1": 1.0, "m2": -1.0}, "seed": 123}

environment episode — one model's scalar vs the environment (higher = better)::

    {"model": "m1", "score": 3.5, "seed": 123}

`episodes_to_records` turns versus episodes into the `(a, b, result)` and
`(a, b, chips_a, chips_b)` record lists that `elo_util.wld_from_records` /
`elo_util.gross_from_records` consume.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------
@dataclass
class GameData:
    game: str
    kind: str                 # "versus" | "environment"
    models: list
    episodes: list
    elo_basis: str = "wins"   # versus only: "wins" | "chips"

    def env_scores(self) -> dict:
        """model -> list[float] of per-episode scalars (environment games)."""
        out: dict = {}
        for ep in self.episodes:
            m = ep.get("model")
            if m is None:
                continue
            out.setdefault(m, []).append(float(ep["score"]))
        return out


def from_dict(d: dict) -> GameData:
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
    return GameData(game=d.get("game", ""), kind=d.get("kind", "versus"),
                    models=list(models), episodes=episodes,
                    elo_basis=d.get("elo_basis", "wins"))


def load_unified(path: str) -> GameData:
    with open(path, encoding="utf-8") as fh:
        return from_dict(json.load(fh))


def episodes_to_records(episodes: list) -> tuple:
    """Derive (wld_records, chip_records) from unified versus episodes.

    Each episode's `scores` map is decomposed into all unordered pairs:
      * wld:  (a, b, +1 | -1 | 0)         — higher payoff wins, equal = draw
      * chip: (a, b, payoff_a, payoff_b)  — for the chip-weighted fit
    A k-player episode yields C(k, 2) records of each kind. These are exactly
    the flat record lists `elo_util.{wld,gross}_from_records` and
    `elo_util.bootstrap_elo` operate on.
    """
    wld, chips = [], []
    for ep in episodes:
        scores = ep.get("scores")
        if not scores:
            continue
        ms = list(scores.keys())
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                sa, sb = float(scores[a]), float(scores[b])
                wld.append((a, b, 1 if sa > sb else (-1 if sa < sb else 0)))
                chips.append((a, b, sa, sb))
    return wld, chips


# ---------------------------------------------------------------------------
# Active-sampling match mechanism (Chatbot Arena Eq. 9)
# ---------------------------------------------------------------------------
@dataclass
class UnitState:
    """Accumulated battle stats for one (game, opponent) sampling unit."""

    game: str
    opponent: str
    n: int = 0
    wins: float = 0.0     # new-model wins; draws add 0.5 here and to losses
    losses: float = 0.0

    def record(self, y: float) -> None:
        """y = new model's outcome: 1 win / 0 loss / 0.5 draw."""
        self.n += 1
        self.wins += y
        self.losses += (1.0 - y)

    @property
    def p(self) -> float:
        return 0.5 if self.n == 0 else self.wins / self.n


def info_gain(unit: UnitState) -> float:
    """Marginal reduction in the pair's standard error from one more vote
    (Chatbot Arena Eq. 9, the diagonal-of-covariance rule). Fewer votes and a
    closer matchup (p≈0.5) ⇒ larger gain; n=0 is maximal (cold start)."""
    if unit.n == 0:
        return 1.0
    var = max(unit.p * (1.0 - unit.p), 1e-6)
    return math.sqrt(var / unit.n) - math.sqrt(var / (unit.n + 1))


def proximity_weight(elo_new: float, elo_opp: float, scale: float = 200.0) -> float:
    """Up-weight opponents close to the new model's current estimate — the
    score/ranking-estimation regime (similar strength == most informative)."""
    return math.exp(-abs(elo_new - elo_opp) / max(scale, 1e-6))


def unit_weights(units: list, elo_new: float, opp_elo: dict, *,
                 proximity_scale: float = 200.0, floor: float = 1e-3) -> np.ndarray:
    """Sampling weight per unit = info_gain × proximity, floored for coverage."""
    w = np.array([
        info_gain(u) * proximity_weight(elo_new, opp_elo.get(u.opponent, elo_new),
                                        proximity_scale)
        for u in units
    ], dtype=float)
    return np.maximum(w, floor)


def sample_units(units: list, weights: np.ndarray, k: int, rng: random.Random) -> list:
    """Draw k units WITH replacement ∝ weights → [(unit, prob), ...].

    With-replacement is deliberate: it lets the scheduler launch the same
    (game, opponent) repeatedly, several opponents in one game, or the same
    opponent across games — the parallelism the onboarding loop needs.
    """
    total = float(weights.sum())
    probs = weights / total if total > 0 else np.full(len(units), 1.0 / len(units))
    idx = rng.choices(range(len(units)), weights=list(probs), k=k)
    return [(units[i], float(probs[i])) for i in idx]


# ---------------------------------------------------------------------------
# Rank spread from confidence intervals (LMArena definition)
# ---------------------------------------------------------------------------
def rank_spread(elo: dict, ci: dict) -> dict:
    """best/worst rank per model from {model: (lo, hi)} CIs.

    best(M)  = 1 + #{x : lo[x] > hi[M]}
    worst(M) = 1 + #{x : hi[x] > lo[M]}
    Models with a missing CI endpoint are skipped in the comparison.
    """
    out = {}
    for m in elo:
        lo_m, hi_m = ci.get(m, (None, None))
        if lo_m is None or hi_m is None:
            out[m] = (None, None)
            continue
        best = 1 + sum(1 for x in elo if x != m and (ci.get(x, (None, None))[0] or -1e18) > hi_m)
        worst = 1 + sum(1 for x in elo if x != m and (ci.get(x, (None, None))[1] or 1e18) > lo_m)
        out[m] = (best, worst)
    return out
