"""Correctness + smoke tests for the Arena rating core and the onboarding loop.

No API calls: battles are synthetic (drawn from latent skills), so this verifies
the math and the scheduler offline.
"""

from __future__ import annotations

import asyncio
import math
import random

import numpy as np
import pytest

from aibattle.eval import arena
from aibattle.eval.arena import Comparison, GameData, UnitState

import importlib.util
import os
import sys

# import scripts/onboard_model.py
_SPEC = importlib.util.spec_from_file_location(
    "onboard_model",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "onboard_model.py"))
onboard_model = importlib.util.module_from_spec(_SPEC)
sys.modules["onboard_model"] = onboard_model
_SPEC.loader.exec_module(onboard_model)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _round_robin_comparisons(skills, n_per_pair, rng):
    """Synthetic pool comparisons drawn from latent skills."""
    comps = []
    models = list(skills)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            p = _sigmoid(skills[a] - skills[b])
            for _ in range(n_per_pair):
                comps.append(Comparison(a, b, 1.0 if rng.random() < p else 0.0))
    return comps


# ---------------------------------------------------------------------------
# BT correctness
# ---------------------------------------------------------------------------
def test_bt_recovers_ranking():
    rng = random.Random(1)
    skills = {"a": 1.5, "b": 0.5, "c": -0.5, "d": -1.5}
    comps = _round_robin_comparisons(skills, 400, rng)
    res = arena.weighted_bt_mle(comps, list(skills))
    assert res.ranking() == ["a", "b", "c", "d"]
    # recovered beta gaps should correlate with true skill gaps
    order = sorted(skills, key=lambda m: res.beta[m])
    assert order == ["d", "c", "b", "a"]


def test_elo_mapping_constants():
    res = arena.weighted_bt_mle([Comparison("a", "b", 1.0)] * 5 +
                                [Comparison("a", "b", 0.0)] * 5, ["a", "b"])
    # equal record -> equal-ish elo near INIT
    assert abs(res.elo["a"] - res.elo["b"]) < 5.0
    assert abs(res.elo["a"] - arena.INIT_RATING) < 60.0


def test_weight_equals_replication():
    """A comparison with weight w must fit identically to w replicated copies.

    This is the mechanism the active sampler relies on: storing 1/P_sample as a
    weight makes a rarely-sampled battle count proportionally more, exactly as
    if it had been observed more often (IPW de-biasing)."""
    models = ["a", "b", "c"]
    mixed = ([Comparison("a", "b", 1.0)] * 3 + [Comparison("a", "b", 0.0)]
             + [Comparison("b", "c", 1.0)] * 2)
    weighted = [Comparison("a", "b", 1.0, 3.0), Comparison("a", "b", 0.0, 1.0),
                Comparison("b", "c", 1.0, 2.0)]
    r1 = arena.weighted_bt_mle(mixed, models, free_prior=1e-4)
    r2 = arena.weighted_bt_mle(weighted, models, free_prior=1e-4)
    for m in models:
        assert abs(r1.beta[m] - r2.beta[m]) < 1e-6


# ---------------------------------------------------------------------------
# Active sampling
# ---------------------------------------------------------------------------
def test_info_gain_monotonic_and_coldstart():
    cold = UnitState("g", "x", n=0)
    assert arena.info_gain(cold) == 1.0  # cold start has max priority
    u1 = UnitState("g", "x", n=4, wins=2, losses=2)
    u2 = UnitState("g", "x", n=40, wins=20, losses=20)
    assert arena.info_gain(u1) > arena.info_gain(u2)  # fewer votes -> more gain


def test_sample_units_with_replacement_and_proportional():
    rng = random.Random(3)
    units = [UnitState("g", "x", n=4, wins=2, losses=2),
             UnitState("g", "y", n=400, wins=200, losses=200)]
    opp_elo = {"x": 1000.0, "y": 1000.0}
    w = arena.unit_weights(units, 1000.0, opp_elo)
    picks = arena.sample_units(units, w, 2000, rng)
    assert len(picks) == 2000  # k draws, with replacement
    counts = {"x": 0, "y": 0}
    for u, p in picks:
        counts[u.opponent] += 1
    # x (fewer votes -> higher gain) should be drawn far more often than y
    assert counts["x"] > counts["y"] * 3


def test_rank_spread():
    elo = {"a": 1100.0, "b": 1000.0, "c": 900.0}
    ci = {"a": (1080, 1120), "b": (980, 1020), "c": (880, 920)}
    spread = arena.rank_spread(elo, ci)
    assert spread["a"] == (1, 1)  # clearly best
    assert spread["c"] == (3, 3)  # clearly worst
    # widen b so it overlaps a and c
    ci2 = {"a": (980, 1120), "b": (970, 1030), "c": (880, 1010)}
    spread2 = arena.rank_spread(elo, ci2)
    assert spread2["b"][0] <= 2 <= spread2["b"][1]


# ---------------------------------------------------------------------------
# Environment placement
# ---------------------------------------------------------------------------
def test_environment_placement():
    rng = random.Random(4)
    pool = {"hi": [rng.gauss(1.0, 0.5) for _ in range(60)],
            "mid": [rng.gauss(0.0, 0.5) for _ in range(60)],
            "lo": [rng.gauss(-1.0, 0.5) for _ in range(60)]}
    new_scores = [rng.gauss(0.0, 0.5) for _ in range(60)]  # ~ mid
    place = arena.environment_placement(new_scores, pool, rounds=100,
                                        rng=random.Random(5))
    assert place.rank in (2, 3)  # ~tied with "mid", clearly below "hi", above "lo"
    assert place.ci[0] < place.mean < place.ci[1]
    assert place.leaderboard[0][0] == "hi" and place.leaderboard[-1][0] == "lo"


# ---------------------------------------------------------------------------
# End-to-end onboarding (synthetic battles, no API)
# ---------------------------------------------------------------------------
def _versus_gamedata(skills, n_per_pair, rng, name="g_versus"):
    episodes = []
    models = list(skills)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            p = _sigmoid(skills[a] - skills[b])
            for _ in range(n_per_pair):
                sa = 1.0 if rng.random() < p else -1.0
                episodes.append({"scores": {a: sa, b: -sa}})
    return GameData(game=name, kind="versus", models=models, episodes=episodes)


def _env_gamedata(pool_means, n, rng, name="g_env"):
    episodes = []
    for m, mu in pool_means.items():
        for _ in range(n):
            episodes.append({"model": m, "score": rng.gauss(mu, 0.5)})
    return GameData(game=name, kind="environment",
                    models=list(pool_means), episodes=episodes)


def test_onboard_end_to_end_and_concurrency():
    rng = random.Random(7)
    pool_skills = {"top": 1.5, "mid": 0.0, "bot": -1.5}
    gd_v = _versus_gamedata(pool_skills, 80, rng)
    gd_e = _env_gamedata({"top": 1.0, "mid": 0.0, "bot": -1.0}, 50, rng)

    # The new model is genuinely mid-strength in versus, top in env.
    NEW_SKILL = 0.0
    peak = {"n": 0, "cur": 0}
    lock = asyncio.Lock()

    async def battle_fn(game, opponent, seed):
        async with lock:
            peak["cur"] += 1
            peak["n"] = max(peak["n"], peak["cur"])
        await asyncio.sleep(0.001)  # force real overlap so the cap is exercised
        r = random.Random(seed)
        p = _sigmoid(NEW_SKILL - pool_skills[opponent])
        ys = [1.0 if r.random() < p else 0.0 for _ in range(2)]
        async with lock:
            peak["cur"] -= 1
        return ys

    async def env_fn(game, seed):
        r = random.Random(seed)
        return [r.gauss(2.0, 0.5)]  # clearly the best in env

    results = asyncio.run(onboard_model.onboard(
        "qX", [gd_v, gd_e], battle_fn=battle_fn, env_fn=env_fn,
        parallel=5, min_battles=20, max_battles=120, ci_target=0.0,
        env_episodes=40, boot_rounds=60, seed=11))

    # concurrency never exceeded the budget
    assert peak["n"] <= 5
    assert peak["n"] >= 2  # actually ran concurrently

    rv = results["g_versus"]
    assert rv["kind"] == "versus"
    # mid-strength new model should land rank 2 of 4 (top, NEW~mid, mid, bot)
    assert rv["rank"] in (2, 3)
    assert rv["leaderboard"][0]["model"] == "top"
    assert rv["battles"] >= 20

    re = results["g_env"]
    assert re["kind"] == "environment"
    assert re["rank"] == 1  # best in env
