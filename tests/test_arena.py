"""Tests for the Arena match mechanism + onboarding loop.

Scoring is the repo's canonical `elo_util`; these tests cover (1) the unified
interface + record derivation, (2) the active-sampling match mechanism, and
(3) an end-to-end synthetic onboarding (no API) that lands a new model at the
right rank using far fewer games than a full round-robin.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import os
import random
import sys

import numpy as np

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS))  # so onboard_model can import elo_util

from aibattle.eval import arena
from aibattle.eval.arena import GameData, UnitState

_SPEC = importlib.util.spec_from_file_location(
    "onboard_model", os.path.join(_SCRIPTS, "onboard_model.py"))
onboard_model = importlib.util.module_from_spec(_SPEC)
sys.modules["onboard_model"] = onboard_model
_SPEC.loader.exec_module(onboard_model)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------
def test_episodes_to_records_decomposition():
    eps = [
        {"scores": {"a": 1.0, "b": -1.0}},          # a beats b
        {"scores": {"a": 0.0, "b": 0.0}},           # draw
        {"scores": {"a": 5, "b": 1, "c": -6}},      # 3-way: a>b, a>c, b>c
    ]
    wld, chip = arena.episodes_to_records(eps)
    assert ("a", "b", 1) in wld and ("a", "b", 0) in wld
    # 3-player episode contributes C(3,2)=3 records
    assert len(wld) == 1 + 1 + 3 and len(chip) == len(wld)
    # chip records keep magnitudes
    assert ("a", "c", 5.0, -6.0) in chip


def test_load_unified_infers_models_and_basis():
    gd = arena.from_dict({"game": "g", "kind": "versus", "elo_basis": "chips",
                          "episodes": [{"scores": {"x": 1, "y": -1}}]})
    assert set(gd.models) == {"x", "y"} and gd.elo_basis == "chips"


# ---------------------------------------------------------------------------
# Active sampling match mechanism
# ---------------------------------------------------------------------------
def test_info_gain_monotonic_and_coldstart():
    assert arena.info_gain(UnitState("g", "x", n=0)) == 1.0
    u1 = UnitState("g", "x", n=4, wins=2, losses=2)
    u2 = UnitState("g", "x", n=40, wins=20, losses=20)
    assert arena.info_gain(u1) > arena.info_gain(u2)


def test_proximity_prefers_close_opponents():
    near = arena.proximity_weight(1500, 1520)
    far = arena.proximity_weight(1500, 1900)
    assert near > far


def test_sample_units_with_replacement_and_proportional():
    rng = random.Random(3)
    units = [UnitState("g", "x", n=4, wins=2, losses=2),
             UnitState("g", "y", n=400, wins=200, losses=200)]
    w = arena.unit_weights(units, 1500.0, {"x": 1500.0, "y": 1500.0})
    picks = arena.sample_units(units, w, 2000, rng)
    assert len(picks) == 2000
    cx = sum(1 for u, _ in picks if u.opponent == "x")
    assert cx > (2000 - cx) * 3  # the low-n unit dominates


def test_rank_spread():
    elo = {"a": 1600, "b": 1500, "c": 1400}
    ci = {"a": (1580, 1620), "b": (1480, 1520), "c": (1380, 1420)}
    spread = arena.rank_spread(elo, ci)
    assert spread["a"] == (1, 1) and spread["c"] == (3, 3)


# ---------------------------------------------------------------------------
# End-to-end onboarding (synthetic battles, elo_util scoring, no API)
# ---------------------------------------------------------------------------
def _versus_gamedata(skills, n_per_pair, rng, name="g_versus", basis="wins"):
    eps, models = [], list(skills)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            p = _sigmoid(skills[a] - skills[b])
            for _ in range(n_per_pair):
                sa = 1.0 if rng.random() < p else -1.0
                eps.append({"scores": {a: sa, b: -sa}})
    return GameData(game=name, kind="versus", models=models, episodes=eps, elo_basis=basis)


def test_onboard_end_to_end_and_concurrency():
    rng = random.Random(7)
    pool_skills = {"top": 1.6, "mid": 0.0, "bot": -1.6}
    gd = _versus_gamedata(pool_skills, 120, rng)

    NEW_SKILL = 0.0  # genuinely mid-strength
    peak = {"n": 0, "cur": 0}
    lock = asyncio.Lock()

    async def battle_fn(game, opponent, seed):
        async with lock:
            peak["cur"] += 1; peak["n"] = max(peak["n"], peak["cur"])
        await asyncio.sleep(0.001)
        r = random.Random(seed)
        p = _sigmoid(NEW_SKILL - pool_skills[opponent])
        pairs = [(1.0, -1.0) if r.random() < p else (-1.0, 1.0) for _ in range(2)]
        async with lock:
            peak["cur"] -= 1
        return pairs

    async def env_fn(game, seed):
        return [random.Random(seed).gauss(2.0, 0.5)]

    results = asyncio.run(onboard_model.onboard(
        "qX", [gd], battle_fn=battle_fn, env_fn=env_fn, parallel=5,
        min_battles=20, max_battles=120, sd_target=0.0, boot_rounds=80, seed=11))

    assert 2 <= peak["n"] <= 5                      # concurrent, within budget
    r = results["g_versus"]
    assert r["leaderboard"][0]["model"] == "top"    # pool order preserved
    assert r["rank"] in (2, 3)                       # new model ~ mid
    assert r["battles"] >= 20
    # Active sampling places the model in fewer games than a full round-robin of
    # the same depth would need (3 pool pairs × 120 = 360 pool games existed;
    # the new model is rated from far fewer fresh battles).
    assert r["battles"] <= 120


def test_onboard_environment_path():
    rng = random.Random(4)
    pool = {"hi": [rng.gauss(1.0, 0.5) for _ in range(40)],
            "lo": [rng.gauss(-1.0, 0.5) for _ in range(40)]}
    eps = [{"model": m, "score": s} for m, ss in pool.items() for s in ss]
    gd = GameData(game="g_env", kind="environment", models=list(pool), episodes=eps)

    async def battle_fn(*a):
        return []

    async def env_fn(game, seed):
        return [random.Random(seed).gauss(2.0, 0.4)]  # clearly best

    results = asyncio.run(onboard_model.onboard(
        "qX", [gd], battle_fn=battle_fn, env_fn=env_fn, parallel=4,
        env_episodes=30, boot_rounds=80, seed=2))
    assert results["g_env"]["rank"] == 1
