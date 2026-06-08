"""Tests for Repeated Colonel Blotto (round flow, encoding, hidden info, scoring)."""

from __future__ import annotations

import json
import random

import pytest

from aibattle.games.registry import make_game, available_games
from aibattle.games.blotto import (
    RepeatedColonelBlotto, BlottoState,
    encode_alloc, parse_alloc, _score_round, _battlefield_outcomes,
    ROUNDS, RESOURCES, VALUES, N_FIELDS,
)
from aibattle.agents.templates.registry import make_template
from aibattle.agents.registry import make_agent, _BUILTINS
from aibattle.types import Move, AgentRequest, Observation
from aibattle import types as _types


def _game():
    return RepeatedColonelBlotto()


# --- AC-1 (Blotto slice): registry + Move dataclass unchanged --------------

def test_registry():
    g = make_game("repeated_colonel_blotto")
    assert isinstance(g, RepeatedColonelBlotto)
    assert "repeated_colonel_blotto" in available_games()
    assert make_template("repeated_colonel_blotto") is not None
    assert make_template("repeated_colonel_blotto", coached=True).coaching(
        AgentRequest("repeated_colonel_blotto", "1.0.0", "player_0",
                     Observation("player_0", {}, {}, [], [encode_alloc([20]*5)], ""),
                     "", 0)
    ).strip() != ""
    assert "blotto_random" in _BUILTINS


def test_move_dataclass_has_no_vector_field():
    # AC-5.2: types.py Move must remain type + optional amount only.
    fields = set(_types.Move.__dataclass_fields__.keys())
    assert fields == {"type", "amount"}


# --- AC-5.2: allocation encoding + validation ------------------------------

def test_encode_parse_roundtrip():
    assert encode_alloc([20, 20, 20, 20, 20]) == "alloc:20,20,20,20,20"
    assert parse_alloc("alloc:20,20,20,20,20") == [20, 20, 20, 20, 20]
    assert parse_alloc("alloc:100,0,0,0,0") == [100, 0, 0, 0, 0]


def test_parse_rejects_malformed():
    assert parse_alloc("alloc:20,20,20,20,21") is None   # sums to 101
    assert parse_alloc("alloc:20,20,20,20,19") is None   # sums to 99
    assert parse_alloc("alloc:50,50") is None            # wrong count
    assert parse_alloc("alloc:-10,30,30,30,20") is None  # negative
    assert parse_alloc("alloc:a,b,c,d,e") is None        # non-integer
    assert parse_alloc("20,20,20,20,20") is None         # missing prefix


def test_validate_action():
    g = _game()
    s = g.initial_state(random.Random(0))
    assert g.validate_action(s, "player_0", Move(type="alloc:30,30,20,10,10"))[0] is True
    assert g.validate_action(s, "player_0", Move(type="alloc:30,30,20,10,11"))[0] is False
    # amount must be None for this encoding.
    assert g.validate_action(s, "player_0", Move(type=encode_alloc([20]*5), amount=1))[0] is False


def test_legal_actions_entries_are_valid_moves():
    g = _game()
    s = g.initial_state(random.Random(0))
    legal = g.legal_actions(s, "player_0")
    assert legal == ["alloc:20,20,20,20,20"]
    for tok in legal:
        ok, _ = g.validate_action(s, "player_0", Move(type=tok))
        assert ok


# --- AC-5.1: round flow + scoring ------------------------------------------

def test_score_round():
    # field values [1,2,3,4,5]; p0 wins fields 0,2,4 (1+3+5=9), p1 wins 1,3 (2+4=6)
    a0 = [30, 0, 30, 0, 40]
    a1 = [0, 30, 0, 30, 0]   # sums to 60; not used via game, just _score_round
    p0, p1 = _score_round(a0, a1)
    assert p0 == 1 + 3 + 5
    assert p1 == 2 + 4


def test_player0_first_then_player1_resolves():
    g = _game()
    s = g.initial_state(random.Random(0))
    assert g.current_player(s) == "player_0"
    s = g.step(s, Move(type=encode_alloc([100, 0, 0, 0, 0])))
    # pending set -> now player_1 acts; round not yet resolved.
    assert g.current_player(s) == "player_1"
    assert s.round == 0
    assert len(s.history) == 0
    s = g.step(s, Move(type=encode_alloc([0, 25, 25, 25, 25])))
    # round resolved: p0 won only field 0 (value 1); p1 won fields 1-4 (2+3+4+5=14)
    assert len(s.history) == 1
    assert s.round == 1
    assert s.scores == {"player_0": 1, "player_1": 14}


def test_full_game_terminates_after_20_rounds():
    g = _game()
    s = g.initial_state(random.Random(0))
    steps = 0
    while not g.is_terminal(s):
        s = g.step(s, Move(type=encode_alloc([20, 20, 20, 20, 20])))
        steps += 1
        assert steps <= 2 * ROUNDS + 1
    assert len(s.history) == ROUNDS
    # all-equal allocations every round -> every field ties -> 0-0 -> draw.
    assert g.returns(s) == {"player_0": 0.0, "player_1": 0.0}


def test_battlefield_outcomes_detail():
    outs = _battlefield_outcomes([30, 0, 30, 0, 40], [0, 30, 0, 30, 0])
    assert len(outs) == N_FIELDS
    assert outs[0] == {"battlefield": 0, "value": 1, "alloc_0": 30, "alloc_1": 0,
                       "winner": "player_0"}
    assert outs[1]["winner"] == "player_1"
    # a tie battlefield scores for nobody
    tied = _battlefield_outcomes([20, 20, 20, 20, 20], [20, 20, 20, 20, 20])
    assert all(o["winner"] == "tie" for o in tied)


def test_resolved_record_has_outcomes_and_cumulative():
    g = _game()
    s = g.initial_state(random.Random(0))
    s = g.step(s, Move(type=encode_alloc([100, 0, 0, 0, 0])))   # p0
    s = g.step(s, Move(type=encode_alloc([0, 25, 25, 25, 25])))  # p1 resolves
    rec = s.history[0]
    # per-battlefield outcomes present and consistent with points
    assert "battlefields" in rec and len(rec["battlefields"]) == N_FIELDS
    assert rec["battlefields"][0]["winner"] == "player_0"   # 100 > 0 on field value 1
    assert rec["points_0"] == 1 and rec["points_1"] == 2 + 3 + 4 + 5
    # cumulative scores after the round are recorded
    assert rec["cumulative"] == {"player_0": 1, "player_1": 14}


def test_terminal_metadata_has_full_round_history():
    g = _game()
    s = g.initial_state(random.Random(5))
    rng = random.Random(5)
    while not g.is_terminal(s):
        # vary allocations a little so battlefields aren't all ties
        a = [20, 20, 20, 20, 20]
        i = rng.randrange(N_FIELDS); j = (i + 1) % N_FIELDS
        a[i] += 1; a[j] -= 1
        s = g.step(s, Move(type=encode_alloc(a)))
    meta = g.episode_metadata(s)
    assert meta["rounds_played"] == ROUNDS
    assert len(meta["round_history"]) == ROUNDS          # all 20 rounds, incl. round 20
    assert meta["round_history"][-1]["round"] == ROUNDS  # final round present
    assert "battlefields" in meta["round_history"][-1]
    assert "cumulative" in meta["round_history"][-1]
    assert meta["battlefield_values"] == list(VALUES)
    # cumulative of the last record matches final scores
    assert meta["round_history"][-1]["cumulative"] == meta["final_scores"]


def test_cumulative_winner_returns():
    g = _game()
    s = BlottoState(round=ROUNDS, scores={"player_0": 50, "player_1": 30},
                    pending=None, history=tuple({"round": i} for i in range(ROUNDS)),
                    done=True)
    assert g.returns(s) == {"player_0": 1.0, "player_1": -1.0}


# --- AC-5.3: hidden-information no leakage ----------------------------------

def test_player1_observation_hides_pending_allocation():
    g = _game()
    s = g.initial_state(random.Random(0))
    secret = [97, 1, 1, 1, 0]
    s = g.step(s, Move(type=encode_alloc(secret)))   # player_0 submits secret
    assert s.pending == tuple(secret)
    # player_1 is now to act and MUST NOT see player_0's pending allocation.
    o1 = g.observation(s, "player_1")
    blob = json.dumps({"private": o1.private, "public": o1.public,
                       "history": o1.history, "rendered": o1.rendered})
    assert "97" not in blob            # the distinctive pending value
    assert o1.history == []            # no resolved round yet
    assert "pending" not in blob.lower()
    # After resolution, the (now-resolved) allocations may appear in history.
    s = g.step(s, Move(type=encode_alloc([20, 20, 20, 20, 20])))
    o_next = g.observation(s, "player_0")
    assert o_next.history and o_next.history[0]["alloc_0"] == secret


def test_pre_resolution_persisted_record_has_no_pending(tmp_path):
    # Simulate what the runner persists for player_1's decision step: it serializes
    # the observation. Ensure player_0's pending allocation is absent from it.
    g = _game()
    s = g.initial_state(random.Random(1))
    secret = [96, 1, 1, 1, 1]
    s = g.step(s, Move(type=encode_alloc(secret)))
    o1 = g.observation(s, "player_1")
    persisted = json.dumps(o1.to_dict())
    assert "96" not in persisted
    assert "alloc_0" not in persisted  # no resolved record leaks the pending alloc


# --- AC-5.4: blotto_random builtin -----------------------------------------

def test_blotto_random_samples_valid_and_varied():
    import asyncio
    agent = make_agent({"type": "builtin", "name": "blotto_random"},
                       game_name="repeated_colonel_blotto")
    g = _game()
    s = g.initial_state(random.Random(0))
    seen = set()
    for step_idx in range(30):
        obs = g.observation(s, g.current_player(s))
        req = AgentRequest("repeated_colonel_blotto", "1.0.0",
                           g.current_player(s), obs, "", step_idx,
                           decision_seed=step_idx * 7919 + 1)
        resp = asyncio.run(agent.act(req))
        alloc = parse_alloc(resp.action)
        assert alloc is not None, f"invalid allocation: {resp.action}"
        assert sum(alloc) == RESOURCES and len(alloc) == N_FIELDS
        seen.add(tuple(alloc))
    assert len(seen) > 1   # genuinely varied, not the single default


# --- template parsing ------------------------------------------------------

def test_template_parse():
    t = make_template("repeated_colonel_blotto")
    obs = Observation("player_0", {}, {}, [], [encode_alloc([20]*5)], "")
    req = AgentRequest("repeated_colonel_blotto", "1.0.0", "player_0", obs, "", 0)
    assert t.parse("alloc:30,30,20,10,10", req) == Move(type="alloc:30,30,20,10,10")
    # spaces tolerated
    assert t.parse("My move: alloc: 40, 30, 10, 10, 10", req) == Move(type="alloc:40,30,10,10,10")
    # bare integers on the last line
    assert t.parse("reasoning\n25 25 25 15 10", req) == Move(type="alloc:25,25,25,15,10")
    # invalid (sums to 99) -> None
    assert t.parse("alloc:20,20,20,20,19", req) is None
