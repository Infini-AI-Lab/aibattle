"""Tests for Othello-lite 6x6 (rules, pass/double-pass, returns, registry, parsing)."""

from __future__ import annotations

import random

import pytest

from aibattle.games.registry import make_game, available_games
from aibattle.games.othello_lite import (
    OthelloLite6x6, coord_to_rc, rc_to_coord, _flips, _legal_cells, _counts,
    SIZE, PASS,
)
from aibattle.agents.templates.registry import make_template
from aibattle.types import Move, AgentRequest, Observation


def _game():
    return OthelloLite6x6()


def _empty_request(legal, player="player_0", board=None):
    obs = Observation(
        player=player,
        private={},
        public={"board": board or [[None] * SIZE for _ in range(SIZE)]},
        history=[],
        legal_actions=legal,
        rendered="",
    )
    return AgentRequest(game="othello_lite_6x6", game_version="1.0.0",
                        player=player, observation=obs, instructions="",
                        step_index=0)


# --- AC-1 (Othello slice): registry ---------------------------------------

def test_registry_make_game():
    g = make_game("othello_lite_6x6")
    assert isinstance(g, OthelloLite6x6)
    assert g.name == "othello_lite_6x6"
    assert "othello_lite_6x6" in available_games()


def test_registry_wrong_key_raises():
    with pytest.raises(ValueError):
        make_game("othello")


def test_template_registers_and_coaches():
    t = make_template("othello_lite_6x6")
    assert t is not None
    coached = make_template("othello_lite_6x6", coached=True)
    assert coached.coaching(_empty_request([PASS])).strip() != ""
    with pytest.raises(ValueError):
        make_template("othello6x6")


# --- AC-2.1: initial board + turn order ------------------------------------

def test_initial_board_and_black_first():
    g = _game()
    s = g.initial_state(random.Random(0))
    counts = _counts(s.grid)
    assert counts == {"player_0": 2, "player_1": 2}  # exactly four center pieces
    # W B / B W on central 2x2 (index 2..3): C3=W, D3=B, C4=B, D4=W.
    assert s.grid[2][2] == "player_1"  # C3 = W
    assert s.grid[2][3] == "player_0"  # D3 = B
    assert s.grid[3][2] == "player_0"  # C4 = B
    assert s.grid[3][3] == "player_1"  # D4 = W
    assert g.current_player(s) == "player_0"  # Black moves first


def test_coord_roundtrip():
    assert coord_to_rc("A1") == (0, 0)
    assert coord_to_rc("F6") == (5, 5)
    assert coord_to_rc("C3") == (2, 2)
    assert rc_to_coord(2, 2) == "C3"
    assert coord_to_rc("G1") is None  # out of range column
    assert coord_to_rc("A7") is None  # out of range row
    assert coord_to_rc("") is None


# --- AC-2.2: legal moves flip >= 1 opponent piece --------------------------

def test_legal_moves_all_flip_at_least_one():
    g = _game()
    s = g.initial_state(random.Random(0))
    legal = g.legal_actions(s, "player_0")
    assert PASS not in legal
    assert len(legal) == 4  # standard Othello opening has four legal moves
    for coord in legal:
        r, c = coord_to_rc(coord)
        captured = _flips(s.grid, r, c, "player_0")
        assert len(captured) >= 1
        ns = g.step(s, Move(type=coord))
        # The placed cell and every captured cell now belong to player_0.
        assert ns.grid[r][c] == "player_0"
        for (fr, fc) in captured:
            assert ns.grid[fr][fc] == "player_0"


def test_validate_rejects_occupied_and_nonflipping():
    g = _game()
    s = g.initial_state(random.Random(0))
    # Occupied center cell.
    ok, reason = g.validate_action(s, "player_0", Move(type="C3"))
    assert not ok
    # Empty corner that flips nothing at the start.
    ok, reason = g.validate_action(s, "player_0", Move(type="A1"))
    assert not ok
    # Amount carried on a discrete game is rejected.
    ok, reason = g.validate_action(s, "player_0", Move(type="D3", amount=1))
    assert not ok


# --- AC-2.3: pass only when no flipping move -------------------------------

def test_pass_only_when_no_move():
    g = _game()
    s = g.initial_state(random.Random(0))
    # At the start there are real moves -> pass is illegal.
    ok, reason = g.validate_action(s, "player_0", Move(type=PASS))
    assert not ok and reason == "pass_not_allowed"

    # Construct a board where player_0 (Black) has no flipping move: a board with
    # only Black pieces (no opponent to bracket) -> no legal move -> must pass.
    grid = [[None] * SIZE for _ in range(SIZE)]
    grid[0][0] = "player_0"
    blackonly = tuple(tuple(row) for row in grid)
    from aibattle.games.othello_lite import OthelloState
    s2 = OthelloState(grid=blackonly, to_act="player_0",
                      last_was_pass=False, done=False)
    assert g.legal_actions(s2, "player_0") == [PASS]
    ok, reason = g.validate_action(s2, "player_0", Move(type=PASS))
    assert ok
    ns = g.step(s2, Move(type=PASS))
    assert ns.grid == s2.grid  # board unchanged on a pass
    assert ns.to_act == "player_1"
    assert ns.last_was_pass is True


# --- AC-2.4: double-pass termination + returns by piece count --------------

def test_double_pass_terminates_and_returns():
    g = _game()
    from aibattle.games.othello_lite import OthelloState
    # Board with 3 Black, 1 White, no possible flips for anyone (isolated cells).
    grid = [[None] * SIZE for _ in range(SIZE)]
    grid[0][0] = "player_0"
    grid[0][5] = "player_0"
    grid[5][0] = "player_0"
    grid[5][5] = "player_1"
    g_t = tuple(tuple(row) for row in grid)

    # player_0 has no move -> passes (first pass).
    s = OthelloState(grid=g_t, to_act="player_0", last_was_pass=False, done=False)
    assert g.legal_actions(s, "player_0") == [PASS]
    s_after_p0 = g.step(s, Move(type=PASS))
    assert s_after_p0.done is False  # only one pass so far
    assert s_after_p0.to_act == "player_1"

    # player_1 also has no move -> second consecutive pass -> terminal.
    assert g.legal_actions(s_after_p0, "player_1") == [PASS]
    s_end = g.step(s_after_p0, Move(type=PASS))
    assert s_end.done is True
    r = g.returns(s_end)
    assert r == {"player_0": 1.0, "player_1": -1.0}  # 3 black > 1 white
    assert sum(r.values()) == 0.0


def test_returns_draw_on_equal_count():
    g = _game()
    from aibattle.games.othello_lite import OthelloState
    grid = [[None] * SIZE for _ in range(SIZE)]
    grid[0][0] = "player_0"
    grid[5][5] = "player_1"
    g_t = tuple(tuple(row) for row in grid)
    s = OthelloState(grid=g_t, to_act="player_0", last_was_pass=True, done=True)
    assert g.returns(s) == {"player_0": 0.0, "player_1": 0.0}


def test_non_terminal_when_move_exists():
    g = _game()
    s = g.initial_state(random.Random(0))
    assert g.is_terminal(s) is False


# --- template parsing ------------------------------------------------------

def test_template_parse_coord_and_pass():
    t = make_template("othello_lite_6x6")
    req = _empty_request(["C3", "D6", "E4"])
    assert t.parse("I will play C3", req) == Move(type="C3")
    assert t.parse("After analysis...\nE4", req) == Move(type="E4")
    assert t.parse("Z9 then maybe", req) is None  # nothing legal
    req_pass = _empty_request([PASS])
    assert t.parse("I have to pass here", req_pass) == Move(type="pass")


def test_fallback_is_neutral_random_legal():
    g = _game()
    s = g.initial_state(random.Random(0))
    legal = ["B2", "A1", "C5"]
    # The substitute move is always legal, deterministic for a given position,
    # and NOT a corner-preferring heuristic (no reward for invalid output).
    fb = g.fallback_action(s, "player_0", legal)
    assert fb.type in legal
    assert fb == g.fallback_action(s, "player_0", legal)  # reproducible
    assert g.fallback_action(s, "player_0", [PASS]) == Move(type=PASS)
    # Across positions/option sets the pick varies (it is not "always first"
    # or "always corner"): different inputs reach different choices.
    picks = set()
    for legal_set in (["B2", "A1", "C5"], ["A1", "B2"], ["C5", "A6", "F1"],
                      ["E4", "A1"], ["D6", "C5", "B2"]):
        picks.add(g.fallback_action(s, "player_0", legal_set).type)
    assert len(picks) > 1


def test_template_parse_last_mention_wins():
    t = make_template("othello_lite_6x6")
    req = _empty_request(["B2", "C5", "E4"])
    # Exact final line, with markdown/punctuation decoration.
    assert t.parse("thinking...\n**C5.**", req) == Move(type="C5")
    assert t.parse("reasoning\n- e4", req) == Move(type="E4")
    # Rejected-then-chosen on one line: the LAST legal coordinate wins.
    assert t.parse("B2 is tempting but C5 is better", req) == Move(type="C5")
    assert t.parse("Not B2. Not E4. Play C5", req) == Move(type="C5")


# --- full self-play smoke (engine-level) -----------------------------------

def test_random_selfplay_terminates():
    g = _game()
    rng = random.Random(123)
    for seed in range(20):
        s = g.initial_state(random.Random(seed))
        steps = 0
        while not g.is_terminal(s):
            player = g.current_player(s)
            legal = g.legal_actions(s, player)
            choice = rng.choice(legal)
            ok, _ = g.validate_action(s, player, Move(type=choice))
            assert ok, f"random legal pick rejected: {choice}"
            s = g.step(s, Move(type=choice))
            steps += 1
            assert steps < 200, "self-play did not terminate"
        r = g.returns(s)
        assert sum(r.values()) == 0.0
