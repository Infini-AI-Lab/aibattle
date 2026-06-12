"""Tests for Independent Blackjack (totals, dealer policy, turn flow, scoring)."""

from __future__ import annotations

import asyncio
import random

import pytest

from aibattle.games.registry import make_game, available_games
from aibattle.games.blackjack import (
    IndependentBlackjack, BlackjackState,
    hand_total, is_bust, is_natural, dealer_should_hit,
)
from aibattle.agents.templates.registry import make_template
from aibattle.agents.registry import make_agent, _BUILTINS
from aibattle.types import Move, AgentRequest, Observation


def _game():
    return IndependentBlackjack()


def _state(player, dealer, phase="player", doubled=False, deck=None, idx=0):
    deck = tuple(deck or [])
    return BlackjackState(deck=deck, player=tuple(player), dealer=tuple(dealer),
                          phase=phase, doubled=doubled, draw_index=idx)


# --- AC-1 (Blackjack slice): registry --------------------------------------

def test_registry():
    g = make_game("independent_blackjack")
    assert isinstance(g, IndependentBlackjack)
    assert "independent_blackjack" in available_games()
    assert make_template("independent_blackjack") is not None
    assert make_template("independent_blackjack", coached=True).coaching(
        AgentRequest("independent_blackjack", "1.0.0", "player_0",
                     Observation("player_0", {}, {}, [], ["hit"], ""), "", 0)
    ).strip() != ""
    assert "blackjack_dealer" in _BUILTINS
    assert "blackjack_random" in _BUILTINS


# --- ace-aware totals ------------------------------------------------------

def test_hand_total_ace_aware():
    assert hand_total(["A", "9"]) == (20, True)       # soft 20
    assert hand_total(["A", "6"]) == (17, True)        # soft 17
    assert hand_total(["A", "6", "K"]) == (17, False)  # ace demoted -> hard 17
    assert hand_total(["10", "7"]) == (17, False)      # hard 17
    assert hand_total(["A", "A", "9"]) == (21, True)   # one ace 11, one ace 1
    assert is_bust(["K", "Q", "5"]) is True
    assert is_natural(["A", "K"]) is True
    assert is_natural(["A", "5", "5"]) is False        # three-card 21 not natural


# --- AC-3.2: dealer policy (stand on all 17s incl. soft 17) ----------------

def test_dealer_policy_soft_17():
    assert dealer_should_hit(["10", "6"]) is True       # 16 -> hit
    assert dealer_should_hit(["10", "7"]) is False       # hard 17 -> stand
    assert dealer_should_hit(["A", "6"]) is False        # soft 17 -> STAND
    assert dealer_should_hit(["A", "5"]) is True         # soft 16 -> hit
    assert dealer_should_hit(["A", "7"]) is False        # soft 18 -> stand


def test_dealer_agent_uses_policy_and_no_llm():
    agent = make_agent({"type": "builtin", "name": "blackjack_dealer"},
                       game_name="independent_blackjack")
    # Build an observation as the game would expose at the dealer's turn.
    obs = Observation("player_1", {}, {"dealer_hand": ["10", "6"],
                                       "dealer_total": 16, "player_hand": ["10", "8"],
                                       "player_total": 18},
                      [], ["hit", "stand"], "")
    req = AgentRequest("independent_blackjack", "1.0.0", "player_1", obs, "", 0)
    resp = asyncio.run(agent.act(req))
    assert resp.action == "hit"          # 16 -> hit
    # Soft 17 -> stand.
    obs2 = Observation("player_1", {}, {"dealer_hand": ["A", "6"], "dealer_total": 17,
                                        "player_hand": ["10", "8"], "player_total": 18},
                       [], ["hit", "stand"], "")
    req2 = AgentRequest("independent_blackjack", "1.0.0", "player_1", obs2, "", 0)
    resp2 = asyncio.run(agent.act(req2))
    assert resp2.action == "stand"
    # The dealer agent is a builtin and holds no model client.
    assert agent.agent_type == "builtin"
    assert not hasattr(agent, "client")


# --- AC-3.1: turn flow -----------------------------------------------------

def test_players_and_initial_turn():
    g = _game()
    assert g.players == ["player_0", "player_1"]
    s = g.initial_state(random.Random(1))
    # If no natural was dealt, the player acts first.
    if not is_natural(s.player) and not is_natural(s.dealer):
        assert g.current_player(s) == "player_0"
        assert "hit" in g.legal_actions(s, "player_0")


def test_stand_hands_over_to_dealer_when_dealer_must_draw():
    g = _game()
    # Dealer 14 (<17) must draw, so stand hands over to the dealer.
    s = _state(["10", "7"], ["9", "5"], phase="player")
    ns = g.step(s, Move(type="stand"))
    assert ns.phase == "dealer"
    assert g.current_player(ns) == "player_1"


def test_stand_terminal_when_dealer_already_17_plus():
    g = _game()
    # AC-3.1: the dealer acts only when it must draw. With the dealer already at
    # a standing total, a player stand goes straight to terminal — no dealer step.
    for dealer in (["10", "7"],        # hard 17 -> stand
                   ["A", "6"],          # soft 17 -> stand
                   ["10", "10"],        # 20 -> stand
                   ["K", "9"]):         # 19 -> stand
        s = _state(["10", "8"], dealer, phase="player")
        ns = g.step(s, Move(type="stand"))
        assert ns.phase == "done", f"dealer {dealer} should not act"
        assert g.is_terminal(ns)


def test_double_terminal_when_dealer_already_17_plus():
    g = _game()
    # Non-busting double with the dealer already standing -> terminal, no dealer step.
    s = _state(["5", "6"], ["10", "7"], phase="player", deck=["7"], idx=0)  # dealer hard 17
    ns = g.step(s, Move(type="double"))  # player -> 18, not bust
    assert ns.doubled is True
    assert not is_bust(ns.player)
    assert ns.phase == "done"
    assert g.is_terminal(ns)
    # Player 18 vs dealer 17, doubled -> +2.
    assert g.returns(ns) == {"player_0": 2.0, "player_1": -2.0}


def test_player_bust_is_terminal_no_dealer_turn():
    g = _game()
    # Player hits into a bust; hand must be terminal and NOT go to the dealer.
    s = _state(["K", "8"], ["9", "5"], phase="player", deck=["7"], idx=0)
    ns = g.step(s, Move(type="hit"))   # draws "7" -> 25 bust
    assert is_bust(ns.player)
    assert ns.phase == "done"
    assert g.is_terminal(ns)
    assert g.current_player(ns) == "player_0"  # never advanced to dealer


def test_double_then_bust_terminal_minus_two():
    g = _game()
    s = _state(["K", "6"], ["9", "5"], phase="player", deck=["Q"], idx=0)
    ns = g.step(s, Move(type="double"))  # draws "Q" -> 26 bust, doubled
    assert ns.doubled is True
    assert is_bust(ns.player)
    assert ns.phase == "done"
    r = g.returns(ns)
    assert r == {"player_0": -2.0, "player_1": 2.0}


def test_double_draws_exactly_one_then_dealer():
    g = _game()
    s = _state(["5", "6"], ["9", "5"], phase="player", deck=["7"], idx=0)
    ns = g.step(s, Move(type="double"))  # draws "7" -> 18, not bust
    assert ns.doubled is True
    assert ns.phase == "dealer"          # forced stand -> dealer's turn
    assert len(ns.player) == 3


# --- AC-3.3: scoring -------------------------------------------------------

def test_scoring_normal_win_loss_push():
    g = _game()
    # Player 20 vs dealer 18 -> win +1.
    assert g.returns(_state(["10", "10"], ["10", "8"], phase="done")) == \
        {"player_0": 1.0, "player_1": -1.0}
    # Player 17 vs dealer 20 -> loss -1.
    assert g.returns(_state(["10", "7"], ["10", "10"], phase="done")) == \
        {"player_0": -1.0, "player_1": 1.0}
    # Push.
    assert g.returns(_state(["10", "9"], ["10", "9"], phase="done")) == \
        {"player_0": 0.0, "player_1": 0.0}


def test_scoring_dealer_bust_pays_player():
    g = _game()
    assert g.returns(_state(["10", "7"], ["K", "Q", "5"], phase="done")) == \
        {"player_0": 1.0, "player_1": -1.0}


def test_scoring_naturals():
    g = _game()
    # Player natural only -> +1.5.
    assert g.returns(_state(["A", "K"], ["10", "7"], phase="done")) == \
        {"player_0": 1.5, "player_1": -1.5}
    # Both natural -> push 0.
    assert g.returns(_state(["A", "K"], ["A", "Q"], phase="done")) == \
        {"player_0": 0.0, "player_1": 0.0}
    # Dealer natural only -> -1.
    assert g.returns(_state(["10", "7"], ["A", "K"], phase="done")) == \
        {"player_0": -1.0, "player_1": 1.0}
    # A three-card 21 is NOT a natural: player 21 (3 cards) vs dealer natural
    # -> dealer natural wins -> -1.
    assert g.returns(_state(["7", "7", "7"], ["A", "K"], phase="done")) == \
        {"player_0": -1.0, "player_1": 1.0}


def test_returns_always_zero_sum():
    g = _game()
    for seed in range(50):
        s = g.initial_state(random.Random(seed))
        # Drive the hand to terminal with a fixed policy (always stand, then dealer).
        guard = 0
        while not g.is_terminal(s):
            p = g.current_player(s)
            legal = g.legal_actions(s, p)
            mv = Move(type="stand") if "stand" in legal else Move(type=legal[0])
            s = g.step(s, mv)
            guard += 1
            assert guard < 50
        r = g.returns(s)
        assert abs(r["player_0"] + r["player_1"]) < 1e-9


# --- AC-3.1: observation hides the hole card during player's turn ----------

def test_dealer_legal_actions_are_policy_tight():
    g = _game()
    # The dealer phase only exists while the dealer must draw, so the only legal
    # dealer action is hit (never an advisory 'stand' that could still cause a hit).
    s = _state(["10", "7"], ["9", "5"], phase="dealer")  # dealer 14 must hit
    assert g.legal_actions(s, "player_1") == ["hit"]


def test_render_perspective_hides_hole_card():
    g = _game()
    s = _state(["10", "7"], ["9", "5"], phase="player")  # hole card is "5"
    # Perspective render must not reveal the dealer hole card during player turn.
    pr = g.render(s, perspective="player_0")
    assert "Dealer shows: 9" in pr
    assert "5" not in pr.split("Dealer shows:")[1]
    # Full-information render (no perspective) also hides the hole during player turn.
    full = g.render(s)
    assert "[hidden]" in full
    # Once the dealer's turn begins, the full render reveals the hand.
    sd = _state(["10", "7"], ["9", "5"], phase="dealer")
    assert "[hidden]" not in g.render(sd)


def test_player_observation_hides_hole_card():
    g = _game()
    s = _state(["10", "7"], ["9", "5"], phase="player")
    obs = g.observation(s, "player_0")
    # Player sees only the dealer upcard, not the full dealer hand.
    assert obs.public.get("dealer_upcard") == "9"
    assert "dealer_hand" not in obs.public
    assert "5" not in obs.rendered.split("Dealer shows:")[1] if "Dealer shows:" in obs.rendered else True
    # At the dealer's turn the dealer hand becomes visible.
    sd = _state(["10", "7"], ["9", "5"], phase="dealer")
    obsd = g.observation(sd, "player_1")
    assert obsd.public.get("dealer_hand") == ["9", "5"]


# --- template parsing ------------------------------------------------------

def test_template_parse():
    t = make_template("independent_blackjack")
    req = AgentRequest("independent_blackjack", "1.0.0", "player_0",
                       Observation("player_0", {}, {}, [],
                                   ["hit", "stand", "double"], ""), "", 0)
    assert t.parse("I'll hit", req) == Move(type="hit")
    assert t.parse("reasoning...\nstand", req) == Move(type="stand")
    assert t.parse("Let's double down here", req) == Move(type="double")
    assert t.parse("no idea", req) is None


def test_template_parse_last_mention_wins_and_no_broad_aliases():
    t = make_template("independent_blackjack")
    req = AgentRequest("independent_blackjack", "1.0.0", "player_0",
                       Observation("player_0", {}, {}, [],
                                   ["hit", "stand", "double"], ""), "", 0)
    # The latest-mentioned legal action is the conclusion.
    assert t.parse("Don't double here, just stand", req) == Move(type="stand")
    assert t.parse("Standing is too passive — hit", req) == Move(type="hit")
    # Exact final line with decoration parses via the fast path.
    assert t.parse("thinking...\n**Stand.**", req) == Move(type="stand")
    # 'take'/'draw' are no longer hit aliases: incidental prose must not parse.
    assert t.parse("take the upcard into account", req) is None
    assert t.parse("the dealer could draw a ten", req) is None
