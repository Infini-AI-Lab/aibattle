"""Tests for Leduc Poker (betting structure, showdown, zero-sum split)."""

from __future__ import annotations

import random

import pytest

from aibattle.games.registry import make_game, available_games
from aibattle.games.leduc import LeducPoker, LeducState, _BET_SIZE
from aibattle.agents.templates.registry import make_template
from aibattle.types import Move, AgentRequest, Observation


def _game():
    return LeducPoker()


def _drive(game, s, moves):
    """Apply a sequence of (action[, amount]) tuples, returning the final state."""
    for m in moves:
        if isinstance(m, tuple):
            mv = Move(type=m[0], amount=m[1])
        else:
            mv = Move(type=m)
        ok, reason = game.validate_action(s, game.current_player(s), mv)
        assert ok, f"invalid move {m}: {reason}"
        s = game.step(s, mv)
    return s


# --- AC-1 (Leduc slice): registry -----------------------------------------

def test_registry():
    g = make_game("leduc_poker")
    assert isinstance(g, LeducPoker)
    assert "leduc_poker" in available_games()
    assert make_template("leduc_poker") is not None
    assert make_template("leduc_poker", coached=True).coaching(
        AgentRequest("leduc_poker", "1.0.0", "player_0",
                     Observation("player_0", {"card": "K"}, {}, [], ["check"], ""),
                     "", 0)
    ).strip() != ""
    with pytest.raises(ValueError):
        make_game("leduc")


# --- AC-4.1: deck, deal, structure ----------------------------------------

def test_initial_deal():
    g = _game()
    s = g.initial_state(random.Random(3))
    assert set(s.cards.keys()) == {"player_0", "player_1"}
    for c in s.cards.values():
        assert c in ("J", "Q", "K")
    assert s.locked == {"player_0": 1, "player_1": 1}  # antes
    assert s.public is None
    assert s.round == 0
    assert g.current_player(s) == "player_0"
    assert g.legal_actions(s, "player_0") == ["check", "bet"]


def test_check_check_reveals_public_and_advances_round():
    g = _game()
    s = g.initial_state(random.Random(3))
    s = _drive(g, s, ["check", "check"])
    assert s.round == 1
    assert s.public in ("J", "Q", "K")
    assert s.public == s.pending_public  # revealed card == the one dealt at setup
    assert not s.done
    assert g.legal_actions(s, "player_0") == ["check", "bet"]


def test_public_card_is_random_dealt_not_lowest_remaining():
    g = _game()
    # The public card must be the third dealt card (random), not a deterministic
    # function of the two hole cards. Across seeds, all three ranks should appear
    # as the revealed public card, including ranks higher than both hole cards.
    seen = set()
    saw_higher_than_holes = False
    for seed in range(200):
        s = g.initial_state(random.Random(seed))
        s = _drive(g, s, ["check", "check"])
        seen.add(s.public)
        from aibattle.games.leduc import _RANK
        holes = [s.cards["player_0"], s.cards["player_1"]]
        if _RANK[s.public] > max(_RANK[h] for h in holes):
            saw_higher_than_holes = True
    assert seen == {"J", "Q", "K"}
    # The buggy "lowest remaining" rule could never reveal a card higher than
    # both holes; a correct random deal does.
    assert saw_higher_than_holes


def test_bet_amount_is_total_commitment_and_one_raise_cap():
    g = _game()
    s = g.initial_state(random.Random(3))
    # Round 0 bet size is 2; bet total = your_commit(0)+2 = 2.
    ok, _ = g.validate_action(s, "player_0", Move(type="bet", amount=2))
    assert ok
    # Wrong amount rejected.
    assert g.validate_action(s, "player_0", Move(type="bet", amount=3))[0] is False
    assert g.validate_action(s, "player_0", Move(type="bet", amount=4))[0] is False
    # bet without amount rejected.
    assert g.validate_action(s, "player_0", Move(type="bet"))[0] is False

    s = g.step(s, Move(type="bet", amount=2))   # p0 bets to 2
    # p1 faces a bet: legal fold/call/raise.
    assert set(g.legal_actions(s, "player_1")) == {"fold", "call", "raise"}
    # raise total = cur_max(2)+2 = 4.
    assert g.validate_action(s, "player_1", Move(type="raise", amount=4))[0] is True
    assert g.validate_action(s, "player_1", Move(type="raise", amount=6))[0] is False
    s = g.step(s, Move(type="raise", amount=4))  # p1 raises to 4
    # One raise used: p0 may now only fold or call (no second raise).
    assert set(g.legal_actions(s, "player_0")) == {"fold", "call"}


def test_check_then_bet_is_legal_and_round_closes_on_call():
    g = _game()
    s = g.initial_state(random.Random(3))
    s = g.step(s, Move(type="check"))            # p0 checks
    assert set(g.legal_actions(s, "player_1")) == {"check", "bet"}
    s = g.step(s, Move(type="bet", amount=2))    # p1 bets
    assert set(g.legal_actions(s, "player_0")) == {"fold", "call", "raise"}
    s = g.step(s, Move(type="call"))             # p0 calls -> round closes
    assert s.round == 1


# --- AC-4.2 + AC-4.3: showdown + zero-sum ----------------------------------

def _showdown_state(c0, c1, public, locked):
    return LeducState(
        cards={"player_0": c0, "player_1": c1}, public=public,
        pending_public=public, round=1,
        street_commit={"player_0": 0, "player_1": 0}, locked=dict(locked),
        to_act="player_0", raises_this_round=0, acted=("player_0", "player_1"),
        folded=None, done=True,
    )


def test_showdown_pair_beats_non_pair():
    g = _game()
    # p1 has Q which pairs the public Q; p0 has K (higher rank but no pair).
    s = _showdown_state("K", "Q", "Q", {"player_0": 3, "player_1": 3})
    r = g.returns(s)
    assert r["player_1"] > 0 and r["player_0"] < 0
    assert sum(r.values()) == 0.0


def test_showdown_higher_card_when_no_pair():
    g = _game()
    s = _showdown_state("K", "J", "Q", {"player_0": 3, "player_1": 3})
    r = g.returns(s)
    assert r["player_0"] > 0 and r["player_1"] < 0
    assert sum(r.values()) == 0.0


def test_showdown_split_equal_strength_zero_sum():
    g = _game()
    # Same private rank, no pair with public -> split. Equal contributions -> 0/0.
    s = _showdown_state("Q", "Q", "K", {"player_0": 3, "player_1": 3})
    r = g.returns(s)
    assert r == {"player_0": 0.0, "player_1": 0.0}


def test_fold_payoff_zero_sum():
    g = _game()
    s = g.initial_state(random.Random(3))
    s = g.step(s, Move(type="bet", amount=2))    # p0 bets to 2 (locked still antes)
    s = g.step(s, Move(type="fold"))             # p1 folds
    assert s.done and s.folded == "player_1"
    r = g.returns(s)
    assert sum(r.values()) == 0.0
    assert r["player_0"] > 0 and r["player_1"] < 0


def test_fold_after_raise_pays_full_contribution_round1():
    g = _game()
    s = g.initial_state(random.Random(3))
    s = g.step(s, Move(type="bet", amount=2))    # p0: ante 1 + bet 2 = 3 committed
    s = g.step(s, Move(type="raise", amount=4))  # p1: ante 1 + raise-to 4 = 5
    s = g.step(s, Move(type="fold"))             # p0 folds
    assert s.done and s.folded == "player_0"
    r = g.returns(s)
    # Folder (p0) loses its full contribution 3; raiser (p1) nets +3.
    assert r == {"player_0": -3.0, "player_1": 3.0}
    assert sum(r.values()) == 0.0


def test_fold_after_raise_round2_includes_locked_and_live():
    g = _game()
    s = g.initial_state(random.Random(3))
    s = _drive(g, s, ["check", "check"])         # round 1 closes (each locked 1)
    assert s.round == 1
    s = g.step(s, Move(type="bet", amount=4))    # p0 bets to 4 (round2 size 4)
    s = g.step(s, Move(type="raise", amount=8))  # p1 raises to 8
    s = g.step(s, Move(type="fold"))             # p0 folds
    assert s.done and s.folded == "player_0"
    r = g.returns(s)
    # p0 contribution = locked 1 + live 4 = 5; p1 = locked 1 + live 8 = 9.
    # Folder loses its 5; opponent nets +5.
    assert r == {"player_0": -5.0, "player_1": 5.0}
    assert sum(r.values()) == 0.0


def test_betting_history_visible_to_next_player():
    g = _game()
    s = g.initial_state(random.Random(3))
    # Before any action, history is empty.
    assert g.observation(s, "player_0").history == []
    s = g.step(s, Move(type="bet", amount=2))    # p0 bets
    # player_1 (next to act) sees player_0's bet in the public history.
    o1 = g.observation(s, "player_1")
    assert o1.history == [{"player": "player_0", "action": "bet", "to": 2}]
    assert "bet" in o1.rendered
    s = g.step(s, Move(type="raise", amount=4))  # p1 raises
    s = g.step(s, Move(type="call"))             # p0 calls -> round closes, reveal
    o = g.observation(s, "player_0")
    # History now includes both bets and the public-card reveal event.
    actions = [r.get("action") for r in o.history if "action" in r]
    assert actions == ["bet", "raise", "call"]
    assert any(r.get("event") == "public_card" for r in o.history)


def test_returns_always_zero_sum_random_play():
    g = _game()
    rng = random.Random(0)
    for seed in range(100):
        s = g.initial_state(random.Random(seed))
        guard = 0
        while not g.is_terminal(s):
            p = g.current_player(s)
            legal = g.legal_actions(s, p)
            atype = rng.choice(legal)
            if atype in ("bet", "raise"):
                pub = g.observation(s, p).public
                size = pub["bet_size"]
                cur_max = pub["your_commit"] + pub["to_call"]
                amt = pub["your_commit"] + size if atype == "bet" else cur_max + size
                mv = Move(type=atype, amount=amt)
            else:
                mv = Move(type=atype)
            ok, reason = g.validate_action(s, p, mv)
            assert ok, f"random legal pick rejected: {atype}: {reason}"
            s = g.step(s, mv)
            guard += 1
            assert guard < 30
        r = g.returns(s)
        assert abs(r["player_0"] + r["player_1"]) < 1e-9


def test_split_odd_chip_to_player_0():
    g = _game()
    # Construct an odd pot with equal-strength split: contributions 3 and 2 -> pot 5.
    # (Unequal contributions only arise from a fold normally, but we test the
    # odd-chip rule directly here.)
    s = LeducState(
        cards={"player_0": "Q", "player_1": "Q"}, public="K", pending_public="K",
        round=1, street_commit={"player_0": 0, "player_1": 0},
        locked={"player_0": 3, "player_1": 2}, to_act="player_0",
        raises_this_round=0, acted=("player_0", "player_1"), folded=None, done=True,
    )
    r = g.returns(s)
    assert sum(r.values()) == 0.0  # zero-sum preserved
    # pot=5, half=2, odd=1 -> p0 gets 2+1-3 = 0, p1 gets 2-2 = 0... verify exact
    assert r["player_0"] == 0.0 and r["player_1"] == 0.0


# --- private card hidden from opponent -------------------------------------

def test_observation_hides_opponent_card():
    g = _game()
    # Use a seed where the two private cards differ so the check is meaningful.
    s = None
    for seed in range(20):
        cand = g.initial_state(random.Random(seed))
        if cand.cards["player_0"] != cand.cards["player_1"]:
            s = cand
            break
    assert s is not None
    o0 = g.observation(s, "player_0")
    # Player_0 sees only its own card in private, and the structured observation
    # exposes nothing keyed to the opponent's hidden card.
    assert o0.private == {"card": s.cards["player_0"]}
    assert "player_1" not in o0.private
    assert "card" not in o0.public  # the opponent card is never in public state
    # The opponent's card is not derivable from the public dict.
    assert s.cards["player_1"] not in [v for v in o0.public.values()
                                       if isinstance(v, str)]


# --- template parsing ------------------------------------------------------

def test_template_parse_fills_amount():
    t = make_template("leduc_poker")
    # Facing no bet, round 1 (bet_size 2), your_commit 0.
    obs = Observation("player_0", {"card": "K"},
                      {"bet_size": 2, "your_commit": 0, "to_call": 0, "round": 1},
                      [], ["check", "bet"], "")
    req = AgentRequest("leduc_poker", "1.0.0", "player_0", obs, "", 0)
    assert t.parse("I check", req) == Move(type="check")
    assert t.parse("bet", req) == Move(type="bet", amount=2)
    # Facing a bet (to_call 2), raise-to should be cur_max(2)+2 = 4.
    obs2 = Observation("player_0", {"card": "K"},
                       {"bet_size": 2, "your_commit": 0, "to_call": 2, "round": 1},
                       [], ["fold", "call", "raise"], "")
    req2 = AgentRequest("leduc_poker", "1.0.0", "player_0", obs2, "", 0)
    assert t.parse("raise", req2) == Move(type="raise", amount=4)
    assert t.parse("call", req2) == Move(type="call")
    assert t.parse("fold", req2) == Move(type="fold")
    assert t.parse("uhh", req2) is None


def test_template_parse_last_mention_wins():
    t = make_template("leduc_poker")
    obs = Observation("player_0", {"card": "K"},
                      {"bet_size": 2, "your_commit": 0, "to_call": 2, "round": 1},
                      [], ["fold", "call", "raise"], "")
    req = AgentRequest("leduc_poker", "1.0.0", "player_0", obs, "", 0)
    # A rejected action mentioned before the chosen one must NOT win: the
    # latest-mentioned legal action is the conclusion.
    assert t.parse("I shouldn't raise here, just call", req) == Move(type="call")
    assert t.parse("Folding is too weak. I call", req) == Move(type="call")
    # Exact final line with decoration parses via the fast path.
    assert t.parse("reasoning about raising...\n**call**", req) == Move(type="call")
    assert t.parse("blah\n- raise:", req) == Move(type="raise", amount=4)
