"""Game-agnostic default-prompt contract (AC3/AC5).

These tests pin that the harnesses' DEFAULT intermediate prompts carry no
poker-specific vocabulary when the underlying template is a board game
(Connect Four / Gomoku). This is what makes the harnesses reusable across all
games — the game-specific content must come from the template, not from the
harness's own wording. Custom-prompt overrides are still honored verbatim.
"""

from __future__ import annotations

import re

import pytest

from aibattle.agents.local.cot import StructuredCoTAgent
from aibattle.agents.local.self_refine import SelfRefineAgent
from aibattle.agents.local.two_stage import TwoStageAgent
from aibattle.agents.templates.connect4 import Connect4Template
from aibattle.agents.templates.gomoku import GomokuTemplate
from tests.conftest import FakeModelClient

# Poker-only vocabulary that must NOT appear in a board game's prompts.
_POKER_WORDS = ["hand", "card", "pot", "odds", "range", "bet", "call", "fold",
                "raise", "chip", "hole", "ante", "bluff", "ev"]


def _assert_no_poker_words(text: str):
    low = text.lower()
    hits = [w for w in _POKER_WORDS if re.search(rf"\b{w}\b", low)]
    assert not hits, f"prompt contains poker-only vocabulary {hits}: {text!r}"


def _board_templates():
    return [("connect4", Connect4Template()), ("gomoku", GomokuTemplate())]


@pytest.mark.parametrize("game,template", _board_templates())
async def test_cot_default_prompt_is_game_neutral(make_board_request, game, template):
    req = make_board_request(game=game)
    client = FakeModelClient([req.observation.legal_actions[0]])
    agent = StructuredCoTAgent(client=client, template=template, name="cot")
    await agent.act(req)
    # The full prompt the model saw = template render + the harness's default CoT.
    _assert_no_poker_words(client.calls[0]["prompt"])


@pytest.mark.parametrize("game,template", _board_templates())
async def test_two_stage_default_prompts_are_game_neutral(make_board_request, game, template):
    req = make_board_request(game=game)
    legal = req.observation.legal_actions[0]
    # Stage-1 assessment text is echoed into stage-2; keep it neutral too.
    client = FakeModelClient(["the opponent is building toward the center", legal])
    agent = TwoStageAgent(client=client, template=template, name="ts")
    await agent.act(req)
    # Both the estimate prompt (call 0) and the decision prompt (call 1) — which
    # carries the stage-2 label — must be free of poker vocabulary.
    _assert_no_poker_words(client.calls[0]["prompt"])
    _assert_no_poker_words(client.calls[1]["prompt"])


@pytest.mark.parametrize("game,template", _board_templates())
async def test_self_refine_default_prompts_are_game_neutral(make_board_request, game, template):
    req = make_board_request(game=game)
    legal = req.observation.legal_actions[0]
    client = FakeModelClient(["draft center", "critique text", "revised", legal])
    agent = SelfRefineAgent(client=client, template=template, name="sr", rounds=1)
    await agent.act(req)
    for c in client.calls:
        _assert_no_poker_words(c["prompt"])


async def test_custom_cot_override_passes_through_on_board(make_board_request):
    req = make_board_request(game="connect4")
    client = FakeModelClient([req.observation.legal_actions[0]])
    agent = StructuredCoTAgent(client=client, template=Connect4Template(),
                               name="cot", cot_instructions="CUSTOM_BOARD_HINT")
    await agent.act(req)
    assert "CUSTOM_BOARD_HINT" in client.calls[0]["prompt"]


async def test_custom_estimate_override_passes_through_on_board(make_board_request):
    req = make_board_request(game="gomoku")
    legal = req.observation.legal_actions[0]
    client = FakeModelClient(["assessment", legal])
    agent = TwoStageAgent(client=client, template=GomokuTemplate(),
                          name="ts", estimate_prompt="CUSTOM_ASSESS_XYZZY")
    await agent.act(req)
    assert "CUSTOM_ASSESS_XYZZY" in client.calls[0]["prompt"]
