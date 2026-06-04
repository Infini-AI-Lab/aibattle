"""End-to-end: harness agents are drop-in Agents through the real Runner (AC6).

Fully offline: a ConstantModelClient drives the harness, the real Kuhn game and
Runner orchestrate the match, and MatchLogger(None) / episode_dir=None means no
filesystem writes. Steps where the constant action is illegal flow through the
runner's invalid-action fallback policy, so the match still completes.
"""

from __future__ import annotations

import pytest

from aibattle.agents.local.cot import StructuredCoTAgent
from aibattle.agents.local.two_stage import TwoStageAgent
from aibattle.agents.registry import make_agent
from aibattle.agents.templates.kuhn import KuhnTemplate
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner
from tests.conftest import ConstantModelClient

pytestmark = pytest.mark.integration


def _runner():
    return Runner(lambda: make_game("kuhn_poker"), on_invalid_action="fallback")


async def test_cot_harness_plays_full_kuhn_match():
    agent_a = StructuredCoTAgent(client=ConstantModelClient("check"),
                                 template=KuhnTemplate(), name="cot-a")
    agent_b = make_agent({"type": "builtin", "name": "random"},
                         game_name="kuhn_poker", seed=7)
    with MatchLogger(None) as lg:
        res = await _runner().run_match(
            agent_a, agent_b, episodes=6, seed=123, seat_swap=True, logger=lg)
    assert len(res.episodes) == 6
    assert res.failures == 0
    for ep in res.episodes:
        # Kuhn is zero-sum.
        assert abs(sum(ep["returns"].values())) < 1e-9


async def test_two_stage_harness_drop_in_runs():
    agent_a = TwoStageAgent(client=ConstantModelClient("bet"),
                            template=KuhnTemplate(), name="ts-a")
    agent_b = TwoStageAgent(client=ConstantModelClient("check"),
                            template=KuhnTemplate(), name="ts-b")
    with MatchLogger(None) as lg:
        res = await _runner().run_match(
            agent_a, agent_b, episodes=4, seed=42, seat_swap=True, logger=lg)
    assert len(res.episodes) == 4
    assert res.failures == 0


async def test_harness_invalid_action_gets_fallback_policy():
    # "bet" is illegal at Kuhn call/fold nodes -> harness returns INVALID ->
    # runner fallback substitutes a legal move; the match still completes.
    agent_a = StructuredCoTAgent(client=ConstantModelClient("bet"),
                                 template=KuhnTemplate(), name="cot-bet")
    agent_b = make_agent({"type": "builtin", "name": "kuhn_heuristic"},
                         game_name="kuhn_poker", seed=3)
    with MatchLogger(None) as lg:
        res = await _runner().run_match(
            agent_a, agent_b, episodes=6, seed=99, seat_swap=True, logger=lg)
    assert res.failures == 0
    assert len(res.episodes) == 6
