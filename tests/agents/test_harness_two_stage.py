"""Two-stage estimate -> decide harness tests (AC3)."""

from __future__ import annotations

from aibattle.agents.local.two_stage import TwoStageAgent
from aibattle.agents.templates.kuhn import KuhnTemplate
from tests.conftest import FakeModelClient


def _agent(scripted, **kw):
    return TwoStageAgent(client=FakeModelClient(scripted),
                         template=KuhnTemplate(), name="ts", **kw)


async def test_two_calls_estimate_then_decide(make_request):
    req = make_request()
    agent = _agent(["opponent likely has a weak card", "bet"])
    resp = await agent.act(req)
    assert resp.action == "bet"
    assert agent.client.call_count == 2
    # Call 1 carries the (game-neutral) assessment instruction.
    assert "assess the opponent" in agent.client.calls[0]["prompt"].lower()
    # Call 2 (decision) carries the assessment text produced in stage 1.
    assert "opponent likely has a weak card" in agent.client.calls[1]["prompt"]


async def test_estimate_recorded_in_metadata(make_request):
    req = make_request()
    agent = _agent(["RANGE_ESTIMATE_TEXT", "check"])
    resp = await agent.act(req)
    assert resp.metadata["harness"]["kind"] == "two_stage"
    assert resp.metadata["harness"]["estimate"] == "RANGE_ESTIMATE_TEXT"


async def test_estimate_prompt_override(make_request):
    req = make_request()
    agent = _agent(["est", "bet"], estimate_prompt="GUESS_THE_OPPONENT_XYZZY")
    await agent.act(req)
    assert "GUESS_THE_OPPONENT_XYZZY" in agent.client.calls[0]["prompt"]


async def test_decision_repairs_on_garbage(make_request):
    req = make_request()
    # estimate, then garbage decision, then a good decision via repair.
    agent = _agent(["est", "garbage", "check"], max_retries=2)
    resp = await agent.act(req)
    assert resp.action == "check"
    assert agent.client.call_count == 3
