"""Self-Refine (draft -> critique -> revise) harness tests (AC3)."""

from __future__ import annotations

from aibattle.agents.local.self_refine import SelfRefineAgent
from aibattle.agents.templates.kuhn import KuhnTemplate
from tests.conftest import FakeModelClient


def _agent(scripted, **kw):
    return SelfRefineAgent(client=FakeModelClient(scripted),
                           template=KuhnTemplate(), name="sr", **kw)


async def test_draft_critique_revise_decide_order(make_request):
    req = make_request()
    # rounds=1 -> draft, critique, revise, then final decision = 4 calls.
    agent = _agent(["DRAFT bet", "CRITIQUE consider check", "REVISE check", "check"],
                   rounds=1)
    resp = await agent.act(req)
    assert resp.action == "check"
    assert agent.client.call_count == 4
    # The final decision prompt must carry the critique text.
    assert "CRITIQUE consider check" in agent.client.calls[3]["prompt"]


async def test_rounds_controls_critique_count(make_request):
    req = make_request()
    # rounds=2 -> draft + 2*(critique+revise) + final = 1 + 4 + 1 = 6 calls.
    agent = _agent(["d", "c1", "r1", "c2", "r2", "check"], rounds=2)
    resp = await agent.act(req)
    assert resp.action == "check"
    assert agent.client.call_count == 6
    assert len(resp.metadata["harness"]["history"]) == 2


async def test_history_recorded(make_request):
    req = make_request()
    agent = _agent(["DRAFT", "CRIT", "REV", "bet"], rounds=1)
    resp = await agent.act(req)
    h = resp.metadata["harness"]
    assert h["kind"] == "self_refine"
    assert h["history"][0]["draft"] == "DRAFT"
    assert h["history"][0]["critique"] == "CRIT"


async def test_critique_prompt_override(make_request):
    req = make_request()
    agent = _agent(["d", "c", "r", "bet"], rounds=1,
                   critique_prompt="CRITIQUE_XYZZY")
    await agent.act(req)
    # Critique is the 2nd call (index 1).
    assert "CRITIQUE_XYZZY" in agent.client.calls[1]["prompt"]
