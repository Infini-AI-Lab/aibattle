"""Self-Consistency (majority-vote) harness tests (AC3)."""

from __future__ import annotations

from aibattle.agents.local.self_consistency import SelfConsistencyAgent
from aibattle.agents.templates.kuhn import KuhnTemplate
from aibattle.models.base import ModelOutput
from aibattle.types import INVALID
from tests.conftest import FakeModelClient


def _agent(scripted, **kw):
    return SelfConsistencyAgent(client=FakeModelClient(scripted),
                                template=KuhnTemplate(), name="sc", **kw)


async def test_majority_vote_wins(make_request):
    req = make_request()
    agent = _agent(["bet", "bet", "check", "bet", "check"], n=5, temperature=0.7)
    resp = await agent.act(req)
    assert resp.action == "bet"  # 3 bets vs 2 checks
    h = resp.metadata["harness"]
    assert h["kind"] == "self_consistency"
    assert h["parsed"] == 5
    assert h["votes"] == {"bet": 3, "check": 2}
    assert h["winner"] == "bet"


async def test_n_and_temperature_forwarded(make_request):
    req = make_request()
    agent = _agent(["check"] * 7, n=7, temperature=0.9)
    await agent.act(req)
    assert agent.client.call_count == 7
    assert all(c["temperature"] == 0.9 for c in agent.client.calls)


async def test_tie_breaks_to_first_occurrence(make_request):
    req = make_request()
    # 2 check, 2 bet -> tie; "check" appears first -> deterministic winner.
    agent = _agent(["check", "bet", "bet", "check"], n=4)
    resp = await agent.act(req)
    assert resp.action == "check"


async def test_all_unparseable_falls_back_to_repair_then_invalid(make_request):
    req = make_request()
    # 3 garbage samples (n=3) then 3 garbage repair-loop attempts (max_retries=2).
    agent = _agent(["g1", "g2", "g3", "r1", "r2", "r3"], n=3, max_retries=2)
    resp = await agent.act(req)
    assert resp.action == INVALID
    assert resp.metadata["harness"]["fallback"] is True
    assert resp.metadata["harness"]["parsed"] == 0


async def test_success_metadata_includes_observability_fields(make_request):
    req = make_request()
    # Script ModelOutputs so the representative winning sample carries model fields.
    win = ModelOutput(content="bet", finish_reason="stop", completion_tokens=7,
                      prompt_tokens=33, reasoning="some thinking")
    other = ModelOutput(content="check", finish_reason="stop")
    agent = SelfConsistencyAgent(client=FakeModelClient([win, win, other]),
                                 template=KuhnTemplate(), name="sc", n=3)
    resp = await agent.act(req)
    assert resp.action == "bet"
    m = resp.metadata
    # Same observability keys the final-loop-backed harnesses record.
    assert m["attempts"] == 1
    assert "latency_ms" in m
    assert m["finish_reason"] == "stop"
    assert m["truncated"] is False
    assert m["completion_tokens"] == 7
    assert m["prompt_tokens"] == 33
    assert m["has_reasoning"] is True
    # Harness-specific block still present.
    assert m["harness"]["winner"] == "bet"
