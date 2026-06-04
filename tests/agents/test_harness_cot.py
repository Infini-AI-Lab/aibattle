"""Structured CoT harness tests (AC3)."""

from __future__ import annotations

from aibattle.agents.local.cot import StructuredCoTAgent
from aibattle.agents.templates.kuhn import KuhnTemplate
from aibattle.types import INVALID
from tests.conftest import FakeModelClient


def _agent(scripted, **kw):
    return StructuredCoTAgent(client=FakeModelClient(scripted),
                              template=KuhnTemplate(), name="cot", **kw)


async def test_prompt_includes_structured_instruction(make_request):
    req = make_request()
    agent = _agent(["check"])
    resp = await agent.act(req)
    assert resp.action == "check"
    # The default CoT instruction must be threaded into the prompt the model saw.
    sent = agent.client.calls[0]["prompt"]
    assert "reason step by step" in sent
    assert resp.metadata["harness"]["kind"] == "structured_cot"


async def test_cot_instructions_override(make_request):
    req = make_request()
    agent = _agent(["bet"], cot_instructions="THINK_ABOUT_XYZZY_FIRST")
    await agent.act(req)
    assert "THINK_ABOUT_XYZZY_FIRST" in agent.client.calls[0]["prompt"]


async def test_garbage_then_repair_then_invalid(make_request):
    req = make_request()
    agent = _agent(["nope", "still bad", "garbage"], max_retries=2)
    resp = await agent.act(req)
    assert resp.action == INVALID
    assert resp.metadata["attempts"] == 3
    assert agent.client.call_count == 3
