"""Tests for the shared template loop (AC1) — the keystone primitive.

Exercised fully offline with FakeModelClient + the real Kuhn/Hold'em templates.
Also pins the regression contract that ModelAgent delegates to this loop with
identical observable behavior and metadata keys.
"""

from __future__ import annotations

import pytest

from aibattle.agents.model_agent import ModelAgent
from aibattle.agents.template_loop import GenerateResult, run_template_loop
from aibattle.agents.templates.holdem import HoldemTemplate
from aibattle.agents.templates.kuhn import KuhnTemplate
from aibattle.models.base import ModelOutput
from aibattle.types import INVALID
from tests.conftest import FakeModelClient


def _gen_from(client):
    """Adapt a FakeModelClient into a GenerateFn for run_template_loop."""
    async def generate(prompt: str) -> GenerateResult:
        out = await client.generate(prompt)
        return GenerateResult(content=out.content, full_text=out.full_text(),
                              meta={"has_reasoning": out.reasoning is not None,
                                    "finish_reason": out.finish_reason,
                                    "truncated": out.truncated,
                                    "completion_tokens": out.completion_tokens,
                                    "prompt_tokens": out.prompt_tokens})
    return generate


async def test_success_first_try_returns_move_and_attempts_1(make_request):
    req = make_request()
    tmpl = KuhnTemplate()
    client = FakeModelClient(["bet"])
    resp = await run_template_loop(tmpl, _gen_from(client), req, max_retries=2)
    assert resp.action == "bet"
    assert resp.amount is None
    assert resp.metadata["attempts"] == 1
    assert client.call_count == 1
    assert client.calls[0]["prompt"] == tmpl.render_prompt(req)


async def test_success_after_one_repair_uses_repair_prompt(make_request):
    req = make_request()
    tmpl = KuhnTemplate()
    client = FakeModelClient(["xyzzy", "check"])
    resp = await run_template_loop(tmpl, _gen_from(client), req, max_retries=2)
    assert resp.action == "check"
    assert resp.metadata["attempts"] == 2
    assert client.call_count == 2
    # The second call must have seen the repair prompt, not the original.
    assert client.calls[1]["prompt"] == tmpl.repair_prompt(req, "xyzzy")


async def test_exhausted_retries_returns_invalid(make_request):
    req = make_request()
    tmpl = KuhnTemplate()
    client = FakeModelClient(["nope", "still nope", "garbage"])
    resp = await run_template_loop(tmpl, _gen_from(client), req, max_retries=2)
    assert resp.action == INVALID
    assert resp.metadata["attempts"] == 3
    assert resp.metadata["invalid"] is True
    assert client.call_count == 3  # no extra calls beyond max_retries+1


async def test_numeric_game_amount_parsed(make_request):
    req = make_request(numeric=True, game="holdem")
    tmpl = HoldemTemplate()
    client = FakeModelClient(["raise 12"])
    resp = await run_template_loop(tmpl, _gen_from(client), req, max_retries=2)
    assert resp.action == "raise"
    assert resp.amount == 12


async def test_numeric_missing_amount_triggers_repair(make_request):
    req = make_request(numeric=True, game="holdem")
    tmpl = HoldemTemplate()
    # "raise" with no number -> HoldemTemplate.parse returns None -> repair.
    client = FakeModelClient(["raise", "raise 8"])
    resp = await run_template_loop(tmpl, _gen_from(client), req, max_retries=2)
    assert resp.action == "raise"
    assert resp.amount == 8
    assert resp.metadata["attempts"] == 2


async def test_metadata_carries_token_and_finish_fields(make_request):
    req = make_request()
    tmpl = KuhnTemplate()
    out = ModelOutput(content="bet", reasoning="long thinking",
                      finish_reason="length", completion_tokens=5, prompt_tokens=40)
    client = FakeModelClient([out])
    resp = await run_template_loop(tmpl, _gen_from(client), req, max_retries=2)
    assert resp.metadata["truncated"] is True
    assert resp.metadata["has_reasoning"] is True
    assert resp.metadata["finish_reason"] == "length"
    assert resp.metadata["completion_tokens"] == 5
    assert resp.metadata["prompt_tokens"] == 40
    # raw_output includes the thinking block (full_text()).
    assert resp.raw_output == out.full_text()
    assert "thinking" in resp.raw_output


async def test_initial_prompt_override_seeds_first_attempt(make_request):
    req = make_request()
    tmpl = KuhnTemplate()
    client = FakeModelClient(["bet"])
    custom = "CUSTOM SEED PROMPT\n" + tmpl.render_prompt(req)
    resp = await run_template_loop(tmpl, _gen_from(client), req,
                                   max_retries=2, initial_prompt=custom)
    assert resp.action == "bet"
    assert client.calls[0]["prompt"] == custom
    assert resp.prompt == custom


async def test_model_agent_delegates_to_shared_loop(make_request):
    """ModelAgent must produce the same result/metadata as the helper (AC1)."""
    req = make_request()
    tmpl = KuhnTemplate()
    out = ModelOutput(content="bet", finish_reason="stop",
                      completion_tokens=3, prompt_tokens=20)

    agent = ModelAgent(FakeModelClient([out]), tmpl, name="m", max_retries=2)
    direct = await run_template_loop(tmpl, _gen_from(FakeModelClient([out])), req,
                                     max_retries=2)
    via_agent = await agent.act(req)

    assert via_agent.action == direct.action == "bet"
    # Same metadata keys, same values for the model-derived fields.
    for k in ("attempts", "has_reasoning", "finish_reason", "truncated",
              "completion_tokens", "prompt_tokens"):
        assert via_agent.metadata[k] == direct.metadata[k]
    assert via_agent.raw_output == direct.raw_output
