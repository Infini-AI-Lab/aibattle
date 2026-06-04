"""Registry dispatch + loader validation tests (AC4).

These stay offline by monkeypatching ``make_client`` so no real model SDK
(openai/anthropic) is needed — the tests pin dispatch and game_name threading,
not the provider clients.
"""

from __future__ import annotations

import pytest

import aibattle.models.registry as model_registry
from aibattle.agents.local.cot import StructuredCoTAgent
from aibattle.agents.local.two_stage import TwoStageAgent
from aibattle.agents.registry import make_agent
from aibattle.config.loader import load_config
from aibattle.agents.templates.holdem import HoldemTemplate
from aibattle.agents.templates.kuhn import KuhnTemplate
from tests.conftest import FakeModelClient


@pytest.fixture(autouse=True)
def stub_make_client(monkeypatch):
    """Replace make_client everywhere the registry imports it."""
    def _fake(cfg):
        return FakeModelClient(["check"])
    monkeypatch.setattr(model_registry, "make_client", _fake)
    yield


def _local_cfg(harness, **extra):
    cfg = {"type": "local", "harness": harness, "name": f"{harness}-bot",
           "model": {"provider": "openai", "model_id": "x", "api_key_env": "X"}}
    cfg.update(extra)
    return cfg


def test_make_agent_local_builds_two_stage():
    agent = make_agent(_local_cfg("two_stage"), game_name="kuhn_poker")
    assert isinstance(agent, TwoStageAgent)
    assert agent.agent_type == "local"
    assert agent.name == "two_stage-bot"


def test_make_agent_local_threads_game_name_to_template():
    a_kuhn = make_agent(_local_cfg("cot"), game_name="kuhn_poker")
    a_holdem = make_agent(_local_cfg("cot"), game_name="holdem")
    assert isinstance(a_kuhn, StructuredCoTAgent)
    assert isinstance(a_kuhn.template, KuhnTemplate)
    assert isinstance(a_holdem.template, HoldemTemplate)


def test_harness_args_passed_through():
    agent = make_agent(
        _local_cfg("two_stage", harness_args={"estimate_prompt": "MY_ESTIMATE"}),
        game_name="kuhn_poker")
    assert agent.estimate_prompt == "MY_ESTIMATE"


def test_unknown_harness_raises():
    with pytest.raises(ValueError, match="Unknown harness"):
        make_agent(_local_cfg("bogus"), game_name="kuhn_poker")


def test_missing_harness_raises():
    cfg = {"type": "local", "name": "x",
           "model": {"provider": "openai", "model_id": "x", "api_key_env": "X"}}
    with pytest.raises(ValueError, match="needs a 'harness'"):
        make_agent(cfg, game_name="kuhn_poker")


def test_unknown_agent_type_raises():
    with pytest.raises(ValueError, match="Unknown agent type"):
        make_agent({"type": "frobnicate"}, game_name="kuhn_poker")


# --- loader validation -------------------------------------------------------

_BASE_YAML = """
game:
  name: kuhn_poker
players:
  player_0:
    agent:
      type: {t0}
      harness: cot
      model: {{ provider: openai, model_id: x, api_key_env: X }}
  player_1:
    agent:
      type: builtin
      name: random
run:
  episodes: 2
"""


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return str(p)


def test_loader_accepts_local_type(tmp_path):
    cfg = load_config(_write(tmp_path, _BASE_YAML.format(t0="local")))
    assert cfg.players["player_0"]["type"] == "local"
    assert cfg.players["player_0"]["harness"] == "cot"


def test_loader_rejects_unknown_type(tmp_path):
    with pytest.raises(ValueError, match="must be"):
        load_config(_write(tmp_path, _BASE_YAML.format(t0="frobnicate")))
