"""In-process reasoning harnesses (inference-time scaffolding) for local agents.

A harness wraps the same ModelClient + GameTemplate that ModelAgent uses, but
orchestrates multiple generations per decision (structured CoT, self-consistency
voting, two-stage estimate->decide, self-refine). They are game-agnostic and
stateless. Selected by ``harness: <name>`` in a ``type: local`` config block.
"""

from __future__ import annotations

from .base import HarnessAgent
from .cot import StructuredCoTAgent
from .self_consistency import SelfConsistencyAgent
from .two_stage import TwoStageAgent
from .self_refine import SelfRefineAgent
from .holdem_estimate_act import HoldemEstimateActAgent

_HARNESSES = {
    "cot": StructuredCoTAgent,
    "self_consistency": SelfConsistencyAgent,
    "two_stage": TwoStageAgent,
    "self_refine": SelfRefineAgent,
    "holdem_estimate_act": HoldemEstimateActAgent,
}


def available_harnesses() -> list:
    return sorted(_HARNESSES)


def make_harness(harness: str, *, client, template, name: str,
                 max_retries: int = 2, harness_args: dict | None = None) -> HarnessAgent:
    """Construct a harness agent by name, passing through ``harness_args``."""
    if harness not in _HARNESSES:
        raise ValueError(
            f"Unknown harness {harness!r}. Available: {available_harnesses()}"
        )
    cls = _HARNESSES[harness]
    return cls(client=client, template=template, name=name,
               max_retries=max_retries, **(harness_args or {}))


__all__ = [
    "HarnessAgent", "StructuredCoTAgent", "SelfConsistencyAgent",
    "TwoStageAgent", "SelfRefineAgent", "HoldemEstimateActAgent",
    "make_harness", "available_harnesses",
]
