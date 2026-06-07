"""Self-Consistency (majority-vote) harness.

Sample the same decision prompt ``n`` times at ``temperature`` > 0, parse each
sample into a Move, and take the majority vote. This trades inference compute
for stability — it cancels the noise of a single stochastic decision. If no
sample parses, it falls back to the standard parse/repair loop so the agent
still returns a diagnosed INVALID rather than crashing.

Ref: Wang et al. 2022, "Self-Consistency Improves CoT Reasoning in LLMs"
(arXiv:2203.11171).
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter

from ...types import AgentRequest, AgentResponse
from .base import HarnessAgent


class SelfConsistencyAgent(HarnessAgent):
    def __init__(self, *, client, template, name="self_consistency", max_retries=2,
                 n: int = 5, temperature: float = 0.7):
        super().__init__(client=client, template=template, name=name,
                         max_retries=max_retries)
        self.n = max(1, int(n))
        self.temperature = float(temperature)

    async def act(self, request: AgentRequest) -> AgentResponse:
        prompt = self._final_prompt(request)
        # Sample n times concurrently at the configured temperature.
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(self._generate(prompt, temperature=self.temperature) for _ in range(self.n))
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Parse each sample once; keep (result, move) so the representative sample
        # for the winning move is found without re-parsing.
        parsed = [(r, self._parse(r.content, request)) for r in results]
        moves = [m for _, m in parsed if m is not None]

        if not moves:
            # No sample parsed -> fall back to the standard repair loop.
            resp = await self._final_loop(request)
            resp.metadata.setdefault("harness", {})
            resp.metadata["harness"].update(
                {"kind": "self_consistency", "n": self.n,
                 "temperature": self.temperature, "parsed": 0, "fallback": True})
            return resp

        winner = self._vote(moves)
        # Vote distribution keyed by the move label, for log/replay inspection.
        dist = Counter(m.label() for m in moves)
        # A representative sample of the winning move supplies message/raw_output
        # and the model observability fields (tokens, finish_reason, ...).
        rep = next((r for r, m in parsed if m == winner), results[0])
        # Carry the same model-output metadata keys the final-loop path records,
        # so self-consistency decisions are as observable as every other harness.
        meta = {"attempts": 1, "latency_ms": latency_ms}
        if rep.meta:
            meta.update(rep.meta)
        meta["harness"] = {
            "kind": "self_consistency",
            "n": self.n,
            "temperature": self.temperature,
            "parsed": len(moves),
            "votes": dict(dist),
            "winner": winner.label(),
        }
        return AgentResponse(
            action=winner.type,
            amount=winner.amount,
            message=rep.content,
            raw_output=rep.full_text if rep.full_text is not None else rep.content,
            prompt=prompt,
            metadata=meta,
        )
