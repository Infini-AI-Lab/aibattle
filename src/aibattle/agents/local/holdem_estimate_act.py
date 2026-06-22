"""Hold'em estimate -> act harness (range estimation, then decision).

A poker-specific two-stage harness:

  Stage 1 (estimate): from Hero's hand, the board, and the betting line, the
    model produces a compact estimate of the OPPONENT's range *relative to
    Hero's current hand* (four buckets that sum to 1), a coarse hero-vs-range
    label, and a one-line justification. It does NOT choose an action.
  Stage 2 (act): the estimate is threaded into the game's own decision prompt
    and the model chooses a legal action through the shared parse/repair loop.

Output handling is SOFT (mode A): Stage 1 is asked for JSON, but a parse failure
never aborts the hand — the raw estimate text is passed through to Stage 2
instead, and whatever was produced (parsed dict or raw text) is recorded under
metadata["harness"] for auditing. The four buckets, if parsed, are normalized to
sum to 1 before being rendered into the decision prompt.

This harness is holdem-specific (the estimate vocabulary assumes Texas Hold'em)
and is selected by ``harness: holdem_estimate_act`` on a ``type: local`` agent.

Ref (motivation): assessing a hidden-information opponent's range before acting —
How Far Are LLMs from Professional Poker Players? (arXiv:2602.00528).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ...types import AgentRequest, AgentResponse
from .base import HarnessAgent

_ESTIMATE_INSTRUCTION = (
    "You are a poker range-estimation module. Do NOT choose an action and do NOT "
    "suggest fold/check/call/bet/raise or any sizing.\n"
    "From Hero's hand, the public board, position, and the betting line above, "
    "estimate the OPPONENT's current range CLASSIFIED RELATIVE TO HERO'S HAND "
    "(not by absolute strength): e.g. if Hero has a set, the opponent's one-pair "
    "hands are worse_made_hands, not a strong threat; if Hero has only top pair, "
    "opponent sets/two-pair/overpairs/better top pairs are ahead_or_strong_threat.\n"
    "Output ONLY a JSON object (no prose outside it) of the form:\n"
    "{\n"
    '  "villain_range_vs_hero": {\n'
    '    "ahead_or_strong_threat": 0.0,  // ahead of Hero or a serious value threat\n'
    '    "worse_made_hands": 0.0,        // showdown value but usually behind Hero\n'
    '    "draws": 0.0,                   // meaningful equity to improve (flush/straight/combo)\n'
    '    "air_or_bluffs": 0.0            // little showdown value, fold-equity/bluffs\n'
    "  },\n"
    '  "hero_vs_range": "far_behind|behind|close|slightly_ahead|ahead|far_ahead",\n'
    '  "justification": "one or two sentences: board texture, betting line, '
    'position, Hero\'s relative strength"\n'
    "}\n"
    "The four probabilities must sum to 1.0. Use rough probabilities, not solver "
    "frequencies. Keep the justification short; do not reveal a long step-by-step "
    "chain."
)

_BUCKETS = ("ahead_or_strong_threat", "worse_made_hands", "draws", "air_or_bluffs")


class HoldemEstimateActAgent(HarnessAgent):
    def __init__(self, *, client, template, name="holdem_estimate_act", max_retries=2,
                 estimate_prompt: str = _ESTIMATE_INSTRUCTION):
        super().__init__(client=client, template=template, name=name,
                         max_retries=max_retries)
        self.estimate_prompt = estimate_prompt

    # --- estimation context: rules + coaching + state, WITHOUT the action ask ---
    def _context(self, request: AgentRequest) -> str:
        t = self.template
        secs = [t.rules(request), t.coaching(request), t.state(request)]
        return "\n\n".join(s for s in secs if s)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Best-effort: pull the outermost {...} and json.loads it. None on fail."""
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)  # first '{' to last '}'
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def _summarize(cls, parsed: Optional[dict], raw: str) -> str:
        """Render a clean estimate block for the decision prompt.

        On a successful parse, normalize the four buckets to sum 1 and format a
        canonical summary. On failure, pass the raw estimate text through.
        """
        if not parsed:
            return f"Opponent/context assessment:\n{(raw or '').strip()}"
        rng = parsed.get("villain_range_vs_hero") or {}
        vals = {}
        for k in _BUCKETS:
            try:
                vals[k] = max(0.0, float(rng.get(k, 0.0)))
            except (ValueError, TypeError):
                vals[k] = 0.0
        total = sum(vals.values())
        if total > 0:
            vals = {k: v / total for k, v in vals.items()}
        else:  # nothing usable parsed -> fall back to raw
            return f"Opponent/context assessment:\n{(raw or '').strip()}"
        hvr = str(parsed.get("hero_vs_range", "")).strip() or "unknown"
        just = str(parsed.get("justification", "")).strip()
        lines = [
            "Opponent range estimate (relative to your current hand):",
            f"  ahead / strong threat: {vals['ahead_or_strong_threat']:.2f}",
            f"  worse made hands:      {vals['worse_made_hands']:.2f}",
            f"  draws:                 {vals['draws']:.2f}",
            f"  air / bluffs:          {vals['air_or_bluffs']:.2f}",
            f"Hero vs range: {hvr}",
        ]
        if just:
            lines.append(f"Why: {just}")
        return "\n".join(lines)

    async def act(self, request: AgentRequest) -> AgentResponse:
        # Stage 1: estimate the opponent's range (no action). Context omits the
        # action instruction so Stage 1 isn't asked to both estimate and decide.
        est_prompt = f"{self._context(request)}\n\n{self.estimate_prompt}"
        est = await self._generate(est_prompt)
        raw = est.content or est.full_text or ""
        parsed = self._extract_json(raw)
        summary = self._summarize(parsed, raw)

        # Stage 2: decide, with the estimate threaded into the decision prompt.
        decide_prompt = self._compose(request, extra_context=summary)
        return await self._final_loop(
            request, prompt=decide_prompt,
            harness_meta={
                "kind": "holdem_estimate_act",
                "estimate_parsed": parsed,        # dict or None
                "estimate_raw": raw,
                "estimate_summary": summary,
            },
        )
