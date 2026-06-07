"""Shared template execution loop: render -> generate -> parse -> repair.

Both ``ModelAgent`` and the text-output reasoning harnesses (``local/``) share
the identical control flow: render a prompt from the request, generate text,
parse it into a legal ``Move``, and on failure retry with a repair nudge. The
only thing that varies is the ``generate`` callable (a model SDK vs a multi-step
harness step) and the metadata it carries.

This module factors that loop out so it is written exactly once. ``ModelAgent``
delegates to it; every harness builds on the same primitive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ..types import INVALID, AgentRequest, AgentResponse, Move
from .templates.base import GameTemplate


@dataclass
class GenerateResult:
    """Normalized output of one generation attempt.

    ``content`` is what gets parsed into an action; ``full_text`` is logged (it
    may include the model's thinking). ``meta`` carries provider/harness-specific
    fields that are merged verbatim into the final ``AgentResponse.metadata``
    (e.g. token counts, finish_reason).
    """

    content: str
    full_text: Optional[str] = None
    meta: Optional[dict] = None

    def text_for_parse(self) -> str:
        """The best available text to feed a repair nudge when parsing fails."""
        return self.content or self.full_text or ""


# A generate function takes the current prompt and returns a GenerateResult.
GenerateFn = Callable[[str], Awaitable[GenerateResult]]


def parse_or_none(template: GameTemplate, text: str, request: AgentRequest) -> Optional[Move]:
    """Thin wrapper over ``template.parse`` so harness steps share one entry point."""
    if not text:
        return None
    return template.parse(text, request)


async def run_template_loop(
    template: GameTemplate,
    generate: GenerateFn,
    request: AgentRequest,
    *,
    max_retries: int = 2,
    initial_prompt: Optional[str] = None,
) -> AgentResponse:
    """Render -> generate -> parse -> repair, returning an ``AgentResponse``.

    Mirrors the exact behavior of the original ``ModelAgent.act``: parse the
    final answer; on a parse failure, re-prompt with ``template.repair_prompt``
    using the visible answer (or full text if there is no answer at all); after
    exhausting retries, return ``INVALID`` and let the runner's invalid-action
    policy decide what happens. ``metadata`` always carries ``attempts`` and
    ``latency_ms``, plus whatever ``GenerateResult.meta`` the generator supplied.

    ``initial_prompt`` lets a harness seed the FIRST attempt with a custom prompt
    (e.g. one carrying an opponent-range estimate or a critique). Retries still
    use the template's game-defined ``repair_prompt``, so repair behavior is
    identical to ModelAgent's; only the opening prompt differs. It is also the
    prompt logged on the response for replay.
    """
    prompt = template.render_prompt(request) if initial_prompt is None else initial_prompt
    input_prompt = prompt  # exact decision context; logged for replay/analysis
    last: Optional[GenerateResult] = None
    t0 = time.perf_counter()

    for attempt in range(max_retries + 1):
        last = await generate(prompt)
        move = parse_or_none(template, last.content, request)
        if move is not None:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            meta = {"attempts": attempt + 1, "latency_ms": latency_ms}
            if last.meta:
                meta.update(last.meta)
            return AgentResponse(
                action=move.type,
                amount=move.amount,
                message=last.content,
                raw_output=last.full_text if last.full_text is not None else last.content,
                prompt=input_prompt,
                metadata=meta,
            )
        # Repair using the visible answer; pass full text if no answer at all.
        prompt = template.repair_prompt(request, last.text_for_parse())

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    meta = {"attempts": max_retries + 1, "latency_ms": latency_ms, "invalid": True}
    if last and last.meta:
        meta.update(last.meta)
    return AgentResponse(
        action=INVALID,
        message=last.content if last else None,
        raw_output=(last.full_text if last and last.full_text is not None else
                    (last.content if last else None)),
        prompt=input_prompt,
        metadata=meta,
    )
