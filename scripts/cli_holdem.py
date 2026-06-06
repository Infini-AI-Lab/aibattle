"""CLI agents play Hold'em (Lite). Backends: claude / codex / grok.

Each decision shells out to a headless CLI invocation, reusing the EXACT same
HoldemTemplate prompt + parser the model agents use. A CLI is therefore an
ordinary Agent; the whole engine (runner, resume, fallback, heartbeat) works
unchanged.

Compares the CLI *agents* (model + harness + system prompt), not raw models.
Each act() is a fresh, stateless headless call (no session memory).

Usage (claude vs codex, 20 hands):
    HANDS=20 A_BACKEND=claude B_BACKEND=codex python scripts/cli_holdem.py
"""

from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("FIREWORKS_API_KEY", "unused")  # CLIs don't use it

from aibattle.agents.base import Agent
from aibattle.agents.templates.registry import make_template
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner
from aibattle.types import INVALID, AgentResponse

HANDS = int(os.environ.get("HANDS", "20"))
TIMEOUT_S = int(os.environ.get("CLI_TIMEOUT", "300"))
A_BACKEND = os.environ.get("A_BACKEND", "claude")
B_BACKEND = os.environ.get("B_BACKEND", "codex")
A_MODEL = os.environ.get("A_MODEL", "claude-sonnet-4-6")
B_MODEL = os.environ.get("B_MODEL", "")  # codex default (gpt-5.5)
OUT = os.environ.get("OUT", "runs/cli_holdem")


def build_cmd(backend: str, prompt: str, model: str) -> list:
    """Headless command per CLI; all print the answer with the action last."""
    if backend == "claude":
        cmd = ["claude", "-p", prompt]
        if model:
            cmd += ["--model", model]
        return cmd
    if backend == "codex":
        cmd = ["codex", "exec", "-s", "read-only"]
        if model:
            cmd += ["-m", model]
        return cmd + [prompt]
    if backend == "grok":
        cmd = ["grok", "-p", prompt]
        if model:
            cmd += ["-m", model]
        return cmd
    raise ValueError(f"unknown backend {backend}")


class CLIAgent(Agent):
    agent_type = "cli"

    def __init__(self, name, backend, model, template, *, max_retries=2, timeout_s=300):
        self.name = name
        self.backend = backend
        self.model = model
        self.template = template
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    async def act(self, request) -> AgentResponse:
        prompt = self.template.render_prompt(request)
        cur = prompt
        out = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            out = await self._run_cli(cur)
            move = self.template.parse(out or "", request)
            if move is not None:
                return AgentResponse(
                    action=move.type, amount=move.amount,
                    message=out, raw_output=out, prompt=prompt,
                    metadata={"attempts": attempt + 1, "backend": self.backend,
                              "model": self.model,
                              "latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
                )
            cur = self.template.repair_prompt(request, out or "")
        return AgentResponse(
            action=INVALID, message=out, raw_output=out, prompt=prompt,
            metadata={"attempts": self.max_retries + 1, "backend": self.backend,
                      "model": self.model, "invalid": True},
        )

    async def _run_cli(self, prompt: str):
        cmd = build_cmd(self.backend, prompt, self.model)
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        return (stdout.decode(errors="replace") or "").strip()


def _on_step(ev):
    raw = (ev.get("raw_output") or "").replace("\n", " ")
    snip = raw[-70:] if len(raw) > 70 else raw
    pub = ev.get("public") or {}
    amt = ev.get("amount")
    print(f"{time.strftime('%H:%M:%S')} street={str(pub.get('street')):<7} "
          f"{ev['agent_name']:<8} -> {ev['action']}{(' '+str(amt)) if amt is not None else ''}"
          f"   «{snip}»", flush=True)


async def main():
    tmpl = make_template("holdem")
    a = CLIAgent(f"{A_BACKEND}", A_BACKEND, A_MODEL, tmpl, timeout_s=TIMEOUT_S)
    b = CLIAgent(f"{B_BACKEND}", B_BACKEND, B_MODEL, tmpl, timeout_s=TIMEOUT_S)
    gdir = os.path.join(OUT, f"{A_BACKEND}__vs__{B_BACKEND}")
    os.makedirs(gdir, exist_ok=True)
    runner = Runner(lambda: make_game("holdem", {"starting_stack": 200}),
                    on_invalid_action="fallback")
    print(f"CLI Hold'em: A={A_BACKEND}({A_MODEL or 'default'})  vs  "
          f"B={B_BACKEND}({B_MODEL or 'default'})  | {HANDS} hands  | resume on\n", flush=True)
    t0 = time.perf_counter()
    with MatchLogger(None) as lg:
        res = await runner.run_match(a, b, episodes=HANDS, seat_swap=False, seed=None,
                                     logger=lg, episode_dir=gdir, on_step=_on_step)

    tally = {A_BACKEND: 0.0, B_BACKEND: 0.0}
    wins = {A_BACKEND: 0, B_BACKEND: 0}; ties = 0; inv = 0
    print(f"\n=== per-hand results ({len(res.episodes)} hands) ===", flush=True)
    for e in sorted(res.episodes, key=lambda x: x["episode"]):
        per = {nm: e["returns"][seat] for seat, nm in e["seat_assignment"].items()}
        for nm, v in per.items():
            tally[nm] += v
        w = e.get("winner_name")
        if w:
            wins[w] += 1
        else:
            ties += 1
        inv += sum((e.get("invalid_count") or {}).values())
        print(f"  hand {e['episode']:>2}: winner={str(w):<8} {per}", flush=True)

    print(f"\n=== FINAL ({HANDS} hands, {time.perf_counter()-t0:.0f}s) ===")
    print(f"  hands won:  {A_BACKEND}={wins[A_BACKEND]}  {B_BACKEND}={wins[B_BACKEND]}  ties={ties}")
    print(f"  chip total: {A_BACKEND}={tally[A_BACKEND]:+.0f}  {B_BACKEND}={tally[B_BACKEND]:+.0f}")
    print(f"  invalid (fallback) actions: {inv}")


if __name__ == "__main__":
    asyncio.run(main())
