"""Multi-backend Hold'em eval orchestrator: CLI agents + Fireworks models.

Runs a fixed set of (mode, A, B, games) configs concurrently under ONE shared
global semaphore (bounds total in-flight CLI subprocesses / API calls). Each
config is an independent run_match with its own resume dir + heartbeat.

Backends per side:
  claude | codex | grok      -> CLIAgent (headless `claude -p` / `codex exec` / `grok -p`)
  fireworks:<model>          -> ModelAgent via Fireworks API (e.g. fireworks:kimi-k2p6)

Reuses the same HoldemTemplate prompt + parser for every backend, so a CLI is
just an Agent. Compares the *agents* (CLIs) and *models* (Fireworks), not raw
models for the CLI side.

Config knobs (env): GLOBAL_CONCURRENCY, LITE_GAMES, MATCH_GAMES, CLI_TIMEOUT,
ONLY (substring filter on the run label, e.g. ONLY=lite or ONLY=kimi).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict

GLOBAL_CONCURRENCY = int(os.environ.get("GLOBAL_CONCURRENCY", "16"))
LITE_GAMES = int(os.environ.get("LITE_GAMES", "50"))
MATCH_GAMES = int(os.environ.get("MATCH_GAMES", "20"))
TIMEOUT_S = int(os.environ.get("CLI_TIMEOUT", "300"))
ONLY = os.environ.get("ONLY", "")        # include-substring filter on run label
EXCLUDE = os.environ.get("EXCLUDE", "")  # exclude-substring filter on run label

KIMI = "fireworks:kimi-k2p6"

# (mode, A, B, games)
CONFIGS = [
    # --- CLI vs CLI ---
    ("lite",  "codex",  "claude", LITE_GAMES),
    ("lite",  "claude", "grok",   LITE_GAMES),
    ("lite",  "grok",   "codex",  LITE_GAMES),
    # --- CLI vs Fireworks kimi ---
    ("lite",  "claude", KIMI,     LITE_GAMES),
    ("lite",  "codex",  KIMI,     LITE_GAMES),
    ("lite",  "grok",   KIMI,     LITE_GAMES),
    # --- match mode: non-grok pairs only (serialized grok match ~= 30h, skipped) ---
    ("match", "codex",  "claude", MATCH_GAMES),
    ("match", "claude", KIMI,     MATCH_GAMES),
    ("match", "codex",  KIMI,     MATCH_GAMES),
]

if any(KIMI in (a, b) for _, a, b, _ in CONFIGS):
    os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())
else:
    os.environ.setdefault("FIREWORKS_API_KEY", "unused")

from aibattle.agents.base import Agent
from aibattle.agents.registry import make_agent
from aibattle.agents.templates.registry import make_template
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner
from aibattle.types import INVALID, AgentResponse
import _heartbeat

CLI_BACKENDS = {"claude", "codex", "grok"}

# The grok CLI is NOT concurrency-safe: parallel `grok -p` calls collide on a
# shared temp path and return empty output. Serialize ALL grok calls through one
# global lock so only one grok subprocess runs at a time (claude/codex/kimi stay
# concurrent). This makes grok pairs slow but correct.
_GROK_LOCK = asyncio.Lock()


def build_cmd(backend, prompt, model):
    if backend == "claude":
        c = ["claude", "-p", prompt]
        return c + (["--model", model] if model else [])
    if backend == "codex":
        c = ["codex", "exec", "-s", "read-only"] + (["-m", model] if model else [])
        return c + [prompt]
    if backend == "grok":
        c = ["grok", "-p", prompt]
        return c + (["-m", model] if model else [])
    raise ValueError(backend)


class CLIAgent(Agent):
    agent_type = "cli"

    def __init__(self, name, backend, template, *, model="", max_retries=2, timeout_s=300):
        self.name = name
        self.backend = backend
        self.model = model
        self.template = template
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    async def act(self, request):
        prompt = self.template.render_prompt(request)
        cur = prompt
        out = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            out = await self._run(cur)
            mv = self.template.parse(out or "", request)
            if mv is not None:
                return AgentResponse(
                    action=mv.type, amount=mv.amount, message=out, raw_output=out, prompt=prompt,
                    metadata={"attempts": attempt + 1, "backend": self.backend,
                              "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
            cur = self.template.repair_prompt(request, out or "")
        return AgentResponse(action=INVALID, message=out, raw_output=out, prompt=prompt,
                             metadata={"attempts": self.max_retries + 1, "backend": self.backend,
                                       "invalid": True})

    async def _run(self, prompt):
        if self.backend == "grok":           # grok is not concurrency-safe
            async with _GROK_LOCK:
                return await self._exec(prompt)
        return await self._exec(prompt)

    async def _exec(self, prompt):
        proc = await asyncio.create_subprocess_exec(
            *build_cmd(self.backend, prompt, self.model),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            so, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        return (so.decode(errors="replace") or "").strip()


def lbl(spec):
    return spec.replace("fireworks:", "fw-")


def build_agent(spec, game_name, template):
    if spec in CLI_BACKENDS:
        return CLIAgent(spec, spec, template, timeout_s=TIMEOUT_S)
    if spec.startswith("fireworks:"):
        name = spec.split(":", 1)[1]
        cfg = {"type": "model", "name": name,
               "model": {"provider": "fireworks",
                         "model_id": f"accounts/fireworks/models/{name}",
                         "api_key_env": "FIREWORKS_API_KEY",
                         "temperature": 0.6, "max_tokens": 131072, "timeout_s": 900},
               "max_retries": 2}
        return make_agent(cfg, game_name=game_name)
    raise ValueError(spec)


async def run_one(mode, A, B, games, global_sem, results):
    game_name = "holdem" if mode == "lite" else "holdem_match"
    template = make_template(game_name)
    a = build_agent(A, game_name, template)
    b = build_agent(B, game_name, template)
    la, lb = lbl(A), lbl(B)
    out = f"runs/cli_eval/{mode}/{la}__vs__{lb}"
    os.makedirs(out, exist_ok=True)
    hb_fh, hb_path = _heartbeat.open_log(f"{mode}_{la}_v_{lb}")
    if mode == "lite":
        factory = lambda: make_game("holdem", {"starting_stack": 200})
    else:
        factory = lambda: make_game("holdem_match", {"starting_stack": 200, "max_hands": 30})
    runner = Runner(factory, on_invalid_action="fallback")
    print(f"[start] {mode:5} {la} vs {lb}  ({games} games)  hb={hb_path}", flush=True)
    t0 = time.perf_counter()
    try:
        with MatchLogger(None) as lg:
            res = await runner.run_match(a, b, episodes=games, seat_swap=False, seed=None,
                                         logger=lg, semaphore=global_sem, episode_dir=out,
                                         on_step=_heartbeat.make_cb(hb_fh, f"{la}v{lb}"))
    except Exception as ex:  # noqa: BLE001
        print(f"[FAIL] {mode} {la} vs {lb}: {ex}", flush=True)
        results.append({"mode": mode, "a": la, "b": lb, "error": str(ex)})
        return

    chips = defaultdict(float)
    wins = defaultdict(int)
    ties = inv = 0
    for e in res.episodes:
        w = e.get("winner_name")
        if w:
            wins[w] += 1
        else:
            ties += 1
        inv += sum((e.get("invalid_count") or {}).values())
        if mode == "lite":
            for seat, nm in e["seat_assignment"].items():
                chips[nm] += e["returns"][seat]
    dt = time.perf_counter() - t0
    rec = {"mode": mode, "a": la, "b": lb, "games": len(res.episodes),
           "wins": dict(wins), "ties": ties, "invalid": inv, "secs": round(dt),
           "chips": {la: round(chips[a.name], 1), lb: round(chips[b.name], 1)} if mode == "lite" else None}
    results.append(rec)
    extra = f"chips={rec['chips']}" if mode == "lite" else ""
    print(f"[done ] {mode:5} {la} vs {lb}  {len(res.episodes)} games {dt:.0f}s | "
          f"wins={dict(wins)} ties={ties} invalid={inv} {extra}", flush=True)


async def main():
    def _lab(c):
        return f"{c[0]}_{lbl(c[1])}_v_{lbl(c[2])}"
    runs = [c for c in CONFIGS
            if (not ONLY or ONLY in _lab(c)) and (not EXCLUDE or EXCLUDE not in _lab(c))]
    print(f"CLI eval: {len(runs)} runs, global concurrency {GLOBAL_CONCURRENCY}, "
          f"lite={LITE_GAMES} match={MATCH_GAMES}\n", flush=True)
    global_sem = asyncio.Semaphore(GLOBAL_CONCURRENCY)
    results = []
    t0 = time.perf_counter()
    await asyncio.gather(*(run_one(*c, global_sem, results) for c in runs))
    print(f"\n========== ALL DONE in {time.perf_counter()-t0:.0f}s ==========")
    for r in sorted(results, key=lambda x: (x["mode"], x["a"])):
        if "error" in r:
            print(f"  {r['mode']:5} {r['a']} vs {r['b']}: ERROR {r['error']}")
        else:
            ex = f" chips={r['chips']}" if r["chips"] else ""
            print(f"  {r['mode']:5} {r['a']:7} vs {r['b']:10} | wins={r['wins']} "
                  f"ties={r['ties']} invalid={r['invalid']}{ex}")


if __name__ == "__main__":
    asyncio.run(main())
