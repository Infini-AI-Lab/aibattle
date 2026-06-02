"""Profile Fireworks rate/concurrency limits across keys to find what's SHARED.

Account topology: key1=.fireworks (acct A); key2=.fireworks-2, key3=.fireworks-3
(both acct B); all one org.

v2 — saturating ramp. The v1 fixed-burst test only measured serverless warmup
noise and never hit a limit. This version:
  1. WARMS UP each key (discarded burst) so cold-start doesn't skew results.
  2. RAMPS concurrency (64 -> 1024) and watches for the 429 onset + the point
     where throughput stops scaling = the real ceiling.
  3. Compares ceilings:
       k2 alone        -> single-key (acct B) ceiling
       k2+k3 (acct B)  -> if 429s start at ~same TOTAL load as k2 alone -> PER-ACCOUNT;
                          if each key reaches its own ceiling (~2x) -> PER-KEY
       k1+k2 (A+B)     -> if ~2x of solos -> accounts independent; if capped -> PER-ORG

Fast model + tiny output so we stress admission/concurrency, not decode.
Run when keys are idle. Usage: python scripts/profile_keys.py
"""

from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/gpt-oss-120b"
PROMPT = "Reply with exactly one word: ok."
MAX_TOKENS = 64
START_CONC = 128                      # ramp start (per key)
MAX_CONC = 8192                       # safety cap (per key)
STOP_429_PCT = 10.0                   # stop ramping once 429 rate exceeds this
REQS_FACTOR = 3                       # requests per level = factor * concurrency
WARMUP = 64
PAUSE_S = 6

KEY_FILES = {"k1": ".fireworks", "k2": ".fireworks-2", "k3": ".fireworks-3"}
ACCT = {"k1": "A", "k2": "B", "k3": "B"}


def load(path: str) -> str:
    return open(path).read().strip()


async def call(client, sem) -> dict:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": PROMPT}],
                temperature=0.0, max_tokens=MAX_TOKENS)
            ct = getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0
            return {"ok": True, "tok": ct, "lat": time.perf_counter() - t0, "k429": False}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "tok": 0, "lat": time.perf_counter() - t0,
                    "k429": ("429" in str(e) or "rate" in str(e).lower())}


async def burst(client, conc: int, n: int) -> dict:
    sem = asyncio.Semaphore(conc)
    t0 = time.perf_counter()
    res = await asyncio.gather(*(call(client, sem) for _ in range(n)))
    wall = time.perf_counter() - t0
    ok = sum(r["ok"] for r in res)
    r429 = sum(r["k429"] for r in res)
    toks = sum(r["tok"] for r in res)
    return {"conc": conc, "n": n, "ok": ok, "r429": r429,
            "tok_s": toks / wall if wall else 0, "req_s": ok / wall if wall else 0}


async def ramp(label: str, clients: dict) -> list:
    """Double concurrency until 429s exceed STOP_429_PCT or MAX_CONC. Logs PER-KEY
    429 breakdown (to tell org-cap [both keys reject] from confound [one key])."""
    print(f"\n>>> {label}: keys={list(clients)}", flush=True)
    rows = []
    conc = START_CONC
    names = list(clients)
    while conc <= MAX_CONC:
        n = REQS_FACTOR * conc
        per = await asyncio.gather(*(burst(c, conc, n) for c in clients.values()))
        agg_tok = sum(p["tok_s"] for p in per)
        agg_req = sum(p["req_s"] for p in per)
        tot429 = sum(p["r429"] for p in per)
        tot_n = sum(p["n"] for p in per)
        pct = 100 * tot429 / tot_n if tot_n else 0
        bd = " ".join(f"{nm}({ACCT[nm]}):{p['r429']}/{p['n']}" for nm, p in zip(names, per))
        print(f"    conc={conc:>5}/key n={tot_n:>6} | {agg_tok:8.0f} tok/s {agg_req:7.1f} req/s "
              f"| 429s={tot429}/{tot_n} ({pct:.0f}%) | per-key: {bd}", flush=True)
        rows.append({"conc": conc, "agg_tok": agg_tok, "agg_req": agg_req,
                     "r429": tot429, "n": tot_n, "perkey": {nm: p["r429"] for nm, p in zip(names, per)}})
        if pct >= STOP_429_PCT:
            print(f"    -> 429 ceiling hit at conc {conc}/key ({pct:.0f}% rejected)", flush=True)
            break
        conc *= 2
        await asyncio.sleep(PAUSE_S)
    return rows


def ceiling(rows: list) -> str:
    peak = max(rows, key=lambda r: r["agg_req"])
    first429 = next((r["conc"] for r in rows if r["r429"] > 0), None)
    s = f"peak {peak['agg_req']:.0f} req/s @ conc {peak['conc']}/key"
    s += f"; first 429 @ conc {first429}/key" if first429 else "; NO 429 up to max"
    return s


async def main():
    keys = {n: load(p) for n, p in KEY_FILES.items()}
    cl = {n: AsyncOpenAI(api_key=k, base_url=BASE_URL, timeout=120) for n, k in keys.items()}
    print(f"Profiler v2 (saturating ramp) | model={MODEL.split('/')[-1]} max_tokens={MAX_TOKENS}")
    print(f"ramp {START_CONC}->{MAX_CONC}/key (x2), reqs={REQS_FACTOR}x conc, stop at {STOP_429_PCT}% 429 | k1=A k2=B k3=B")

    print("\n--- warmup (discarded) ---", flush=True)
    await asyncio.gather(*(burst(c, WARMUP, WARMUP) for c in cl.values()))
    await asyncio.sleep(PAUSE_S)

    # CROSS-ACCOUNT FOCUS. Run fresh baselines then the cross test FIRST, before
    # repeatedly saturating B, so k2 isn't pre-throttled. Per-key 429 breakdown
    # then tells: both keys reject ~evenly = ORG cap; only k2 = confound/independent.
    COOLDOWN = 90
    r_k1 = await ramp("k1 alone (acct A)", {"k1": cl["k1"]})
    print(f"\n--- cooldown {COOLDOWN}s ---", flush=True)
    await asyncio.sleep(COOLDOWN)
    r_k2 = await ramp("k2 alone (acct B)", {"k2": cl["k2"]})
    print(f"\n--- cooldown {COOLDOWN}s ---", flush=True)
    await asyncio.sleep(COOLDOWN)
    r_ab = await ramp("k1+k2 (cross acct A+B) [decisive]", {"k1": cl["k1"], "k2": cl["k2"]})
    r_k3 = r_bb = None  # not needed; per-account already established

    def onset(rows):
        return next((r["conc"] for r in rows if r["r429"] > 0), None)

    print("\n================ CROSS-ACCOUNT VERDICT ================", flush=True)
    print(f"k1 alone (A)        : 429 onset @ conc {onset(r_k1)}/key")
    print(f"k2 alone (B)        : 429 onset @ conc {onset(r_k2)}/key")
    print(f"k1+k2 (A+B) combined: 429 onset @ conc {onset(r_ab)}/key")
    # Decisive: at the level where k1+k2 first 429s, who got rejected?
    first = next((r for r in r_ab if r["r429"] > 0), None)
    if first:
        pk = first["perkey"]
        print(f"\nAt the k1+k2 429 onset (conc {first['conc']}/key), per-key rejections: {pk}")
        a = pk.get("k1", 0); b = pk.get("k2", 0)
        o1, o2 = onset(r_k1), onset(r_k2)
        if a > 0 and b > 0:
            print("  => BOTH accounts reject together at a combined load each sustains alone")
            print("     => ORG-LEVEL cap (accounts A and B share a budget).")
        elif b > 0 and a == 0:
            print("  => only k2(B) rejects; k1(A) fine => accounts INDEPENDENT")
            print("     (k1+k2 early 429 was just B hitting its own per-account cap).")
        elif a > 0 and b == 0:
            print("  => only k1(A) rejects => A's ceiling is simply lower than B's; accounts INDEPENDENT.")
    else:
        print("\n  k1+k2 never hit 429 up to MAX_CONC — accounts scale independently (no shared cap found).")


if __name__ == "__main__":
    asyncio.run(main())
