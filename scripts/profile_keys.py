"""Profile Fireworks rate/throughput limits across keys to find what's SHARED.

Account topology (given):
  key1 = .fireworks    -> account A
  key2 = .fireworks-2  -> account B
  key3 = .fireworks-3  -> account B   (same account as key2)
All under one org.

Question: does adding a key raise capacity, and is the limit per-KEY,
per-ACCOUNT, or per-ORG?

Method — differential load test. For each CONDITION we fire identical bursts on
the active key(s) *simultaneously* and measure aggregate throughput + 429 rate:

  base_k1      key1 alone            (account A baseline)
  base_k2      key2 alone            (account B baseline)
  base_k3      key3 alone            (account B; sanity vs k2)
  same_acct    key2 + key3 together  (both account B)
  cross_acct   key1 + key2 together  (A + B)
  all_three    key1 + key2 + key3

Verdict logic:
  - same_acct aggregate ~= base_k2  -> per-ACCOUNT limit (3rd key adds nothing)
  - same_acct aggregate ~= 2*base_k2 -> per-KEY limit (new key adds capacity)
  - cross_acct ~= base_k1 + base_k2 -> accounts independent
  - cross_acct capped below the sum -> per-ORG limit

We use the FASTEST model (gpt-oss-120b) with a fixed tiny prompt + bounded output
so we measure ENDPOINT limits (admission / TPM), not per-stream decode speed.

Run AFTER the eval finishes (so keys are idle). Usage:
    python scripts/profile_keys.py
"""

from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/gpt-oss-120b"  # fastest decoder -> infra-bound
PROMPT = "Reply with exactly one word: ok."
MAX_TOKENS = 256          # bounded; enough to register TPM without huge cost
REQS_PER_KEY = 120        # burst size per active key per condition
CONCURRENCY = 64          # in-flight per key; high enough to probe admission limits
PAUSE_S = 20              # cooldown between conditions so windows don't bleed over

KEY_FILES = {"k1": ".fireworks", "k2": ".fireworks-2", "k3": ".fireworks-3"}
ACCT = {"k1": "A", "k2": "B", "k3": "B"}


def load_key(path: str) -> str:
    return open(path).read().strip()


async def one_call(client: AsyncOpenAI, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0.0, max_tokens=MAX_TOKENS,
            )
            ct = getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0
            return {"ok": True, "tok": ct, "lat": time.perf_counter() - t0, "kind": "ok"}
        except Exception as e:  # noqa: BLE001
            kind = "429" if "429" in str(e) or "rate" in str(e).lower() else "error"
            return {"ok": False, "tok": 0, "lat": time.perf_counter() - t0, "kind": kind}


async def burst_one_key(name: str, key: str) -> dict:
    """Fire REQS_PER_KEY calls on one key at CONCURRENCY; return per-key metrics."""
    client = AsyncOpenAI(api_key=key, base_url=BASE_URL, timeout=120)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_call(client, sem) for _ in range(REQS_PER_KEY)))
    wall = time.perf_counter() - t0
    ok = sum(r["ok"] for r in results)
    r429 = sum(r["kind"] == "429" for r in results)
    err = sum(r["kind"] == "error" for r in results)
    toks = sum(r["tok"] for r in results)
    return {"key": name, "acct": ACCT[name], "wall": wall, "ok": ok,
            "r429": r429, "err": err, "toks": toks,
            "tok_s": toks / wall if wall else 0, "req_s": ok / wall if wall else 0}


async def condition(label: str, keys: dict) -> dict:
    """Run the named keys' bursts SIMULTANEOUSLY; return per-key + aggregate."""
    print(f"\n>>> {label}: keys={list(keys)} "
          f"({REQS_PER_KEY} reqs/key @ conc {CONCURRENCY})", flush=True)
    per = await asyncio.gather(*(burst_one_key(n, k) for n, k in keys.items()))
    agg_tok_s = sum(p["tok_s"] for p in per)
    agg_req_s = sum(p["req_s"] for p in per)
    tot_429 = sum(p["r429"] for p in per)
    for p in per:
        print(f"    {p['key']}(acct {p['acct']}): ok={p['ok']:>3} 429={p['r429']:>3} "
              f"err={p['err']:>2} | {p['tok_s']:6.0f} tok/s  {p['req_s']:5.1f} req/s "
              f"(wall {p['wall']:.1f}s)", flush=True)
    print(f"    AGGREGATE: {agg_tok_s:.0f} tok/s  {agg_req_s:.1f} req/s  429s={tot_429}",
          flush=True)
    return {"label": label, "per": per, "agg_tok_s": agg_tok_s,
            "agg_req_s": agg_req_s, "tot_429": tot_429}


async def main():
    keys = {n: load_key(p) for n, p in KEY_FILES.items()}
    print(f"Profiling Fireworks limits | model={MODEL.split('/')[-1]} "
          f"| {REQS_PER_KEY} reqs/key @ conc {CONCURRENCY} | max_tokens={MAX_TOKENS}")
    print("accounts: k1=A  k2=B  k3=B (k2/k3 share an account)")

    plan = [
        ("base_k1   (A alone)",        {"k1": keys["k1"]}),
        ("base_k2   (B alone)",        {"k2": keys["k2"]}),
        ("base_k3   (B alone, sanity)",{"k3": keys["k3"]}),
        ("same_acct (k2+k3, both B)",  {"k2": keys["k2"], "k3": keys["k3"]}),
        ("cross_acct(k1+k2, A+B)",     {"k1": keys["k1"], "k2": keys["k2"]}),
        ("all_three (k1+k2+k3)",       keys),
    ]
    out = {}
    for label, ks in plan:
        out[label.split("(")[0].strip()] = await condition(label, ks)
        await asyncio.sleep(PAUSE_S)

    # ---- verdict ----
    b2 = out["base_k2"]["agg_tok_s"]
    b1 = out["base_k1"]["agg_tok_s"]
    same = out["same_acct"]["agg_tok_s"]
    cross = out["cross_acct"]["agg_tok_s"]
    print("\n================ VERDICT ================")
    print(f"base_k2 (1 key, acct B):      {b2:.0f} tok/s, 429s={out['base_k2']['tot_429']}")
    print(f"same_acct (k2+k3, acct B):    {same:.0f} tok/s, 429s={out['same_acct']['tot_429']}")
    ratio = same / b2 if b2 else 0
    print(f"  -> same-account scaling: {ratio:.2f}x")
    if ratio < 1.3:
        print("  => PER-ACCOUNT limit: a 2nd key on the SAME account adds ~no capacity.")
    elif ratio > 1.7:
        print("  => PER-KEY limit: a new key ~doubles capacity (keys metered independently).")
    else:
        print("  => PARTIAL sharing (between per-account and per-key).")
    print(f"cross_acct (k1+k2, A+B):      {cross:.0f} tok/s, 429s={out['cross_acct']['tot_429']}")
    print(f"  sum of solo baselines b1+b2:{b1 + b2:.0f} tok/s")
    cratio = cross / (b1 + b2) if (b1 + b2) else 0
    print(f"  -> cross-account scaling: {cratio:.2f}x of solo sum")
    if cratio < 0.7:
        print("  => suggests an ORG-level cap (different accounts still throttle together).")
    else:
        print("  => accounts scale independently (no obvious org-level cap at this load).")
    print(f"all_three:                    {out['all_three']['agg_tok_s']:.0f} tok/s, "
          f"429s={out['all_three']['tot_429']}")


if __name__ == "__main__":
    asyncio.run(main())
