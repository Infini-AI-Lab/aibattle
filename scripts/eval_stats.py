"""Shared per-model token-usage + truncation stats, read from per-episode files.

Every tournament persists one ``ep<NNN>.json`` per episode under each match dir,
each holding the full step records (response.metadata + raw_output). This reads
those and aggregates per model:
  - decisions, truncated count/rate (metadata.finish_reason == "length")
  - completion-token distribution: EXACT from metadata.completion_tokens when
    present, else ESTIMATED from len(raw_output)/4 (older runs without token
    logging). The ``exact`` flag records which.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

_CHARS_PER_TOKEN = 4.0


def collect(run_dir: str) -> dict:
    """run_dir e.g. 'runs/board_tournament' -> {model: stats dict}."""
    per = defaultdict(lambda: {"decisions": 0, "truncated": 0, "toks": [], "exact": 0})
    for f in glob.glob(os.path.join(run_dir, "*", "ep*.json")):
        try:
            e = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        sa = e.get("seat_assignment", {})
        for s in e.get("steps", []):
            resp = s.get("response") or {}
            meta = resp.get("metadata") or {}
            nm = s.get("agent_name") or sa.get(s.get("player"))
            if not nm:
                continue
            d = per[nm]
            d["decisions"] += 1
            if meta.get("truncated"):
                d["truncated"] += 1
            ct = meta.get("completion_tokens")
            if ct is not None:
                d["toks"].append(int(ct))
                d["exact"] += 1
            else:
                raw = resp.get("raw_output")
                if raw is not None:
                    d["toks"].append(len(raw) / _CHARS_PER_TOKEN)
    return per


def rows(per: dict) -> list:
    out = []
    for m, d in per.items():
        toks = sorted(d["toks"])
        n = len(toks)
        dec = d["decisions"] or 1

        def q(p):
            return toks[min(n - 1, int(n * p))] if n else 0

        out.append({
            "model": m,
            "decisions": d["decisions"],
            "truncated": d["truncated"],
            "trunc_rate": round(d["truncated"] / dec, 4),
            "tok_p50": round(q(0.5)),
            "tok_p90": round(q(0.9)),
            "tok_p99": round(q(0.99)),
            "tok_max": round(max(toks)) if toks else 0,
            "tok_mean": round(sum(toks) / n) if n else 0,
            "exact": d["exact"] == n and n > 0,  # all counts exact (vs estimated)
        })
    out.sort(key=lambda r: r["tok_mean"], reverse=True)
    return out
