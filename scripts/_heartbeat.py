"""Tiny per-move debug heartbeat for the Hold'em tournaments.

Writes one timestamped line per decision to ``data-log/<mode>_<ts>.log`` (a
gitignored folder, kept OUT of the runs/ data tree) so a long single-episode
run (match/table) is observable mid-flight. Wired via each runner's ``on_step``
hook. The runs/ folder stays pure experiment data; debug prints live here.
"""

from __future__ import annotations

import os
import time

LOG_DIR = "data-log"


def open_log(mode: str):
    """Open (append, line-buffered) a fresh per-run heartbeat file; return (fh, path)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    return open(path, "a", buffering=1), path


def make_cb(fh, label: str):
    """Build an on_step callback that appends a heartbeat line per decision."""
    def cb(ev):
        pub = ev.get("public") or {}
        hand = pub.get("match_hand") or pub.get("table_hand") or ev.get("episode")
        mx = pub.get("match_max_hands") or pub.get("table_max_hands") or "?"
        street = pub.get("street") or "?"
        amt = ev.get("amount")
        raw = ev.get("raw_output") or ""
        fh.write(
            f"{time.strftime('%H:%M:%S')} [{label}] hand {hand}/{mx} "
            f"{street:<7} s{ev['step']:02d} {ev['agent_name']} -> "
            f"{ev['action']}{(' ' + str(amt)) if amt is not None else ''} "
            f"(out_len={len(raw)})\n"
        )
    return cb
