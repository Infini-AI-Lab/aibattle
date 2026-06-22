"""Per-pair supervisor + watchdog for board-game coverage extension, with a
shared cross-process concurrency pool.

Design goals (all three at once):
  1. SHARED concurrency: every pair draws from ONE global pool of TOTAL_CAP
     slots (a CrossProcGate of lock files in GATE_DIR), not a static per-pair
     split. Whoever needs capacity gets it; the fleet total never exceeds
     TOTAL_CAP, so the account ceiling is respected.
  2. ISOLATION: each under-target pair runs as its OWN extend_board.py
     subprocess (scoped via PAIRS=a/b). A hang in one pair cannot touch another.
  3. INDIVIDUAL RESTART: a watchdog polls each pair's last move-time in
     steps.jsonl; a pair alive but silent > STALE_MIN minutes is killed and
     relaunched on fresh sockets ALONE. Per-episode resume refills only its
     missing episodes; its freed gate slots are reaped automatically.

A pair that reaches TARGET exits on its own and is reaped. When all pairs reach
TARGET the supervisor exits.

Eval settings are unchanged: children inherit max_tokens / temperature /
timeout / retries verbatim; concurrency is the same frozen TOTAL_CAP, now
shared rather than partitioned.

Env: GAME (gomoku), BOARD_TARGET (30), STALE_MIN (60), TOTAL_CAP (64),
POLL_SEC (120), GATE_DIR (default /tmp/aibattle_gate_<GAME>).
"""
from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import time

GAME = os.environ.get("GAME", "gomoku")
TARGET = int(os.environ.get("BOARD_TARGET", "30"))
STALE_MIN = float(os.environ.get("STALE_MIN", "60"))
TOTAL_CAP = int(os.environ.get("TOTAL_CAP", "64"))
POLL = float(os.environ.get("POLL_SEC", "120"))
GATE_DIR = os.environ.get("GATE_DIR", f"/tmp/aibattle_gate_{GAME}")

SEVEN = {"deepseek-v4-pro", "kimi-k2p6", "minimax-m3", "glm-5p1", "gpt-oss-120b",
         "qwen3p7-plus", "glm-5p2"}
STEPS = f"runs/{GAME}/steps.jsonl"


def bare(x):
    return x[: -len("-coached")] if x.endswith("-coached") else x


def under_pairs():
    """List (a, b, n) for 7-set pairs of GAME with n < TARGET episodes on disk."""
    out = []
    for d in sorted(glob.glob(f"runs/{GAME}/{GAME}__*")):
        if not os.path.isdir(d):
            continue
        nm = os.path.basename(d)[len(GAME) + 2:]
        if "__vs__" not in nm:
            continue
        a, b = (bare(x) for x in nm.split("__vs__", 1))
        if not (a in SEVEN and b in SEVEN):
            continue
        n = len(glob.glob(os.path.join(d, "ep*.json")))
        if n < TARGET:
            out.append((a, b, n))
    return out


def last_move_age():
    """pair-label -> minutes since its most recent step in steps.jsonl."""
    if not os.path.exists(STEPS):
        return {}
    now = time.time()
    latest = {}
    with open(STEPS, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            p = r.get("pair")
            t = r.get("t", 0)
            if p is None:
                continue
            if p not in latest or t > latest[p]:
                latest[p] = t
    return {p: (now - t) / 60 for p, t in latest.items()}


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reap_gate(extra_dead_pid=None):
    """Unlink gate slot files whose holder PID is dead (or a just-killed PID)."""
    if not os.path.isdir(GATE_DIR):
        return
    for f in glob.glob(os.path.join(GATE_DIR, "slot*")):
        try:
            with open(f, encoding="utf-8") as fh:
                pid = int(fh.read() or 0)
        except (OSError, ValueError):
            continue
        if pid and (pid == extra_dead_pid or not _alive(pid)):
            try:
                os.unlink(f)
            except OSError:
                pass


def launch(a, b):
    env = dict(os.environ)
    env["GAMES"] = GAME
    env["PAIRS"] = f"{a}/{b}"
    env["BOARD_TARGET"] = str(TARGET)
    env["GATE_DIR"] = GATE_DIR
    env["GATE_N"] = str(TOTAL_CAP)
    env.pop("DRY_RUN", None)
    log = open(f"/tmp/sup_{GAME}_{a}__{b}.log", "a", encoding="utf-8")
    return subprocess.Popen([sys.executable, "scripts/extend_board.py"],
                            env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    initial = under_pairs()
    if not initial:
        print(f"supervise {GAME}: nothing under {TARGET}/pair — done.", flush=True)
        return
    os.makedirs(GATE_DIR, exist_ok=True)
    # Start from a clean gate so no stale slots from a prior run reduce capacity.
    for f in glob.glob(os.path.join(GATE_DIR, "slot*")):
        try:
            os.unlink(f)
        except OSError:
            pass
    print(f"supervise {GAME}: {len(initial)} pairs < {TARGET}, shared gate {TOTAL_CAP} "
          f"@ {GATE_DIR}, stale {STALE_MIN}m, poll {POLL:.0f}s", flush=True)

    procs = {}    # (a, b) -> Popen
    started = {}  # (a, b) -> launch timestamp (grace before stale checks)

    while True:
        up = under_pairs()
        if not up:
            for pr in procs.values():
                if pr.poll() is None:
                    pr.send_signal(signal.SIGKILL)
            print("ALL PAIRS COMPLETE", flush=True)
            return
        live = {(a, b) for (a, b, _) in up}

        # Reap/stop procs whose pair is now complete.
        for key in list(procs):
            if key not in live:
                pr = procs.pop(key)
                started.pop(key, None)
                if pr.poll() is None:
                    pr.send_signal(signal.SIGKILL)
                    reap_gate(extra_dead_pid=pr.pid)
                print(f"complete {key[0]}/{key[1]} -> stopped", flush=True)

        reap_gate()  # clear any slots left by dead/restarted children
        ages = last_move_age()
        for (a, b, n) in up:
            key = (a, b)
            label = f"{a}-coached__vs__{b}-coached"
            pr = procs.get(key)
            if pr is None or pr.poll() is not None:
                procs[key] = launch(a, b)
                started[key] = time.time()
                print(f"launch {a}/{b} (n={n}/{TARGET})", flush=True)
                continue
            # Watchdog: alive but no move for > STALE_MIN, past its launch grace.
            grace = (time.time() - started.get(key, 0)) / 60 < STALE_MIN
            age = ages.get(label, float("inf"))
            if not grace and age > STALE_MIN:
                pr.send_signal(signal.SIGKILL)
                time.sleep(2)
                reap_gate(extra_dead_pid=pr.pid)
                procs[key] = launch(a, b)
                started[key] = time.time()
                print(f"RESTART {a}/{b} stale {age:.0f}m (n={n}/{TARGET}) on fresh sockets",
                      flush=True)

        time.sleep(POLL)


if __name__ == "__main__":
    main()
