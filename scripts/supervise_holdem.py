"""Per-match supervisor for the holdem_match estimate-act A/B.

Runs N matches per model, each as its OWN subprocess (a single holdem_match
episode = up to 20 hands, plain vs estimate-act). Because a match is sequential
(one API call in flight at a time), MODELS*N matches use ~MODELS*N concurrent
calls, kept under the account's 64 ceiling.

Each match is fully isolated: a watchdog polls every match's match.jsonl; a match
alive but silent > STALE_MIN minutes is killed and relaunched ALONE on fresh
sockets (its incomplete run dir is cleared first; completed matches are other
units and untouched). A match is done when its match.jsonl has an episode record.

Seats alternate by match index so each model plays plain and estimate-act in both
seats equally (cancels position bias); seeds are shared across models so glm and
kimi face identical deals.

Env: N (20), STALE_MIN (60), POLL_SEC (120), BASE_SEED (5000),
MAX_PARALLEL (40), OUT (run-harness/holdem_match_ab).
"""
from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import time

import yaml

N = int(os.environ.get("N", "20"))
STALE_MIN = float(os.environ.get("STALE_MIN", "60"))
POLL = float(os.environ.get("POLL_SEC", "120"))
BASE_SEED = int(os.environ.get("BASE_SEED", "5000"))
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "40"))
OUT = os.environ.get("OUT", "run-harness/holdem_match_ab")
MODELS = {"glm5p2": "glm-5p2", "kimi": "kimi-k2p6"}  # label -> fireworks base


def agent_cfg(base, kind):
    a = {"coached": True,
         "model": {"provider": "fireworks",
                   "model_id": f"accounts/fireworks/models/{base}",
                   "api_key_env": "FIREWORKS_API_KEY",
                   "temperature": 0.6, "max_tokens": 131072, "timeout_s": 900},
         "max_retries": 2}
    if kind == "plain":
        return {"type": "model", "name": f"{base}-plain", **a}
    return {"type": "local", "harness": "holdem_estimate_act",
            "name": f"{base}-estimate-act", **a}


def unit_dir(label, idx):
    return os.path.join(OUT, label, f"m{idx:02d}")


def write_config(label, base, idx):
    d = unit_dir(label, idx)
    os.makedirs(d, exist_ok=True)
    # alternate which seat holds estimate-act, by index
    if idx % 2 == 0:
        p0, p1 = agent_cfg(base, "plain"), agent_cfg(base, "estimate_act")
    else:
        p0, p1 = agent_cfg(base, "estimate_act"), agent_cfg(base, "plain")
    cfg = {
        "game": {"name": "holdem_match", "version": "1.0.0", "params": {}},
        "players": {"player_0": {"agent": p0}, "player_1": {"agent": p1}},
        "run": {"episodes": 1, "seed": BASE_SEED + idx, "seat_swap": False,
                "on_invalid_action": "fallback", "max_concurrency": 1},
        "output": {"dir": d, "save_full_log": True, "save_summary": True,
                   "save_trajectories": True, "save_transcripts": True},
    }
    path = os.path.join(d, "config.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def match_jsonl(label, idx):
    g = glob.glob(os.path.join(unit_dir(label, idx), "run_*", "match.jsonl"))
    return max(g, key=os.path.getmtime) if g else None


def is_done(label, idx):
    mj = match_jsonl(label, idx)
    if not mj:
        return False
    try:
        with open(mj, encoding="utf-8") as fh:
            return any(json.loads(l).get("record_type") == "episode" for l in fh)
    except OSError:
        return False


def clean_runs(label, idx):
    import shutil
    for r in glob.glob(os.path.join(unit_dir(label, idx), "run_*")):
        shutil.rmtree(r, ignore_errors=True)


def launch(label, base, idx):
    clean_runs(label, idx)  # fresh start (incomplete match has no kept episode)
    cfg = write_config(label, base, idx)
    log = open(os.path.join(unit_dir(label, idx), "supervisor.log"), "a", encoding="utf-8")
    env = dict(os.environ)
    return subprocess.Popen([sys.executable, "-m", "aibattle.cli", "run", cfg],
                            env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    units = [(label, base, i) for label, base in MODELS.items() for i in range(N)]
    total = len(units)
    print(f"supervise holdem_match: {len(MODELS)} models x {N} = {total} matches, "
          f"stale {STALE_MIN}m, max parallel {MAX_PARALLEL}, out {OUT}", flush=True)
    procs = {}    # (label, idx) -> Popen
    started = {}  # (label, idx) -> launch ts
    restarts = {}

    while True:
        pending = [(l, b, i) for (l, b, i) in units if not is_done(l, i)]
        if not pending:
            for pr in procs.values():
                if pr.poll() is None:
                    pr.send_signal(signal.SIGKILL)
            print(f"ALL {total} MATCHES COMPLETE", flush=True)
            return
        running = sum(1 for pr in procs.values() if pr.poll() is None)

        for (label, base, idx) in pending:
            key = (label, idx)
            pr = procs.get(key)
            alive = pr is not None and pr.poll() is None
            if alive:
                # watchdog: silent too long?
                mj = match_jsonl(label, idx)
                age_src = os.path.getmtime(mj) if mj else started.get(key, time.time())
                stale_min = (time.time() - age_src) / 60
                grace = (time.time() - started.get(key, 0)) / 60 < STALE_MIN
                if not grace and stale_min > STALE_MIN:
                    pr.send_signal(signal.SIGKILL)
                    time.sleep(1)
                    restarts[key] = restarts.get(key, 0) + 1
                    procs[key] = launch(label, base, idx)
                    started[key] = time.time()
                    print(f"RESTART {label} m{idx:02d} (stale {stale_min:.0f}m, "
                          f"restart #{restarts[key]})", flush=True)
                continue
            # not alive: process is None (never launched) or exited-but-not-done (crashed)
            if pr is not None and pr.poll() is not None:
                restarts[key] = restarts.get(key, 0) + 1
                tag = f" (crash relaunch #{restarts[key]})"
            else:
                tag = ""
            if running >= MAX_PARALLEL:
                continue  # respect parallelism cap; pick up next poll
            procs[key] = launch(label, base, idx)
            started[key] = time.time()
            running += 1
            print(f"launch {label} m{idx:02d} seed {BASE_SEED+idx}{tag}", flush=True)

        done_n = total - len(pending)
        print(f"[progress] done {done_n}/{total}  running {running}  "
              f"restarts {sum(restarts.values())}", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
