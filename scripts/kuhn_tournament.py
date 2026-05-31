"""Round-robin Kuhn Poker ("poker light") tournament.

Each of C(5,2)=10 model pairs plays EPISODES seat-swapped hands. Kuhn is a tiny
1-card game (~2 decisions/hand) and very high variance, so treat results as a
directional smell test. Each hand is one episode (per-episode resume on).
Reports win rate, net chips/hand, invalid rate, and truncated rate per model;
writes runs/kuhn_tournament/kuhn_data.json and a board-style HTML report.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time
import traceback
from collections import defaultdict

os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner

# qwen3p6-plus excluded (restrictive per-model 429 limit on this account).
MODELS = ["deepseek-v4-pro", "gpt-oss-120b", "kimi-k2p6", "glm-5p1", "minimax-m2p7"]
EPISODES = 30               # hands per pair (seat-swapped); small smell test
MAX_CONCURRENCY = 128
OUT = "runs/kuhn_tournament"
REPORT_DIR = "reports"
os.makedirs(OUT, exist_ok=True)

_STYLE = """
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1117;color:#e6e6e6;}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 22px 80px;}
  h1{font-size:25px;} .sub{color:#8b93a7;}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px;}
  th,td{padding:6px 8px;text-align:center;border-bottom:1px solid #20242e;}
  th{color:#9aa3b5;} td.model,th.model{text-align:left;font-weight:600;color:#cdd6f4;}
"""


def acfg(name: str) -> dict:
    return {
        "type": "model", "name": name,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{name}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": 0.0, "max_tokens": 131072, "timeout_s": 300,
        },
        "max_retries": 2,
    }


def aggregate(pairs_data: list) -> list:
    hands = defaultdict(int); wins = defaultdict(int); net = defaultdict(float)
    decisions = defaultdict(int); invalid = defaultdict(int); trunc = defaultdict(int)
    for g in pairs_data:
        for e in g["episodes"]:
            seat = e["seat_assignment"]
            wname = e.get("winner_name")
            for p in ("player_0", "player_1"):
                nm = seat[p]
                hands[nm] += 1
                net[nm] += e["returns"][p]
            if wname:
                wins[wname] += 1
            for s in e.get("steps", []):
                nm = s.get("agent_name") or seat.get(s.get("player"))
                decisions[nm] += 1
                if s.get("invalid"):
                    invalid[nm] += 1
                if (s.get("response") or {}).get("metadata", {}).get("truncated"):
                    trunc[nm] += 1
    rows = []
    for m in set(hands):
        h = hands[m] or 1; d = decisions[m] or 1
        rows.append({"model": m, "hands": hands[m],
                     "win_rate": round(wins[m] / h, 3),
                     "net_per_hand": round(net[m] / h, 3),
                     "invalid_rate": round(invalid[m] / d, 4),
                     "truncated_rate": round(trunc[m] / d, 4)})
    rows.sort(key=lambda r: r["net_per_hand"], reverse=True)
    return rows


def write_report(rows: list):
    trows = "".join(
        f"<tr><td>{i}</td><td class='model'>{r['model']}</td>"
        f"<td>{r['win_rate']*100:.0f}%</td><td>{r['net_per_hand']:+.3f}</td>"
        f"<td>{r['invalid_rate']*100:.1f}%</td><td>{r['truncated_rate']*100:.1f}%</td></tr>"
        for i, r in enumerate(rows, 1))
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AI Battle Arena — Kuhn Poker</title><style>{_STYLE}</style></head>
<body><div class="wrap"><h1>🃏 AI Battle Arena — Kuhn Poker (poker light)</h1>
<div class="sub">{EPISODES} hands/pair, round-robin · high variance, directional only</div>
<table><tr><th>#</th><th class='model'>model</th><th>win%</th><th>net/hand</th>
<th>invalid%</th><th>truncated%</th></tr>{trows}</table></div></body></html>"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    for p in (os.path.join(OUT, "kuhn_report.html"),
              os.path.join(REPORT_DIR, "kuhn_tournament_report.html")):
        open(p, "w", encoding="utf-8").write(html)


async def main():
    pairs = list(itertools.combinations(MODELS, 2))
    total = len(pairs)
    data = {"game": "kuhn_poker", "models": MODELS, "episodes_per_pair": EPISODES,
            "pairs": []}
    done = 0
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.perf_counter()
    print(f"Kuhn tournament: {total} pairs x {EPISODES} hands, cap {MAX_CONCURRENCY}, "
          f"per-episode resume on\n", flush=True)

    def save():
        json.dump(data, open(os.path.join(OUT, "kuhn_data.json"), "w"))

    async def play(a, b, seed):
        nonlocal done
        gdir = os.path.join(OUT, f"{a}__vs__{b}")
        os.makedirs(gdir, exist_ok=True)
        runner = Runner(lambda: make_game("kuhn_poker"), on_invalid_action="fallback")
        ta = time.perf_counter()
        try:
            with MatchLogger(None) as lg:
                res = await runner.run_match(
                    make_agent(acfg(a), game_name="kuhn_poker"),
                    make_agent(acfg(b), game_name="kuhn_poker"),
                    episodes=EPISODES, seed=seed, seat_swap=True,
                    logger=lg, semaphore=sem, episode_dir=gdir)
            data["pairs"].append({"a": a, "b": b, "seed": seed, "episodes": res.episodes})
            save()
            done += 1
            drop = f"  DROPPED {res.failures}" if res.failures else ""
            print(f"[{done}/{total}] {a} vs {b} done in {time.perf_counter()-ta:.0f}s "
                  f"| hands={len(res.episodes)}/{EPISODES}{drop}", flush=True)
        except Exception as ex:
            done += 1
            print(f"[{done}/{total}] {a} vs {b} FAILED: {ex}", flush=True)
            traceback.print_exc()

    specs = [(a, b, 9000 + i) for i, (a, b) in enumerate(pairs)]
    random.Random(7).shuffle(specs)
    await asyncio.gather(*(play(a, b, s) for a, b, s in specs))
    save()
    rows = aggregate(data["pairs"])
    write_report(rows)
    print(f"\nKUHN TOURNAMENT DONE in {time.perf_counter()-t0:.0f}s")
    print(f"{'model':<18} win%  net/hand  invalid%  trunc%")
    for r in rows:
        print(f"{r['model']:<18} {r['win_rate']*100:>3.0f}%  {r['net_per_hand']:>+7.3f}  "
              f"{r['invalid_rate']*100:>6.1f}%  {r['truncated_rate']*100:>5.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
