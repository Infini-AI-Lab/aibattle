"""Onboard a NEW model into an existing pool with Arena-style active sampling.

The point of this script is the **match mechanism**, not a new rating: it uses
Chatbot Arena active sampling to (a) automatically pick informative opponents
for a new model and (b) reach a stable rating in far fewer games than a full
round-robin. Scoring reuses the repo's canonical `scripts/elo_util.py`
(Bradley-Terry → Elo, field mean 1500, chip-weighted for poker, bootstrap CIs),
so the onboarded model's number is directly comparable to the existing reports.

Per game:
  * versus      — active-sampling scheduler plays new-vs-opponent battles, then
    fits Elo over (pool records + new battles) via elo_util; stops when the new
    model's bootstrap error bar is tight enough (or a battle cap is hit).
  * environment — plays the new model solo vs the environment and places it by
    mean score against the pool (bootstrap CI).

A single GLOBAL scheduler spans every game's (game, opponent) units under one
`--parallel` budget; units are drawn with replacement (different pairs / same
pair repeated / same opponent across games). The async core `onboard()` takes
injected `battle_fn` / `env_fn` so it is testable without API calls.

Usage:
  PYTHONPATH=src python scripts/onboard_model.py --model qwen3p6-plus --pool pool_unified --parallel 10
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

# elo_util lives beside this script (the analyzers import it the same way).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elo_util  # noqa: E402

from aibattle.eval import arena  # noqa: E402
from aibattle.eval.arena import GameData, UnitState  # noqa: E402

NEW = "<new>"  # internal placeholder name for the model being onboarded


# ---------------------------------------------------------------------------
# Scoring — thin wrappers over elo_util (the canonical rater)
# ---------------------------------------------------------------------------
def score_versus(models, wld_records, chip_records, basis, n_boot=300):
    """Return (elo dict, ci dict) using the same fit the reports use."""
    if basis == "chips":
        h2h = elo_util.gross_from_records(models, chip_records)
        _, elo = elo_util.bradley_terry(models, h2h)
        ci = elo_util.bootstrap_elo(models, chip_records,
                                    lambda s: elo_util.gross_from_records(models, s),
                                    n_boot=n_boot)
    else:
        h2h = elo_util.wld_from_records(models, wld_records)
        _, elo = elo_util.bradley_terry(models, h2h)
        ci = elo_util.bootstrap_elo(models, wld_records,
                                    lambda s: elo_util.wld_from_records(models, s),
                                    n_boot=n_boot)
    return elo, ci


def _rank_of(elo, model):
    """1-based rank of `model` by Elo (unrated models sink to the bottom)."""
    order = sorted(elo, key=lambda m: elo_util.elo_key(elo, m), reverse=True)
    return order.index(model) + 1, order


# ---------------------------------------------------------------------------
# Per-game scheduling state
# ---------------------------------------------------------------------------
@dataclass
class GameRun:
    data: GameData
    kind: str
    pool_models: list
    basis: str = "wins"
    pool_wld: list = field(default_factory=list)
    pool_chip: list = field(default_factory=list)
    new_wld: list = field(default_factory=list)
    new_chip: list = field(default_factory=list)
    units: dict = field(default_factory=dict)         # opponent -> UnitState
    new_env_scores: list = field(default_factory=list)
    elo: dict = field(default_factory=dict)           # current Elo (pool + new)
    elo_new: float = 1500.0
    done: bool = False
    result: Optional[dict] = None

    @property
    def all_wld(self):
        return self.pool_wld + self.new_wld

    @property
    def all_chip(self):
        return self.pool_chip + self.new_chip


# battle_fn(game, opponent, seed) -> list[(new_payoff, opp_payoff)]
# env_fn(game, seed)              -> list[new_payoff]
BattleFn = Callable[[str, str, int], Awaitable[list]]
EnvFn = Callable[[str, int], Awaitable[list]]


def _build_game_run(gd: GameData) -> GameRun:
    gr = GameRun(data=gd, kind=gd.kind, pool_models=list(gd.models), basis=gd.elo_basis)
    if gd.kind == "versus":
        gr.pool_wld, gr.pool_chip = arena.episodes_to_records(gd.episodes)
        gr.units = {opp: UnitState(game=gd.game, opponent=opp) for opp in gr.pool_models}
    return gr


def _refit_versus(gr: GameRun, boot_rounds: int) -> dict:
    models = gr.pool_models + [NEW]
    elo, ci = score_versus(models, gr.all_wld, gr.all_chip, gr.basis, n_boot=boot_rounds)
    gr.elo = elo
    gr.elo_new = float(elo[NEW]) if elo[NEW] is not None else 1500.0
    rank, order = _rank_of(elo, NEW)
    ci_lo, ci_hi, sd = ci[NEW]["lo"], ci[NEW]["hi"], ci[NEW]["sd"]
    spread = arena.rank_spread(elo, {m: (ci[m]["lo"], ci[m]["hi"]) for m in models})
    return {
        "game": gr.data.game, "kind": "versus", "elo_basis": gr.basis,
        "elo": elo[NEW], "ci": [ci_lo, ci_hi], "sd": sd,
        "rank": rank, "rank_spread": list(spread[NEW]),
        "battles": sum(u.n for u in gr.units.values()),
        "leaderboard": [{"model": m, "elo": elo[m],
                         "ci": [ci[m]["lo"], ci[m]["hi"]]} for m in order],
    }


def _stop_versus(res, min_battles, max_battles, sd_target) -> bool:
    b = res["battles"]
    if b >= max_battles:
        return True
    return b >= min_battles and res["sd"] is not None and res["sd"] <= sd_target


def _place_env(new_scores, pool_scores, boot_rounds, rng):
    pool_means = {m: sum(s) / len(s) for m, s in pool_scores.items() if s}
    new_mean = sum(new_scores) / len(new_scores) if new_scores else 0.0

    def boot(scores):
        if not scores:
            return (0.0, 0.0)
        n = len(scores)
        means = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(boot_rounds))
        return (means[int(0.025 * len(means))], means[min(len(means) - 1, int(0.975 * len(means)))])

    new_ci = boot(new_scores)
    rank = 1 + sum(1 for mu in pool_means.values() if mu > new_mean)
    lb = sorted([(NEW, new_mean)] + list(pool_means.items()),
                key=lambda kv: kv[1], reverse=True)
    return {"game": "", "kind": "environment", "mean": round(new_mean, 3),
            "ci": [round(new_ci[0], 3), round(new_ci[1], 3)], "rank": rank,
            "episodes": len(new_scores),
            "leaderboard": [{"model": m, "mean": round(mu, 3)} for m, mu in lb]}


# ---------------------------------------------------------------------------
# The global active-sampling scheduler
# ---------------------------------------------------------------------------
async def onboard(new_model, games, *, battle_fn, env_fn, parallel=10,
                  min_battles=20, max_battles=200, sd_target=15.0,
                  env_episodes=40, boot_rounds=300, proximity_scale=200.0,
                  seed=0, on_progress=None):
    rng = random.Random(seed)
    versus = [_build_game_run(g) for g in games if g.kind == "versus"]
    envs = [_build_game_run(g) for g in games if g.kind == "environment"]
    results: dict = {}

    async def run_env(gr: GameRun):
        sem = asyncio.Semaphore(parallel)

        async def one(i):
            async with sem:
                gr.new_env_scores.extend(await env_fn(gr.data.game, seed * 100000 + i))

        await asyncio.gather(*(one(i) for i in range(env_episodes)))
        res = _place_env(gr.new_env_scores, gr.data.env_scores(), boot_rounds,
                         random.Random(seed + 1))
        res["game"] = gr.data.game
        gr.result, gr.done, results[gr.data.game] = res, True, res

    async def run_versus_pool():
        if not versus:
            return
        for gr in versus:
            gr.result = _refit_versus(gr, boot_rounds)
        inflight: set = set()
        next_id = 0

        def draw_one():
            us, grs = [], []
            for gr in versus:
                if gr.done:
                    continue
                for u in gr.units.values():
                    us.append(u); grs.append(gr)
            if not us:
                return None
            import numpy as np
            w = np.array([
                arena.info_gain(u) * arena.proximity_weight(
                    gr.elo_new,
                    (gr.elo.get(u.opponent) if gr.elo.get(u.opponent) is not None else 1500.0),
                    proximity_scale)
                for u, gr in zip(us, grs)], dtype=float)
            w = np.maximum(w, 1e-3)
            probs = w / w.sum()
            k = rng.choices(range(len(us)), weights=list(probs), k=1)[0]
            return us[k], grs[k]

        async def play(unit, gr, bid):
            pairs = await battle_fn(gr.data.game, unit.opponent, seed * 100000 + bid)
            for np_, op_ in pairs:
                y = 1.0 if np_ > op_ else (0.0 if np_ < op_ else 0.5)
                unit.record(y)
                gr.new_wld.append((NEW, unit.opponent, 1 if y == 1.0 else (-1 if y == 0.0 else 0)))
                gr.new_chip.append((NEW, unit.opponent, float(np_), float(op_)))
            gr.result = _refit_versus(gr, boot_rounds)
            if _stop_versus(gr.result, min_battles, max_battles, sd_target):
                gr.done = True
            results[gr.data.game] = gr.result
            if on_progress:
                on_progress(gr.data.game, gr.result)

        while True:
            while len(inflight) < parallel:
                pick = draw_one()
                if pick is None:
                    break
                unit, gr = pick
                inflight.add(asyncio.create_task(play(unit, gr, next_id)))
                next_id += 1
            if not inflight:
                break
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()

    await asyncio.gather(run_versus_pool(), *(run_env(gr) for gr in envs))
    for gr in versus:
        results[gr.data.game] = gr.result
    return results


# ---------------------------------------------------------------------------
# Real-battle wiring (Fireworks via the existing runner)
# ---------------------------------------------------------------------------
def _acfg(label, max_tokens, timeout_s, coached):
    base = label[:-len("-coached")] if label.endswith("-coached") else label
    return {"type": "model", "name": label, "coached": coached,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{base}",
                      "api_key_env": "FIREWORKS_API_KEY", "temperature": 0.6,
                      "max_tokens": max_tokens, "timeout_s": timeout_s},
            "max_retries": 2}


def _make_real_fns(new_model, out_dir, ep_per_battle, coached, max_tokens, timeout_s,
                   env_opponent="blackjack_dealer"):
    from aibattle.agents.registry import make_agent
    from aibattle.games.registry import make_game
    from aibattle.logging.logger import MatchLogger
    from aibattle.runner.runner import Runner

    async def battle_fn(game, opponent, seed):
        gdir = os.path.join(out_dir, game, f"{new_model}__vs__{opponent}_{seed}")
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(_acfg(new_model, max_tokens, timeout_s, coached), game_name=game),
                make_agent(_acfg(opponent, max_tokens, timeout_s, coached), game_name=game),
                episodes=ep_per_battle, seed=seed, seat_swap=True,
                logger=lg, max_concurrency=ep_per_battle, episode_dir=gdir)
        pairs = []
        for e in res.episodes:
            seat, ret = e["seat_assignment"], e["returns"]
            ns = "player_0" if seat["player_0"] == new_model else "player_1"
            os_ = "player_1" if ns == "player_0" else "player_0"
            pairs.append((ret[ns], ret[os_]))
        return pairs

    async def env_fn(game, seed):
        gdir = os.path.join(out_dir, game, f"{new_model}__vs__env_{seed}")
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(_acfg(new_model, max_tokens, timeout_s, coached), game_name=game),
                make_agent({"type": "builtin", "name": env_opponent}, game_name=game),
                episodes=1, seed=seed, seat_swap=False, logger=lg, episode_dir=gdir)
        return [float(e["returns"]["player_0"]) for e in res.episodes]

    return battle_fn, env_fn


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(x):
    return "—" if x is None else str(x)


def write_report(new_model, results, out_dir, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    lines = [f"# Onboarding report — {new_model}",
             "", "_Ratings via the repo's canonical Bradley-Terry Elo "
             "(field mean 1500); opponents chosen by Arena active sampling._", ""]
    for game in sorted(results):
        r = results[game]
        if not r:
            continue
        lines.append(f"## {game}  ({r['kind']}{'/' + r['elo_basis'] if r['kind']=='versus' else ''})")
        if r["kind"] == "versus":
            prelim = " _(Preliminary)_" if (r["sd"] is None or r["sd"] > 15) else ""
            lines += [f"- rank **{r['rank']}** (spread {r['rank_spread']}), "
                      f"Elo **{_fmt(r['elo'])}** [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}] "
                      f"±{_fmt(r['sd'])}, {r['battles']} battles{prelim}", "",
                      "| rank | model | elo | ci |", "|---|---|---|---|"]
            for i, row in enumerate(r["leaderboard"], 1):
                mk = " ⟵ new" if row["model"] == NEW else ""
                lines.append(f"| {i} | {row['model']}{mk} | {_fmt(row['elo'])} | "
                             f"[{_fmt(row['ci'][0])}, {_fmt(row['ci'][1])}] |")
        else:
            lines += [f"- rank **{r['rank']}**, mean **{r['mean']}** "
                      f"[{r['ci'][0]}, {r['ci'][1]}], {r['episodes']} episodes", "",
                      "| rank | model | mean |", "|---|---|---|"]
            for i, row in enumerate(r["leaderboard"], 1):
                mk = " ⟵ new" if row["model"] == NEW else ""
                lines.append(f"| {i} | {row['model']}{mk} | {row['mean']} |")
        lines.append("")
    md = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write(md)
    rp = os.path.join(report_dir, f"onboarding_{new_model}.md")
    open(rp, "w", encoding="utf-8").write(md)
    json.dump(results, open(os.path.join(report_dir, f"onboarding_{new_model}.json"), "w"), indent=2)
    print(f"\nReport -> {rp}")


def _load_pool(pool_dir, want):
    games = []
    for f in sorted(glob.glob(os.path.join(pool_dir, "*.json"))):
        gd = arena.load_unified(f)
        if want and gd.game not in want:
            continue
        games.append(gd)
    return games


async def _amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="dir of unified per-game JSON")
    ap.add_argument("--games", default="")
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--min-battles", type=int, default=20)
    ap.add_argument("--max-battles", type=int, default=200)
    ap.add_argument("--sd-target", type=float, default=15.0,
                    help="stop a game once the new model's bootstrap Elo SD ≤ this")
    ap.add_argument("--env-episodes", type=int, default=40)
    ap.add_argument("--ep-per-battle", type=int, default=2)
    ap.add_argument("--coached", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    if "FIREWORKS_API_KEY" not in os.environ and os.path.exists(".fireworks"):
        os.environ["FIREWORKS_API_KEY"] = open(".fireworks").read().strip()

    out_dir = args.out or f"runs/onboarding_{args.model}"
    want = set(args.games.split(",")) if args.games else None
    games = _load_pool(args.pool, want)
    if not games:
        raise SystemExit(f"No unified game files found in {args.pool}")

    battle_fn, env_fn = _make_real_fns(
        args.model, out_dir, args.ep_per_battle, args.coached,
        max_tokens=int(os.environ.get("MAX_TOKENS", "131072")),
        timeout_s=float(os.environ.get("MODEL_TIMEOUT_S", "900")))

    t0 = time.perf_counter()
    results = await onboard(
        args.model, games, battle_fn=battle_fn, env_fn=env_fn,
        parallel=args.parallel, min_battles=args.min_battles,
        max_battles=args.max_battles, sd_target=args.sd_target,
        env_episodes=args.env_episodes,
        on_progress=lambda g, r: print(
            f"  [{g}] battles={r.get('battles','-')} elo={_fmt(r.get('elo'))} "
            f"±{_fmt(r.get('sd'))} rank={r.get('rank')}", flush=True))
    write_report(args.model, results, out_dir, args.report_dir)
    print(f"DONE in {time.perf_counter()-t0:.0f}s")


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
