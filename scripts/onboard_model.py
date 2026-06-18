"""Onboard a NEW model into an existing pool, Chatbot-Arena style (offline).

Given a new model name and a directory of unified per-game raw-data files (see
``aibattle.eval.arena`` for the interface), this script, *per game*:

  * versus games  -> uses Chatbot Arena active sampling to pick opponents,
    plays battles, fits weighted Bradley-Terry (pool semi-frozen via an L2
    anchor, IPW de-biased), bootstraps CIs, and reports rank / Elo / CI /
    rank-spread (with a "Preliminary" flag until the CI tightens).
  * environment games -> plays the new model solo vs the environment and places
    it by mean score against the pool (bootstrap CI + rank spread).

A single GLOBAL scheduler spans every game's (game, opponent) sampling units
under one `--parallel` budget (default 10). Because units are drawn WITH
replacement each time a slot frees, the scheduler naturally runs different
pairs in one game, the same opponent across games, or the same pair repeatedly
— whatever currently maximises information gain.

The async core ``onboard()`` takes injected `battle_fn` / `env_fn`, so it can be
smoke-tested with synthetic outcomes (no API). The CLI wires in real `Runner`
battles.

Usage:
  PYTHONPATH=src python scripts/onboard_model.py \
      --model qwen3p6-plus --pool pool_unified --parallel 10
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from aibattle.eval import arena
from aibattle.eval.arena import GameData, UnitState

NEW = "<new>"  # internal placeholder name for the model being onboarded


# ---------------------------------------------------------------------------
# Per-game scheduling state
# ---------------------------------------------------------------------------
@dataclass
class GameRun:
    data: GameData                     # pool data (unified)
    kind: str
    pool_models: list
    anchor: dict = field(default_factory=dict)     # pool BT prior (frozen-ish)
    pool_comparisons: list = field(default_factory=list)
    new_comparisons: list = field(default_factory=list)   # new model's battles
    units: dict = field(default_factory=dict)      # opponent -> UnitState (versus)
    new_env_scores: list = field(default_factory=list)    # env games
    elo_new: float = arena.INIT_RATING
    done: bool = False
    result: Optional[dict] = None

    @property
    def all_comparisons(self) -> list:
        return self.pool_comparisons + self.new_comparisons


# Outcome of one scheduled battle/episode-block.
#   versus: battle_fn(game, opponent, seed) -> list[float] of new-model outcomes
#   env:    env_fn(game, seed)              -> list[float] of new-model scalars
BattleFn = Callable[[str, str, int], Awaitable[list]]
EnvFn = Callable[[str, int], Awaitable[list]]


def _build_game_run(gd: GameData, anchor_strength: float) -> GameRun:
    gr = GameRun(data=gd, kind=gd.kind, pool_models=list(gd.models))
    if gd.kind == "versus":
        gr.pool_comparisons = gd.comparisons
        # Pool prior: BT over pool-only data, mean-centred -> the semi-frozen anchor.
        pool_bt = arena.weighted_bt_mle(gr.pool_comparisons, gr.pool_models)
        mean_beta = sum(pool_bt.beta.values()) / max(1, len(pool_bt.beta))
        gr.anchor = {m: pool_bt.beta[m] - mean_beta for m in gr.pool_models}
        gr.units = {opp: UnitState(game=gd.game, opponent=opp)
                    for opp in gr.pool_models}
    return gr


# ---------------------------------------------------------------------------
# Scoring / stop condition
# ---------------------------------------------------------------------------
def _refit_versus(gr: GameRun, anchor_strength: float, boot_rounds: int,
                  rng: random.Random) -> dict:
    models = gr.pool_models + [NEW]
    comps = gr.all_comparisons
    bt = arena.weighted_bt_mle(comps, models, anchor=gr.anchor,
                               anchor_strength=anchor_strength)
    gr.elo_new = bt.elo[NEW]
    ci = arena.bootstrap_ci(comps, models, rounds=boot_rounds, rng=rng,
                            anchor=gr.anchor, anchor_strength=anchor_strength)
    spread = arena.rank_spread(bt.elo, ci)
    ranking = bt.ranking()
    return {
        "game": gr.data.game, "kind": "versus",
        "elo": round(bt.elo[NEW], 1),
        "ci": [round(ci[NEW][0], 1), round(ci[NEW][1], 1)],
        "ci_width": round(ci[NEW][1] - ci[NEW][0], 1),
        "rank": ranking.index(NEW) + 1,
        "rank_spread": list(spread[NEW]),
        "battles": sum(u.n for u in gr.units.values()),
        "leaderboard": [{"model": m, "elo": round(bt.elo[m], 1),
                         "ci": [round(ci[m][0], 1), round(ci[m][1], 1)]}
                        for m in ranking],
    }


def _stop_versus(gr: GameRun, res: dict, min_battles: int, max_battles: int,
                 ci_target: float) -> bool:
    b = res["battles"]
    if b >= max_battles:
        return True
    return b >= min_battles and res["ci_width"] <= ci_target


# ---------------------------------------------------------------------------
# The global active-sampling scheduler
# ---------------------------------------------------------------------------
async def onboard(
    new_model: str,
    games: list,                       # list[GameData]
    *,
    battle_fn: BattleFn,
    env_fn: EnvFn,
    parallel: int = 10,
    anchor_strength: float = 5.0,
    min_battles: int = 20,
    max_battles: int = 200,
    ci_target: float = 30.0,
    env_episodes: int = 40,
    boot_rounds: int = 200,
    proximity_scale: float = 200.0,
    seed: int = 0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Drive the global scheduler and return {game: result}.

    `battle_fn`/`env_fn` are injected so the loop is testable without API calls.
    """
    rng = random.Random(seed)
    versus = [_build_game_run(g, anchor_strength) for g in games if g.kind == "versus"]
    envs = [_build_game_run(g, anchor_strength) for g in games if g.kind == "environment"]
    vmap = {gr.data.game: gr for gr in versus}

    results: dict = {}

    # ---- environment games: independent, fixed-budget solo runs ----
    async def run_env(gr: GameRun):
        sem = asyncio.Semaphore(parallel)

        async def one(i):
            async with sem:
                ys = await env_fn(gr.data.game, seed * 100000 + i)
                gr.new_env_scores.extend(ys)

        await asyncio.gather(*(one(i) for i in range(env_episodes)))
        pool_scores = gr.data.env_scores()
        place = arena.environment_placement(gr.new_env_scores, pool_scores,
                                            rounds=boot_rounds,
                                            rng=random.Random(seed + 1))
        lb = [{"model": (m if m != "<new>" else NEW), "mean": round(mu, 3)}
              for m, mu in place.leaderboard]
        gr.result = {
            "game": gr.data.game, "kind": "environment",
            "mean": round(place.mean, 3),
            "ci": [round(place.ci[0], 3), round(place.ci[1], 3)],
            "rank": place.rank, "rank_spread": list(place.rank_spread),
            "episodes": len(gr.new_env_scores), "leaderboard": lb,
        }
        gr.done = True
        results[gr.data.game] = gr.result

    # ---- versus games: one shared active-sampling scheduler ----
    async def run_versus_pool():
        if not versus:
            return
        # seed an initial result so partial state is always reportable
        for gr in versus:
            gr.result = _refit_versus(gr, anchor_strength, boot_rounds,
                                      random.Random(seed))

        inflight: set = set()
        next_id = 0

        def active_units():
            us, grs = [], []
            for gr in versus:
                if gr.done:
                    continue
                for u in gr.units.values():
                    us.append(u)
                    grs.append(gr)
            return us, grs

        def draw_one():
            us, grs = active_units()
            if not us:
                return None
            opp_elo = {}
            for gr in versus:
                bt_elo = {m["model"]: m["elo"] for m in gr.result["leaderboard"]}
                for u in gr.units.values():
                    opp_elo[(gr.data.game, u.opponent)] = bt_elo.get(u.opponent,
                                                                     arena.INIT_RATING)
            w = []
            for u, gr in zip(us, grs):
                g = arena.info_gain(u)
                prox = arena.proximity_weight(
                    gr.elo_new, opp_elo[(gr.data.game, u.opponent)], proximity_scale)
                w.append(max(g * prox, 1e-3))
            import numpy as np
            w = np.array(w)
            probs = w / w.sum()
            k = rng.choices(range(len(us)), weights=list(probs), k=1)[0]
            return us[k], grs[k], float(probs[k])

        async def play(unit: UnitState, gr: GameRun, p_sample: float, bid: int):
            ys = await battle_fn(gr.data.game, unit.opponent, seed * 100000 + bid)
            weight = 1.0 / max(p_sample, 1e-9)
            for y in ys:
                unit.record(y)
                gr.new_comparisons.append(
                    arena.Comparison(NEW, unit.opponent, y, weight))
            gr.result = _refit_versus(gr, anchor_strength, boot_rounds,
                                      random.Random(seed + bid))
            if _stop_versus(gr, gr.result, min_battles, max_battles, ci_target):
                gr.done = True
            results[gr.data.game] = gr.result
            if on_progress:
                on_progress(gr.data.game, gr.result)

        # continuous fill: keep `parallel` battles in flight
        while True:
            while len(inflight) < parallel:
                pick = draw_one()
                if pick is None:
                    break
                unit, gr, p = pick
                t = asyncio.create_task(play(unit, gr, p, next_id))
                next_id += 1
                inflight.add(t)
            if not inflight:
                break
            done, inflight = await asyncio.wait(inflight,
                                                return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()  # surface exceptions

    await asyncio.gather(run_versus_pool(), *(run_env(gr) for gr in envs))
    for gr in versus:
        results[gr.data.game] = gr.result
    return results


# ---------------------------------------------------------------------------
# Real-battle wiring (Fireworks via the existing runner)
# ---------------------------------------------------------------------------
def _acfg(label: str, max_tokens: int, timeout_s: float, coached: bool) -> dict:
    base = label[:-len("-coached")] if label.endswith("-coached") else label
    return {
        "type": "model", "name": label, "coached": coached,
        "model": {
            "provider": "fireworks",
            "model_id": f"accounts/fireworks/models/{base}",
            "api_key_env": "FIREWORKS_API_KEY",
            "temperature": 0.6, "max_tokens": max_tokens, "timeout_s": timeout_s,
        },
        "max_retries": 2,
    }


def _make_real_fns(new_model: str, out_dir: str, ep_per_battle: int,
                   coached: bool, max_tokens: int, timeout_s: float):
    from aibattle.agents.registry import make_agent
    from aibattle.games.registry import make_game
    from aibattle.logging.logger import MatchLogger
    from aibattle.runner.runner import Runner

    def _outcomes(res, game) -> list:
        ys = []
        for e in res.episodes:
            seat, ret = e["seat_assignment"], e["returns"]
            ns = "player_0" if seat["player_0"] == new_model else "player_1"
            os_ = "player_1" if ns == "player_0" else "player_0"
            d = ret[ns] - ret[os_]
            ys.append(1.0 if d > 0 else (0.0 if d < 0 else 0.5))
        return ys

    async def battle_fn(game, opponent, seed):
        gdir = os.path.join(out_dir, game, f"{new_model}__vs__{opponent}_{seed}")
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(_acfg(new_model, max_tokens, timeout_s, coached),
                           game_name=game),
                make_agent(_acfg(opponent, max_tokens, timeout_s, coached),
                           game_name=game),
                episodes=ep_per_battle, seed=seed, seat_swap=True,
                logger=lg, max_concurrency=ep_per_battle, episode_dir=gdir)
        return _outcomes(res, game)

    async def env_fn(game, seed):
        gdir = os.path.join(out_dir, game, f"{new_model}__vs__env_{seed}")
        runner = Runner(lambda: make_game(game), on_invalid_action="fallback")
        with MatchLogger(None) as lg:
            res = await runner.run_match(
                make_agent(_acfg(new_model, max_tokens, timeout_s, coached),
                           game_name=game),
                make_agent({"type": "builtin", "name": "blackjack_dealer"},
                           game_name=game),
                episodes=1, seed=seed, seat_swap=False,
                logger=lg, episode_dir=gdir)
        return [float(e["returns"]["player_0"]) for e in res.episodes]

    return battle_fn, env_fn


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(new_model: str, results: dict, out_dir: str, report_dir: str):
    os.makedirs(report_dir, exist_ok=True)
    lines = [f"# Onboarding report — {new_model}", ""]
    for game in sorted(results):
        r = results[game]
        if r is None:
            continue
        prelim = (r.get("kind") == "versus"
                  and r["ci_width"] > 30.0)
        lines.append(f"## {game}  ({r['kind']})")
        tag = "  _(Preliminary)_" if prelim else ""
        if r["kind"] == "versus":
            lines.append(f"- rank **{r['rank']}** (spread {r['rank_spread']}), "
                         f"Elo **{r['elo']}** CI {r['ci']} (width {r['ci_width']}), "
                         f"{r['battles']} battles{tag}")
            lines.append("")
            lines.append("| rank | model | elo | ci |")
            lines.append("|---|---|---|---|")
            for i, row in enumerate(r["leaderboard"], 1):
                mark = " ⟵ new" if row["model"] == NEW else ""
                lines.append(f"| {i} | {row['model']}{mark} | {row['elo']} | {row['ci']} |")
        else:
            lines.append(f"- rank **{r['rank']}** (spread {r['rank_spread']}), "
                         f"mean **{r['mean']}** CI {r['ci']}, {r['episodes']} episodes")
            lines.append("")
            lines.append("| rank | model | mean |")
            lines.append("|---|---|---|")
            for i, row in enumerate(r["leaderboard"], 1):
                mark = " ⟵ new" if row["model"] == NEW else ""
                lines.append(f"| {i} | {row['model']}{mark} | {row['mean']} |")
        lines.append("")
    md = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write(md)
    rp = os.path.join(report_dir, f"onboarding_{new_model}.md")
    open(rp, "w", encoding="utf-8").write(md)
    json.dump(results, open(os.path.join(report_dir,
              f"onboarding_{new_model}.json"), "w"), indent=2)
    print(f"\nReport -> {rp}")


def _load_pool(pool_dir: str, want_games: Optional[set]) -> list:
    games = []
    for f in sorted(glob.glob(os.path.join(pool_dir, "*.json"))):
        gd = arena.load_unified(f)
        if want_games and gd.game not in want_games:
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
    ap.add_argument("--ci-target", type=float, default=30.0)
    ap.add_argument("--env-episodes", type=int, default=40)
    ap.add_argument("--anchor-strength", type=float, default=5.0)
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
        parallel=args.parallel, anchor_strength=args.anchor_strength,
        min_battles=args.min_battles, max_battles=args.max_battles,
        ci_target=args.ci_target, env_episodes=args.env_episodes,
        on_progress=lambda g, r: print(
            f"  [{g}] battles={r.get('battles','-')} elo={r.get('elo','-')} "
            f"ci_w={r.get('ci_width','-')} rank={r.get('rank')}", flush=True))
    write_report(args.model, results, out_dir, args.report_dir)
    print(f"DONE in {time.perf_counter()-t0:.0f}s")


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
