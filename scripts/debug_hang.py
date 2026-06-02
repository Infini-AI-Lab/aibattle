"""Instrumented replay of one stuck connect4 episode to localize the hang.

Replays ep002 of the connect4 deepseek-v4-pro vs glm-5p1 pair (deal_seed
163226644, deepseek=player_0) move by move, wrapping each agent.act() in a
per-move timeout so a hang surfaces with its exact step/player instead of
blocking forever. Prints timing + the raw output of each move.
"""

from __future__ import annotations

import asyncio
import os
import random
import time

os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.runner.runner import resolve_action
from aibattle.types import AgentRequest, MatchContext

DEAL_SEED = 163226644          # ep002
P0, P1 = "deepseek-v4-pro", "glm-5p1"
PER_MOVE_TIMEOUT = 2700        # give the runaway move room to actually finish
CLIENT_TIMEOUT = 2700
STOP_AFTER_STEP = 3            # we only need to confirm the known-bad step 3


def acfg(name: str) -> dict:
    return {"type": "model", "name": name,
            "model": {"provider": "fireworks",
                      "model_id": f"accounts/fireworks/models/{name}",
                      "api_key_env": "FIREWORKS_API_KEY",
                      "temperature": 0.0, "max_tokens": 131072, "timeout_s": CLIENT_TIMEOUT},
            "max_retries": 2}


async def main():
    game = make_game("connect4", {"random_open": 2})
    agents = {"player_0": make_agent(acfg(P0), game_name="connect4"),
              "player_1": make_agent(acfg(P1), game_name="connect4")}
    rng = random.Random(DEAL_SEED)
    state = game.initial_state(rng)
    step = 0
    print(f"replay ep002 connect4 {P0}(p0) vs {P1}(p1) deal={DEAL_SEED}", flush=True)
    while not game.is_terminal(state):
        player = game.current_player(state)
        obs = game.observation(state, player)
        req = AgentRequest(
            game=game.name, game_version=game.version, player=player,
            observation=obs, instructions="Respond with exactly one legal action token.",
            step_index=step,
            decision_seed=(DEAL_SEED * 1000003 + step * 9176 + game.players.index(player)) & 0x7FFFFFFF,
            match=MatchContext(episode=2, total_episodes=10, you=agents[player].name, standing={}),
        )
        t0 = time.perf_counter()
        print(f"[step {step}] {player}={agents[player].name} calling act()...", flush=True)
        try:
            resp = await asyncio.wait_for(agents[player].act(req), timeout=PER_MOVE_TIMEOUT)
        except asyncio.TimeoutError:
            print(f"  *** HANG: act() exceeded {PER_MOVE_TIMEOUT}s at step {step} player {player} "
                  f"({agents[player].name}) ***", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  *** EXCEPTION at step {step}: {type(e).__name__}: {e}", flush=True)
            return
        dt = time.perf_counter() - t0
        move, info = resolve_action(game, state, player, resp, "fallback")
        rawlen = len(resp.raw_output or "")
        meta = resp.metadata or {}
        print(f"  -> action={move.type if move else 'INVALID'} amount={move.amount if move else None} "
              f"in {dt:.1f}s | invalid={info.invalid} finish={meta.get('finish_reason')} "
              f"toks={meta.get('completion_tokens')} rawlen={rawlen} attempts={meta.get('attempts')}", flush=True)
        if move is None:
            print(f"  forfeit by {player} at step {step}", flush=True)
            break
        state = game.step(state, move)
        step += 1
        if STOP_AFTER_STEP is not None and step > STOP_AFTER_STEP:
            print(f"  (confirmed through step {STOP_AFTER_STEP}; stopping diagnostic)", flush=True)
            return
        if step > 60:
            print("  *** STOP: >60 steps, game not terminating (state not advancing?) ***", flush=True)
            return
    ret = game.returns(state)
    print(f"GAME COMPLETE in {step} steps, returns={ret}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
