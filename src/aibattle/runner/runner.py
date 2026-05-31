"""Runner layer: coordinates the game/agent interaction loop.

The runner knows nothing about strategy, model providers, or agent internals.
It only orchestrates the standardized protocol and records what happened.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..games.base import Game
from ..games.kuhn import ACTION_PRIORITY
from ..agents.base import Agent
from ..logging.logger import MatchLogger, serialize_step
from ..types import (
    INVALID,
    AgentRequest,
    AgentResponse,
    InvalidInfo,
    StepRecord,
)


def resolve_action(response: AgentResponse, legal: list, policy: str):
    """Apply the invalid-action policy. Returns (action_or_None, InvalidInfo).

    A None action signals a forfeit (only under policy == "forfeit").
    """
    requested = response.action
    if requested in legal:
        return requested, InvalidInfo(invalid=False)

    reason = "no_action" if requested == INVALID else "illegal_action"
    if policy == "forfeit":
        return None, InvalidInfo(True, reason, requested, "forfeit")

    # fallback: pick the highest-priority legal action deterministically
    fallback = next((a for a in ACTION_PRIORITY if a in legal), legal[0])
    return fallback, InvalidInfo(True, reason, requested, "fallback")


@dataclass
class RunResult:
    episodes: list  # list of episode-summary dicts
    log_path: Optional[str]


class Runner:
    def __init__(self, game_factory: Callable[[], Game], *,
                 on_invalid_action: str = "fallback"):
        self.game_factory = game_factory
        self.on_invalid_action = on_invalid_action

    async def run_match(
        self,
        agent_a: Agent,
        agent_b: Agent,
        *,
        episodes: int,
        seed: int,
        seat_swap: bool,
        logger: MatchLogger,
        max_concurrency: int = 1,
        progress: Optional[Callable] = None,
        on_episode_start: Optional[Callable] = None,
        on_step: Optional[Callable] = None,
        on_episode_end: Optional[Callable] = None,
    ) -> RunResult:
        game = self.game_factory()
        master = random.Random(seed)

        logger.match_header({
            "game": game.name,
            "game_version": game.version,
            "agents": {
                "agent_a": {"name": agent_a.name, "type": agent_a.agent_type},
                "agent_b": {"name": agent_b.name, "type": agent_b.agent_type},
            },
            "episodes": episodes,
            "seed": seed,
            "seat_swap": seat_swap,
            "on_invalid_action": self.on_invalid_action,
            "max_concurrency": max_concurrency,
        })

        # Build the full episode plan deterministically up front. Deal seeds are
        # drawn sequentially from the master RNG, so deals are reproducible
        # regardless of the order episodes actually execute in.
        specs = []  # (ep_index, pair_id, deal_seed, p0_agent, p1_agent)
        ep_index = 0
        pair_id = 0
        while ep_index < episodes:
            deal_seed = master.randrange(2**31)
            seatings = [(agent_a, agent_b)]
            if seat_swap:
                seatings.append((agent_b, agent_a))
            for p0_agent, p1_agent in seatings:
                if ep_index >= episodes:
                    break
                specs.append((ep_index, pair_id, deal_seed, p0_agent, p1_agent))
                ep_index += 1
            pair_id += 1

        # Episodes are independent and the game is pure, so run them concurrently
        # under a semaphore. Results are stored by plan index, so output order is
        # deterministic even though completion order is not.
        results = [None] * len(specs)
        sem = asyncio.Semaphore(max(1, max_concurrency))
        completed = 0

        async def _worker(idx, spec):
            nonlocal completed
            ep_i, pair, deal_seed, p0, p1 = spec
            agents = {"player_0": p0, "player_1": p1}
            async with sem:
                res = await self._play_episode(
                    game, agents, deal_seed, ep_i, pair, logger,
                    on_episode_start=on_episode_start,
                    on_step=on_step,
                    on_episode_end=on_episode_end,
                )
            results[idx] = res
            completed += 1
            if progress is not None:
                progress(completed, len(specs), res)

        await asyncio.gather(*(_worker(i, s) for i, s in enumerate(specs)))
        return RunResult(episodes=results, log_path=logger.path)

    async def _play_episode(self, game, agents, deal_seed, ep_index, pair_id, logger,
                            *, on_episode_start=None, on_step=None, on_episode_end=None):
        rng = random.Random(deal_seed)
        state = game.initial_state(rng)
        step_index = 0
        invalid_count = {p: 0 for p in game.players}
        forfeiter = None
        steps = []  # accumulated for the per-episode trajectory

        if on_episode_start is not None:
            on_episode_start({
                "episode": ep_index,
                "pair_id": pair_id,
                "seat_assignment": {p: agents[p].name for p in game.players},
                "agent_types": {p: agents[p].agent_type for p in game.players},
            })

        while not game.is_terminal(state):
            player = game.current_player(state)
            obs = game.observation(state, player)
            request = AgentRequest(
                game=game.name,
                game_version=game.version,
                player=player,
                observation=obs,
                instructions="Respond with exactly one legal action token.",
                step_index=step_index,
                # Deterministic per-decision seed: a pure function of the deal,
                # step, and seat — independent of execution order.
                decision_seed=(deal_seed * 1000003 + step_index * 9176
                               + game.players.index(player)) & 0x7FFFFFFF,
            )
            response = await agents[player].act(request)
            action, info = resolve_action(
                response, obs.legal_actions, self.on_invalid_action
            )
            if info.invalid:
                invalid_count[player] += 1

            rec = StepRecord(
                step_index=step_index,
                player=player,
                observation=obs,
                response=response,
                selected_action=action if action is not None else INVALID,
                invalid_info=info,
            )
            logger.step(ep_index, pair_id, rec)
            steps.append({"agent_name": agents[player].name, **serialize_step(rec)})

            if on_step is not None:
                on_step({
                    "episode": ep_index,
                    "step": step_index,
                    "player": player,
                    "agent_name": agents[player].name,
                    "agent_type": agents[player].agent_type,
                    "action": action if action is not None else INVALID,
                    "raw_output": response.raw_output,
                    "message": response.message,
                })

            if action is None:  # forfeit
                forfeiter = player
                break

            state = game.step(state, action)
            step_index += 1

        if forfeiter is not None:
            opponent = next(p for p in game.players if p != forfeiter)
            returns = {forfeiter: -1.0, opponent: 1.0}
            winner = opponent
        else:
            returns = game.returns(state)
            winner = max(returns, key=returns.get) if returns else None
            if returns and len(set(returns.values())) == 1:
                winner = None  # tie (not possible in Kuhn, but general)

        summary = {
            "episode": ep_index,
            "pair_id": pair_id,
            "seed": deal_seed,
            "seat_assignment": {p: agents[p].name for p in game.players},
            "returns": returns,
            "winner": winner,
            "winner_name": agents[winner].name if winner else None,
            "length": step_index,
            "invalid_count": invalid_count,
            "forfeit": forfeiter is not None,
        }
        logger.episode_end(summary)
        if on_episode_end is not None:
            # Full-information render (reveals all hidden state) for end-of-game.
            on_episode_end(summary, game.render(state))
        # Full self-contained trajectory: summary fields + nested steps.
        return {
            "game": game.name,
            "game_version": game.version,
            **summary,
            "steps": steps,
        }
