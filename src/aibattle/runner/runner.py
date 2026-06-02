"""Runner layer: coordinates the game/agent interaction loop.

The runner knows nothing about strategy, model providers, or agent internals.
It only orchestrates the standardized protocol and records what happened.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..games.base import Game
from ..agents.base import Agent
from ..logging.logger import MatchLogger, serialize_step
from ..types import (
    INVALID,
    AgentRequest,
    AgentResponse,
    InvalidInfo,
    MatchContext,
    Move,
    StepRecord,
)

def resolve_action(game: Game, state, player, response: AgentResponse, policy: str):
    """Validate the agent's move via the game; apply the invalid-action policy.

    Returns (move_or_None, InvalidInfo). A None move signals a forfeit (only
    under policy == "forfeit").
    """
    move = Move(type=response.action, amount=response.amount)
    ok, reason = game.validate_action(state, player, move)
    if ok:
        return move, InvalidInfo(invalid=False)

    if reason is None:
        reason = "no_action" if move.type == INVALID else "illegal_action"
    requested = move.label()
    if policy == "forfeit":
        return None, InvalidInfo(True, reason, requested, "forfeit")

    # Fallback: the game decides a safe legal move (e.g. center column for
    # Connect Four; check>fold>call for poker).
    legal = game.legal_actions(state, player)
    fb = game.fallback_action(state, player, legal)
    return fb, InvalidInfo(True, reason, requested, "fallback")


@dataclass
class RunResult:
    episodes: list  # list of episode-summary dicts (failed episodes excluded)
    log_path: Optional[str]
    failures: int = 0  # episodes dropped due to an error (e.g. exhausted retries)


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
        seed: Optional[int] = None,
        seat_swap: bool,
        logger: MatchLogger,
        max_concurrency: int = 1,
        semaphore: "Optional[asyncio.Semaphore]" = None,
        episode_dir: Optional[str] = None,
        progress: Optional[Callable] = None,
        on_episode_start: Optional[Callable] = None,
        on_step: Optional[Callable] = None,
        on_episode_end: Optional[Callable] = None,
    ) -> RunResult:
        game = self.game_factory()
        # seed=None -> every episode draws its own independent random deal seed
        # (fully random, independent games; the seed is saved per-episode so each
        # game is self-describing). seed=int -> deterministic sequence, used where
        # reproducible deals are wanted (e.g. the board tournament's fixed seeds).
        master = random.SystemRandom() if seed is None else random.Random(seed)

        # Per-episode persistence + resume. When episode_dir is set, every episode
        # is written to its own self-contained file ``ep<NNN>.json`` (data + full
        # step log) the moment it finishes, via atomic temp+rename. On a later run
        # an episode whose file already exists is loaded and skipped — no API
        # calls, no shared aggregate to corrupt. Each episode is an independent
        # unit of work keyed by its index, so an interrupt loses at most the
        # episodes that were mid-flight.
        if episode_dir:
            os.makedirs(episode_dir, exist_ok=True)

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

        # Build the full episode plan up front so each episode's deal seed is
        # fixed before execution (independent of completion order). With a fixed
        # `seed` the sequence is reproducible; with seed=None each draw is an
        # independent OS-entropy random.
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
        # An externally supplied semaphore lets several matches share one global
        # concurrency budget (e.g. a tournament running all games in parallel).
        sem = semaphore if semaphore is not None else asyncio.Semaphore(max(1, max_concurrency))
        completed = 0
        # Running cumulative chip standing per agent name, across hands. Exact for
        # sequential (e.g. human) play. Under parallelism, completion order is
        # nondeterministic, so we keep the tally only for sequential play and do
        # NOT expose it in prompts otherwise (see expose_standing) — exposing it
        # would leak run-to-run variation into otherwise reproducible decisions.
        standing = {agent_a.name: 0.0, agent_b.name: 0.0}
        total = len(specs)
        expose_standing = not (semaphore is not None or max_concurrency > 1)

        failures = 0

        async def _worker(idx, spec):
            nonlocal completed, failures
            ep_i, pair, deal_seed, p0, p1 = spec
            epath = (os.path.join(episode_dir, f"ep{ep_i:03d}.json")
                     if episode_dir else None)
            # Resume: a previously-completed episode is loaded from its file and
            # skipped (no model calls). A half-written file can't exist because we
            # persist via atomic rename, but guard against an unreadable one.
            if epath and os.path.exists(epath):
                try:
                    with open(epath, encoding="utf-8") as fh:
                        results[idx] = json.load(fh)
                    completed += 1
                    if progress is not None:
                        progress(completed, len(specs), results[idx])
                    return
                except (json.JSONDecodeError, OSError):
                    pass  # unreadable -> fall through and replay
            agents = {"player_0": p0, "player_1": p1}
            try:
                async with sem:
                    res = await self._play_episode(
                        game, agents, deal_seed, ep_i, pair, logger,
                        standing=standing, total_episodes=total,
                        expose_standing=expose_standing,
                        on_episode_start=on_episode_start,
                        on_step=on_step,
                        on_episode_end=on_episode_end,
                    )
            except Exception as e:  # noqa: BLE001
                # Isolate a single episode's failure (e.g. an exhausted-retry API
                # error) so it neither aborts the match nor orphans its siblings.
                failures += 1
                results[idx] = None
                return
            if epath:
                self._persist_episode(epath, res)
            results[idx] = res
            completed += 1
            if progress is not None:
                progress(completed, len(specs), res)

        await asyncio.gather(*(_worker(i, s) for i, s in enumerate(specs)))
        # Drop episodes that failed (kept None), so callers get a clean list.
        episodes = [r for r in results if r is not None]
        return RunResult(episodes=episodes, log_path=logger.path, failures=failures)

    async def run_table(
        self,
        agents: list,
        *,
        episodes: int,
        seed: Optional[int] = None,
        logger: MatchLogger,
        max_concurrency: int = 1,
        semaphore: "Optional[asyncio.Semaphore]" = None,
        episode_dir: Optional[str] = None,
        seat_rotate: bool = True,
        progress: Optional[Callable] = None,
    ) -> RunResult:
        """Run an N-player game (e.g. Hold'em Table Mode) for ``episodes`` table
        sessions. ``agents`` is a list of N agents, one per seat.

        Mirrors ``run_match`` but seats N agents instead of two and reuses the
        same general per-episode loop (``_play_episode``), per-episode resume
        (``episode_dir``), and shared-semaphore concurrency. By default the
        agent→seat assignment rotates each episode to neutralize positional
        advantage; the game also randomizes the button per session. Standings are
        never exposed in prompts here (sessions run in parallel and are
        nondeterministic in completion order).
        """
        game = self.game_factory()
        n = len(game.players)
        if len(agents) != n:
            raise ValueError(f"run_table needs {n} agents for {game.name}, got {len(agents)}")
        # seed=None -> each session is fully random and independent (its own
        # OS-entropy deal seed, saved per-episode). seed=int -> deterministic.
        master = random.SystemRandom() if seed is None else random.Random(seed)
        if episode_dir:
            os.makedirs(episode_dir, exist_ok=True)

        logger.match_header({
            "game": game.name,
            "game_version": game.version,
            "agents": {game.players[i]: {"name": agents[i].name,
                                         "type": agents[i].agent_type}
                       for i in range(n)},
            "episodes": episodes,
            "seed": seed,
            "seat_rotate": seat_rotate,
            "on_invalid_action": self.on_invalid_action,
        })

        # Plan: each session gets a deal seed (fixed up front so it is independent
        # of completion order) and a seat assignment.
        specs = []  # (ep_index, deal_seed, agents_by_player)
        for ep in range(episodes):
            deal_seed = master.randrange(2**31)
            if seed is None and seat_rotate:
                # Fully random, independent session: seats are a random permutation
                # derived from this session's own seed, so the single saved seed
                # reproduces both the deck and the seating.
                shuffled = list(agents)
                random.Random(deal_seed).shuffle(shuffled)
                by_player = {game.players[i]: shuffled[i] for i in range(n)}
            else:
                # Deterministic seating: rotate by episode index (or pin order).
                rot = ep % n if seat_rotate else 0
                by_player = {game.players[i]: agents[(i + rot) % n] for i in range(n)}
            specs.append((ep, deal_seed, by_player))

        results = [None] * len(specs)
        sem = semaphore if semaphore is not None else asyncio.Semaphore(max(1, max_concurrency))
        completed = 0
        failures = 0
        standing = {a.name: 0.0 for a in agents}

        async def _worker(idx, spec):
            nonlocal completed, failures
            ep_i, deal_seed, by_player = spec
            epath = (os.path.join(episode_dir, f"ep{ep_i:03d}.json")
                     if episode_dir else None)
            if epath and os.path.exists(epath):
                try:
                    with open(epath, encoding="utf-8") as fh:
                        results[idx] = json.load(fh)
                    completed += 1
                    if progress is not None:
                        progress(completed, len(specs), results[idx])
                    return
                except (json.JSONDecodeError, OSError):
                    pass
            try:
                async with sem:
                    res = await self._play_episode(
                        game, by_player, deal_seed, ep_i, 0, logger,
                        standing=standing, total_episodes=len(specs),
                        expose_standing=False,
                    )
            except Exception:  # noqa: BLE001
                failures += 1
                results[idx] = None
                return
            if epath:
                self._persist_episode(epath, res)
            results[idx] = res
            completed += 1
            if progress is not None:
                progress(completed, len(specs), res)

        await asyncio.gather(*(_worker(i, s) for i, s in enumerate(specs)))
        episodes_out = [r for r in results if r is not None]
        return RunResult(episodes=episodes_out, log_path=logger.path, failures=failures)

    @staticmethod
    def _persist_episode(path: str, record: dict) -> None:
        """Write one episode's full record to its own file atomically.

        Write to a temp sibling then os.replace() (atomic on POSIX), so a reader
        or a resume only ever sees a complete file — never a partial write from
        an interrupted run.
        """
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
        os.replace(tmp, path)

    async def _play_episode(self, game, agents, deal_seed, ep_index, pair_id, logger,
                            *, standing=None, total_episodes=0, expose_standing=True,
                            on_episode_start=None, on_step=None, on_episode_end=None):
        rng = random.Random(deal_seed)
        # Snapshot the standing before this hand so all decisions this hand see
        # the same pre-hand totals. Under parallel execution the snapshot is
        # nondeterministic, so it is withheld from the prompt (empty) to keep
        # decisions reproducible; the tally itself is still maintained.
        standing = standing if standing is not None else {}
        pre_standing = dict(standing) if expose_standing else {}
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
                match=MatchContext(
                    episode=ep_index,
                    total_episodes=total_episodes,
                    you=agents[player].name,
                    standing=pre_standing,
                ),
            )
            response = await agents[player].act(request)
            move, info = resolve_action(
                game, state, player, response, self.on_invalid_action
            )
            if info.invalid:
                invalid_count[player] += 1

            sel_type = move.type if move is not None else INVALID
            sel_amount = move.amount if move is not None else None
            rec = StepRecord(
                step_index=step_index,
                player=player,
                observation=obs,
                response=response,
                selected_action=sel_type,
                selected_amount=sel_amount,
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
                    "action": sel_type,
                    "amount": sel_amount,
                    "raw_output": response.raw_output,
                    "message": response.message,
                })

            if move is None:  # forfeit
                forfeiter = player
                break

            state = game.step(state, move)
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

        # Update the running match standing (by agent name) for later hands.
        for p in game.players:
            standing[agents[p].name] = standing.get(agents[p].name, 0.0) + returns[p]

        meta = game.episode_metadata(state) if forfeiter is None else {}
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
            **meta,
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
