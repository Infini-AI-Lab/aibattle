"""Built-in baselines for the board games (Connect Four, Gomoku).

- RandomBoardAgent: uniform over legal moves (works for any board game).
- Connect4HeuristicAgent / GomokuHeuristicAgent: a tactical rule set — take an
  immediate win, else block the opponent's immediate win, else play near center.

All reconstruct the board from the observation and reuse the game's win
detector, so they need no access to the live game object. Stochastic choices use
the request's deterministic per-decision seed (reproducible under parallelism).
"""

from __future__ import annotations

import random

from ..games.board import connects, with_cell
from ..games.gomoku import coord_to_rc, rc_to_coord
from ..types import AgentRequest, AgentResponse
from .base import Agent


def _rng_for(seed_base: int, request: AgentRequest) -> random.Random:
    ds = request.decision_seed
    if ds is not None:
        return random.Random((seed_base * 2654435761 ^ ds) & 0x7FFFFFFF)
    return random.Random(seed_base)


def _grid_tuple(board):
    return tuple(tuple(row) for row in board)


class RandomBoardAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "board_random", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        rng = _rng_for(self._seed, request)
        action = rng.choice(request.observation.legal_actions)
        return AgentResponse(action=action, message="random",
                             metadata={"policy": "board_random"})


class Connect4HeuristicAgent(Agent):
    agent_type = "builtin"
    NEED = 4

    def __init__(self, name: str = "connect4_heuristic", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    def _landing_row(self, grid, col):
        for r in range(len(grid) - 1, -1, -1):
            if grid[r][col] is None:
                return r
        return None

    async def act(self, request: AgentRequest) -> AgentResponse:
        obs = request.observation
        me = request.player
        opp = "player_1" if me == "player_0" else "player_0"
        grid = _grid_tuple(obs.public["board"])
        legal = [int(c) for c in obs.legal_actions]

        # 1) take an immediate win
        for c in legal:
            r = self._landing_row(grid, c)
            if r is not None and connects(with_cell(grid, r, c, me), r, c, me, self.NEED):
                return self._resp(str(c), "win")
        # 2) block opponent's immediate win
        for c in legal:
            r = self._landing_row(grid, c)
            if r is not None and connects(with_cell(grid, r, c, opp), r, c, opp, self.NEED):
                return self._resp(str(c), "block")
        # 3) prefer center
        cols = len(grid[0]); center = cols // 2
        best = min(legal, key=lambda c: abs(c - center))
        return self._resp(str(best), "center")

    def _resp(self, col, why):
        return AgentResponse(action=col, message=f"heuristic:{why}",
                             metadata={"policy": "connect4_heuristic", "why": why})


class GomokuHeuristicAgent(Agent):
    agent_type = "builtin"
    NEED = 5

    def __init__(self, name: str = "gomoku_heuristic", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        obs = request.observation
        me = request.player
        opp = "player_1" if me == "player_0" else "player_0"
        grid = _grid_tuple(obs.public["board"])
        size = len(grid)
        legal = obs.legal_actions

        # 1) take an immediate win
        for coord in legal:
            r, c = coord_to_rc(coord)
            if connects(with_cell(grid, r, c, me), r, c, me, self.NEED):
                return self._resp(coord, "win")
        # 2) block opponent's immediate win
        for coord in legal:
            r, c = coord_to_rc(coord)
            if connects(with_cell(grid, r, c, opp), r, c, opp, self.NEED):
                return self._resp(coord, "block")
        # 3) play adjacent to an existing stone if possible, else center
        center = size // 2
        adj = []
        for coord in legal:
            r, c = coord_to_rc(coord)
            if any(0 <= r + dr < size and 0 <= c + dc < size
                   and grid[r + dr][c + dc] is not None
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                adj.append(coord)
        pool = adj or legal
        best = min(pool, key=lambda x: (lambda rc: abs(rc[0] - center) + abs(rc[1] - center))(coord_to_rc(x)))
        return self._resp(best, "extend" if adj else "center")

    def _resp(self, coord, why):
        return AgentResponse(action=coord, message=f"heuristic:{why}",
                             metadata={"policy": "gomoku_heuristic", "why": why})
