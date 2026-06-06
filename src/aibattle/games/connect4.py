"""Connect Four (Four in a Row) — 6x7, two players, perfect information.

Players alternate dropping a piece into a column; it falls to the lowest empty
cell. First to connect four (H/V/diagonal) wins; a full board is a draw.
Actions are a single column id ("0".."6") — discrete, no amount.

To create variety across episodes (two deterministic agents would otherwise
replay the identical game), ``random_open`` forced-random legal plies are
played from the per-episode RNG before the agents take over.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game
from .board import connects, empty_grid, is_full, with_cell

_PLAYERS = ["player_0", "player_1"]
_SYM = {"player_0": "X", "player_1": "O", None: "."}
ROWS, COLS, NEED = 6, 7, 4


def _other(p: PlayerId) -> PlayerId:
    return _PLAYERS[1 - _PLAYERS.index(p)]


@dataclass
class Connect4State:
    grid: tuple                       # ROWS x COLS, None | player id
    to_act: PlayerId
    last: Optional[tuple]             # (r, c) of last move
    winner: Optional[PlayerId]
    done: bool


class ConnectFour(Game):
    name = "connect4"
    version = "1.0.0"
    players = list(_PLAYERS)

    def __init__(self, random_open: int = 2):
        self.random_open = int(random_open)

    # -- setup --------------------------------------------------------------
    def _drop(self, grid, col: int, player: PlayerId):
        """Place player's piece in column; return (new_grid, landing_row)."""
        for r in range(ROWS - 1, -1, -1):
            if grid[r][col] is None:
                return with_cell(grid, r, col, player), r
        raise ValueError("column full")

    def initial_state(self, rng: random.Random) -> Connect4State:
        grid = empty_grid(ROWS, COLS)
        to_act = "player_0"
        last = None
        # Forced random opening plies for episode variety. Only play moves that
        # keep the position NON-TERMINAL (no win, board not full), so the agents
        # always inherit a live, undecided game regardless of random_open.
        for _ in range(self.random_open):
            candidates = []
            for c in range(COLS):
                if grid[0][c] is None:
                    ng, r = self._drop(grid, c, to_act)
                    if not connects(ng, r, c, to_act, NEED) and not is_full(ng):
                        candidates.append((c, ng, r))
            if not candidates:
                break
            col, grid, r = rng.choice(candidates)
            last = (r, col)
            to_act = _other(to_act)
        return Connect4State(grid=grid, to_act=to_act, last=last,
                             winner=None, done=False)

    # -- core ---------------------------------------------------------------
    def current_player(self, s: Connect4State) -> PlayerId:
        return s.to_act

    def is_terminal(self, s: Connect4State) -> bool:
        return s.done

    def legal_actions(self, s: Connect4State, player: PlayerId) -> list:
        return [str(c) for c in range(COLS) if s.grid[0][c] is None]

    def step(self, s: Connect4State, move: Move) -> Connect4State:
        assert not s.done
        col = int(move.type)
        player = s.to_act
        grid, r = self._drop(s.grid, col, player)
        won = connects(grid, r, col, player, NEED)
        full = is_full(grid)
        return Connect4State(
            grid=grid, to_act=_other(player), last=(r, col),
            winner=player if won else None, done=won or full,
        )

    def returns(self, s: Connect4State) -> dict:
        assert s.done
        if s.winner is None:
            return {p: 0.0 for p in _PLAYERS}
        loser = _other(s.winner)
        return {s.winner: 1.0, loser: -1.0}

    def episode_metadata(self, s: Connect4State) -> dict:
        return {"reason": "win" if s.winner else "draw"}

    # -- invalid-move fallback: center column, else nearest legal to center --
    def fallback_action(self, s: Connect4State, player: PlayerId, legal: list) -> Move:
        center = COLS // 2
        cols = sorted((int(c) for c in legal), key=lambda c: abs(c - center))
        return Move(type=str(cols[0])) if cols else Move(type="__invalid__")

    # -- observation / render ----------------------------------------------
    def observation(self, s: Connect4State, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        return Observation(
            player=player,
            private={},  # perfect information: nothing hidden
            public={
                "board": [[c for c in row] for row in s.grid],
                "your_symbol": _SYM[player],
                "opp_symbol": _SYM[_other(player)],
                "rows": ROWS, "cols": COLS,
            },
            history=[],
            legal_actions=legal,
            rendered=self._render(s, player, legal),
        )

    def _grid_str(self, s: Connect4State) -> str:
        header = " " + " ".join(str(c) for c in range(COLS))
        lines = [header]
        for row in s.grid:
            lines.append(" " + " ".join(_SYM[cell] for cell in row))
        return "\n".join(lines)

    def _render(self, s: Connect4State, player: PlayerId, legal: list) -> str:
        # Pure state; the rules prose lives in the agent's prompt template.
        return (
            f"You are {_SYM[player]}.\n"
            f"{self._grid_str(s)}\n"
            f"Legal columns: {', '.join(legal)}."
        )

    def render(self, s: Connect4State, *, perspective: Optional[PlayerId] = None) -> str:
        tag = ""
        if s.done:
            tag = f"  [winner: {_SYM[s.winner]}]" if s.winner else "  [draw]"
        return self._grid_str(s) + tag
