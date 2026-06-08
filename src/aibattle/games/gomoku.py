"""Gomoku-Lite (Five in a Row) — 9x9, two players, perfect information.

Players alternate placing a stone on any empty cell; first to five in a row
(H/V/diagonal) wins; a full board is a draw. Forbidden-move rules are disabled
in v0. Actions are a coordinate like "E5" — discrete, no amount.

Coordinates: columns A-I (left->right), rows 1-9 (top->bottom). "E5" is the
center. ``random_open`` forced-random plies seed episode variety.
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
SIZE, NEED = 9, 5
_COLS = "ABCDEFGHI"


def _other(p: PlayerId) -> PlayerId:
    return _PLAYERS[1 - _PLAYERS.index(p)]


def coord_to_rc(coord: str):
    """'E5' -> (row_index, col_index), or None if malformed/out of range."""
    coord = coord.strip().upper().replace("-", "")
    if len(coord) < 2:
        return None
    letter, num = coord[0], coord[1:]
    if letter not in _COLS or not num.isdigit():
        return None
    c = _COLS.index(letter)
    r = int(num) - 1
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        return None
    return (r, c)


def rc_to_coord(r: int, c: int) -> str:
    return f"{_COLS[c]}{r + 1}"


@dataclass
class GomokuState:
    grid: tuple
    to_act: PlayerId
    last: Optional[tuple]
    winner: Optional[PlayerId]
    done: bool


class Gomoku(Game):
    name = "gomoku"
    version = "1.0.0"
    players = list(_PLAYERS)

    def __init__(self, random_open: int = 2):
        self.random_open = int(random_open)

    def initial_state(self, rng: random.Random) -> GomokuState:
        grid = empty_grid(SIZE, SIZE)
        to_act = "player_0"
        last = None
        # Only play opening moves that keep the position NON-TERMINAL (no win,
        # board not full), so the agents always inherit a live game.
        for _ in range(self.random_open):
            candidates = []
            for r in range(SIZE):
                for c in range(SIZE):
                    if grid[r][c] is None:
                        ng = with_cell(grid, r, c, to_act)
                        if not connects(ng, r, c, to_act, NEED) and not is_full(ng):
                            candidates.append((r, c, ng))
            if not candidates:
                break
            r, c, grid = rng.choice(candidates)
            last = (r, c)
            to_act = _other(to_act)
        return GomokuState(grid=grid, to_act=to_act, last=last,
                           winner=None, done=False)

    def current_player(self, s: GomokuState) -> PlayerId:
        return s.to_act

    def is_terminal(self, s: GomokuState) -> bool:
        return s.done

    def legal_actions(self, s: GomokuState, player: PlayerId) -> list:
        return [rc_to_coord(r, c) for r in range(SIZE) for c in range(SIZE)
                if s.grid[r][c] is None]

    def validate_action(self, s: GomokuState, player: PlayerId, move: Move):
        if move.amount is not None:
            return False, "unexpected_amount"
        rc = coord_to_rc(move.type)
        if rc is None:
            return False, "illegal_action_type"
        r, c = rc
        if s.grid[r][c] is not None:
            return False, "illegal_action_type"  # occupied cell
        return True, None

    def step(self, s: GomokuState, move: Move) -> GomokuState:
        assert not s.done
        r, c = coord_to_rc(move.type)
        player = s.to_act
        grid = with_cell(s.grid, r, c, player)
        won = connects(grid, r, c, player, NEED)
        full = is_full(grid)
        return GomokuState(grid=grid, to_act=_other(player), last=(r, c),
                           winner=player if won else None, done=won or full)

    def returns(self, s: GomokuState) -> dict:
        assert s.done
        if s.winner is None:
            return {p: 0.0 for p in _PLAYERS}
        return {s.winner: 1.0, _other(s.winner): -1.0}

    def episode_metadata(self, s: GomokuState) -> dict:
        return {"reason": "win" if s.winner else "draw"}

    def fallback_action(self, s: GomokuState, player: PlayerId, legal: list) -> Move:
        # On an unparseable/invalid action, pick a RANDOM legal cell rather than
        # the center, so the penalty for failing to answer is unbiased (a fixed
        # center pick would both help the failing model and skew results).
        return Move(type=str(random.choice(list(legal)))) if legal else Move(type="__invalid__")

    # -- observation / render ----------------------------------------------
    def observation(self, s: GomokuState, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        return Observation(
            player=player,
            private={},
            public={
                "board": [[c for c in row] for row in s.grid],
                "your_symbol": _SYM[player],
                "opp_symbol": _SYM[_other(player)],
                "size": SIZE,
            },
            history=[],
            legal_actions=legal,
            rendered=self._render(s, player),
        )

    def _grid_str(self, s: GomokuState) -> str:
        header = "   " + " ".join(_COLS)
        lines = [header]
        for r in range(SIZE):
            lines.append(f"{r + 1:>2} " + " ".join(_SYM[s.grid[r][c]] for c in range(SIZE)))
        return "\n".join(lines)

    def _render(self, s: GomokuState, player: PlayerId) -> str:
        # Pure state; the rules prose lives in the agent's prompt template.
        return (
            f"You are {_SYM[player]}.\n"
            f"{self._grid_str(s)}"
        )

    def render(self, s: GomokuState, *, perspective: Optional[PlayerId] = None) -> str:
        tag = ""
        if s.done:
            tag = f"  [winner: {_SYM[s.winner]}]" if s.winner else "  [draw]"
        return self._grid_str(s) + tag
