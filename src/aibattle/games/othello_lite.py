"""Othello-lite 6x6 — Reversi on a small board, two players, perfect information.

Black (player_0) moves first. A legal move places a piece on an empty cell that
flips at least one opponent piece along one or more of the eight directions
(horizontal, vertical, diagonal). If a player has no flipping move they must
``pass``; when both players pass in succession the game ends and the player with
more pieces wins.

Board geometry: rows 1..6 top to bottom, columns A..F left to right. A cell is
addressed by a column letter then a row number, e.g. ``C3``. Internally the grid
is row-major ``grid[r][c]`` (``r`` increasing downward, ``c`` rightward), an
empty cell is ``None``. ``pass`` is encoded as the literal action ``"pass"``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game
from .board import empty_grid, with_cell

_PLAYERS = ["player_0", "player_1"]
# player_0 is Black ("B") and moves first; player_1 is White ("W").
_SYM = {"player_0": "B", "player_1": "W", None: "."}
_COLS = "ABCDEF"
SIZE = 6
PASS = "pass"

# Eight directions: horizontal, vertical, and the two diagonals (both signs).
_DIRS = [(-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1),  (1, 0),  (1, 1)]


def _other(p: PlayerId) -> PlayerId:
    return _PLAYERS[1 - _PLAYERS.index(p)]


def coord_to_rc(coord: str):
    """'C3' -> (row_index, col_index), or None if malformed/out of range."""
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


def _flips(grid, r: int, c: int, player: PlayerId) -> list:
    """Cells that placing ``player`` at empty (r, c) would flip. Empty if illegal."""
    if grid[r][c] is not None:
        return []
    opp = _other(player)
    captured = []
    for dr, dc in _DIRS:
        line = []
        rr, cc = r + dr, c + dc
        while 0 <= rr < SIZE and 0 <= cc < SIZE and grid[rr][cc] == opp:
            line.append((rr, cc))
            rr += dr
            cc += dc
        # The run of opponent pieces must be bracketed by one of our own pieces.
        if line and 0 <= rr < SIZE and 0 <= cc < SIZE and grid[rr][cc] == player:
            captured.extend(line)
    return captured


def _legal_cells(grid, player: PlayerId) -> list:
    """All (r, c) where ``player`` has a flipping move, scanning empty cells."""
    out = []
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c] is None and _flips(grid, r, c, player):
                out.append((r, c))
    return out


def _counts(grid) -> dict:
    counts = {"player_0": 0, "player_1": 0}
    for row in grid:
        for cell in row:
            if cell is not None:
                counts[cell] += 1
    return counts


@dataclass
class OthelloState:
    grid: tuple                 # SIZE x SIZE, None | player id
    to_act: PlayerId
    last_was_pass: bool         # did the immediately previous ply pass?
    done: bool


class OthelloLite6x6(Game):
    name = "othello_lite_6x6"
    version = "1.0.0"
    players = list(_PLAYERS)

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> OthelloState:
        grid = empty_grid(SIZE, SIZE)
        # Central 2x2 (rows/cols index 2..3): W B / B W.
        grid = with_cell(grid, 2, 2, "player_1")  # C3 = W
        grid = with_cell(grid, 2, 3, "player_0")  # D3 = B
        grid = with_cell(grid, 3, 2, "player_0")  # C4 = B
        grid = with_cell(grid, 3, 3, "player_1")  # D4 = W
        return OthelloState(grid=grid, to_act="player_0",
                            last_was_pass=False, done=False)

    # -- core ---------------------------------------------------------------
    def current_player(self, s: OthelloState) -> PlayerId:
        return s.to_act

    def is_terminal(self, s: OthelloState) -> bool:
        return s.done

    def legal_actions(self, s: OthelloState, player: PlayerId) -> list:
        cells = _legal_cells(s.grid, player)
        if not cells:
            return [PASS]
        return [rc_to_coord(r, c) for (r, c) in cells]

    def validate_action(self, s: OthelloState, player: PlayerId, move: Move):
        if move.amount is not None:
            return False, "unexpected_amount"
        legal = self.legal_actions(s, player)
        # ``pass`` is only legal when there is genuinely no flipping move.
        if move.type == PASS:
            return (True, None) if legal == [PASS] else (False, "pass_not_allowed")
        if move.type not in legal:
            return False, "illegal_action_type"
        return True, None

    def step(self, s: OthelloState, move: Move) -> OthelloState:
        assert not s.done
        player = s.to_act
        if move.type == PASS:
            # A pass after a pass ends the game (both players are stuck).
            done = s.last_was_pass
            return OthelloState(grid=s.grid, to_act=_other(player),
                                last_was_pass=True, done=done)
        rc = coord_to_rc(move.type)
        assert rc is not None, f"illegal coord {move.type!r}"
        r, c = rc
        captured = _flips(s.grid, r, c, player)
        assert captured, f"move {move.type!r} flips nothing"
        grid = with_cell(s.grid, r, c, player)
        for (fr, fc) in captured:
            grid = with_cell(grid, fr, fc, player)
        nxt = _other(player)
        # If the next player is full-board-stuck and so are we, the game is over.
        done = not _legal_cells(grid, nxt) and not _legal_cells(grid, player)
        return OthelloState(grid=grid, to_act=nxt,
                            last_was_pass=False, done=done)

    def returns(self, s: OthelloState) -> dict:
        assert s.done, "returns() called on non-terminal state"
        counts = _counts(s.grid)
        c0, c1 = counts["player_0"], counts["player_1"]
        if c0 == c1:
            return {"player_0": 0.0, "player_1": 0.0}
        winner = "player_0" if c0 > c1 else "player_1"
        return {winner: 1.0, _other(winner): -1.0}

    def episode_metadata(self, s: OthelloState) -> dict:
        counts = _counts(s.grid)
        c0, c1 = counts["player_0"], counts["player_1"]
        reason = "draw" if c0 == c1 else "majority"
        return {"reason": reason, "piece_counts": counts}

    # -- invalid-move fallback: a corner if available, else first legal -----
    def fallback_action(self, s: OthelloState, player: PlayerId, legal: list) -> Move:
        if legal == [PASS]:
            return Move(type=PASS)
        corners = {rc_to_coord(r, c) for r in (0, SIZE - 1) for c in (0, SIZE - 1)}
        for coord in legal:
            if coord in corners:
                return Move(type=coord)
        return Move(type=legal[0]) if legal else Move(type="__invalid__")

    # -- observation / render ----------------------------------------------
    def observation(self, s: OthelloState, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        counts = _counts(s.grid)
        return Observation(
            player=player,
            private={},  # perfect information: nothing hidden
            public={
                "board": [[c for c in row] for row in s.grid],
                "your_symbol": _SYM[player],
                "opp_symbol": _SYM[_other(player)],
                "size": SIZE,
                "piece_counts": counts,
            },
            history=[],
            legal_actions=legal,
            rendered=self._render(s, player, legal, counts),
        )

    def _grid_str(self, s: OthelloState) -> str:
        header = "  " + " ".join(_COLS)
        lines = [header]
        for r in range(SIZE):
            cells = " ".join(_SYM[s.grid[r][c]] for c in range(SIZE))
            lines.append(f"{r + 1} {cells}")
        return "\n".join(lines)

    def _render(self, s: OthelloState, player, legal, counts) -> str:
        # Pure state; rules prose lives in the agent's prompt template.
        moves = ", ".join(legal)
        return (
            f"You are {_SYM[player]} (B={counts['player_0']}, W={counts['player_1']}).\n"
            f"{self._grid_str(s)}\n"
            f"Legal moves: {moves}."
        )

    def render(self, s: OthelloState, *, perspective: Optional[PlayerId] = None) -> str:
        tag = ""
        if s.done:
            counts = _counts(s.grid)
            c0, c1 = counts["player_0"], counts["player_1"]
            if c0 == c1:
                tag = "  [draw]"
            else:
                tag = f"  [winner: {_SYM['player_0'] if c0 > c1 else _SYM['player_1']}]"
        return self._grid_str(s) + tag
