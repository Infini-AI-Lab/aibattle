"""Shared helpers for grid board games (Connect Four, Gomoku).

The win detector scans only the four axes through the last-placed cell, so it is
cheap and correct regardless of board size. Grids are row-major:
``grid[r][c]`` with ``r`` increasing downward, ``c`` increasing rightward; an
empty cell is ``None``.
"""

from __future__ import annotations

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, two diagonals


def connects(grid, r: int, c: int, player, need: int) -> bool:
    """True if the piece at (r, c) completes a line of >= ``need`` for ``player``."""
    rows, cols = len(grid), len(grid[0])
    for dr, dc in _DIRS:
        count = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == player:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= need:
            return True
    return False


def is_full(grid) -> bool:
    return all(cell is not None for row in grid for cell in row)


def empty_grid(rows: int, cols: int) -> tuple:
    return tuple(tuple(None for _ in range(cols)) for _ in range(rows))


def with_cell(grid, r: int, c: int, value) -> tuple:
    """Return a new immutable grid with (r, c) set to ``value``."""
    rowlist = list(grid[r])
    rowlist[c] = value
    return grid[:r] + (tuple(rowlist),) + grid[r + 1:]
