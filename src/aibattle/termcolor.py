"""Tiny ANSI helper for the interactive terminal display.

Colorizes poker card tokens (e.g. "As", "7h") into suit-colored glyphs
("A♠", "7♥"). Used only for human-facing terminal output; disabled when stdout
is not a TTY or NO_COLOR is set, so logs/files stay clean.
"""

from __future__ import annotations

import os
import re
import sys

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"

_SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
_CARD = re.compile(r"\b([2-9TJQKA])([cdhs])\b")
_NUM = re.compile(r"\b\d+\b")


def _enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def colorize_numbers(text: str) -> str:
    """Bold-yellow any standalone integer (pot, stacks, to-call, ranges)."""
    if not text or not _enabled():
        return text
    return _NUM.sub(lambda m: f"{_BOLD}{_YELLOW}{m.group(0)}{_RESET}", text)


def colorize_cards(text: str) -> str:
    """Replace 2-char card tokens with bold, suit-colored glyphs."""
    if not text or not _enabled():
        return text

    def repl(m: "re.Match") -> str:
        rank, suit = m.group(1), m.group(2)
        color = _RED if suit in "hd" else _CYAN
        return f"{_BOLD}{color}{rank}{_SUIT_SYM[suit]}{_RESET}"

    return _CARD.sub(repl, text)


def decorate(text: str) -> str:
    """Highlight numbers, then cards. Numbers run first so card-rank digits
    (glued to a suit letter, e.g. '7h') are left for the card pass."""
    return colorize_cards(colorize_numbers(text))


def rule(width: int = 64) -> str:
    """A horizontal separator line (dim when colors are enabled)."""
    line = "─" * width
    return f"{_DIM}{line}{_RESET}" if _enabled() else line


def action_label(action: str, amount=None) -> str:
    """Render an action (and optional amount) — bold green action, yellow amount."""
    plain = action if amount is None else f"{action} {amount}"
    if not _enabled():
        return plain
    out = f"{_BOLD}{_GREEN}{action}{_RESET}"
    if amount is not None:
        out += f" {_BOLD}{_YELLOW}{amount}{_RESET}"
    return out
