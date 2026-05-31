"""Internal 7-card poker hand evaluator.

Given 7 cards (2 hole + 5 board), returns a comparable rank key such that a
stronger hand compares greater, with correct kicker tie-breaking. Evaluates all
C(7,5)=21 five-card combinations and keeps the best — simple and fast enough for
heads-up play.

Cards are 2-character strings: rank in "23456789TJQKA", suit in "cdhs"
(e.g. "Ah", "Td", "2c").
"""

from __future__ import annotations

from itertools import combinations

_RANKS = "23456789TJQKA"
_RANK_VAL = {r: i for i, r in enumerate(_RANKS, start=2)}  # 2..14

# Category ranks (higher is better).
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8


def card_rank(card: str) -> int:
    return _RANK_VAL[card[0]]


def _straight_high(rank_set) -> int:
    """Return the high card of the best straight in a set of rank values, or 0.

    Handles the wheel (A-2-3-4-5) where the Ace plays low (high card = 5).
    """
    ranks = set(rank_set)
    if 14 in ranks:
        ranks.add(1)  # Ace can be low
    best = 0
    for high in range(14, 4, -1):
        if all((high - i) in ranks for i in range(5)):
            best = high
            break
    return best


def _eval_5(cards) -> tuple:
    """Evaluate exactly five cards -> (category, tiebreak_tuple)."""
    ranks = sorted((card_rank(c) for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    # Count rank multiplicities; sort by (count, rank) descending.
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    pattern = [c for _, c in by_count]      # e.g. [3,2] for full house
    ordered_ranks = [r for r, _ in by_count]  # ranks ordered by group strength

    if is_flush and straight_high:
        return (STRAIGHT_FLUSH, (straight_high,))
    if pattern[0] == 4:
        quad = ordered_ranks[0]
        kicker = max(r for r in ranks if r != quad)
        return (QUADS, (quad, kicker))
    if pattern[0] == 3 and pattern[1] == 2:
        return (FULL_HOUSE, (ordered_ranks[0], ordered_ranks[1]))
    if is_flush:
        return (FLUSH, tuple(ranks))
    if straight_high:
        return (STRAIGHT, (straight_high,))
    if pattern[0] == 3:
        trip = ordered_ranks[0]
        kickers = sorted((r for r in ranks if r != trip), reverse=True)
        return (TRIPS, (trip, *kickers))
    if pattern[0] == 2 and pattern[1] == 2:
        high_pair, low_pair = ordered_ranks[0], ordered_ranks[1]
        kicker = max(r for r in ranks if r != high_pair and r != low_pair)
        return (TWO_PAIR, (high_pair, low_pair, kicker))
    if pattern[0] == 2:
        pair = ordered_ranks[0]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)
        return (PAIR, (pair, *kickers))
    return (HIGH_CARD, tuple(ranks))


def evaluate7(cards) -> tuple:
    """Best 5-of-7 hand key. Larger compares as the stronger hand."""
    assert len(cards) == 7, f"expected 7 cards, got {len(cards)}"
    return max(_eval_5(combo) for combo in combinations(cards, 5))


def full_deck() -> list:
    return [r + s for r in _RANKS for s in "cdhs"]


_CATEGORY_NAME = {
    HIGH_CARD: "high card", PAIR: "pair", TWO_PAIR: "two pair", TRIPS: "trips",
    STRAIGHT: "straight", FLUSH: "flush", FULL_HOUSE: "full house",
    QUADS: "four of a kind", STRAIGHT_FLUSH: "straight flush",
}


def category_name(key: tuple) -> str:
    return _CATEGORY_NAME.get(key[0], "?")
