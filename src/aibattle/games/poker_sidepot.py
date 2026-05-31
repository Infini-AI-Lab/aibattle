"""Side-pot construction for multi-player Texas Hold'em.

When players go all-in for different amounts, not everyone is eligible to win
every chip. This module partitions each player's total hand contribution into
layered pots (main pot + side pots), each carrying the set of players eligible
to win it.

This is the #1 correctness hotspot of multi-player poker, so it is kept as a
small, pure, independently fuzz-tested function (see tests). Invariants it must
satisfy:

- **Conservation:** the sum of all pot amounts equals the sum of all
  contributions (no chips created or destroyed).
- **Eligibility:** a folded player is never eligible to win a pot, though their
  contributed chips still go into the pots.
- **Uncalled chips:** a top layer contributed by a single player (an uncalled
  bet/raise) forms a pot they alone are eligible for — i.e. it is returned to
  them at award time.
"""

from __future__ import annotations

from typing import Iterable


def build_pots(contributions: dict, folded: Iterable) -> list:
    """Partition per-player total contributions into main + side pots.

    Args:
        contributions: {player: total chips contributed this hand} (>= 0).
        folded: players who folded (contributed chips stay in, but they cannot
            win).

    Returns:
        A list of pots from the lowest layer (main pot) upward, each a dict
        ``{"amount": int, "eligible": [players]}``. ``eligible`` lists the
        non-folded contributors at that layer. A pot may have an empty
        ``eligible`` list only in degenerate inputs (all of a layer's
        contributors folded); callers should fold such chips into an adjacent
        pot or award by table rules — in normal showdowns it does not occur.
    """
    folded = set(folded)
    # Work only with players who put chips in.
    remaining = {p: c for p, c in contributions.items() if c > 0}
    pots = []
    while remaining:
        layer = min(remaining.values())          # smallest remaining stake
        contributors = list(remaining.keys())
        amount = layer * len(contributors)
        eligible = [p for p in contributors if p not in folded]
        pots.append({"amount": amount, "eligible": eligible})
        # Peel this layer off every contributor; drop those now at zero.
        remaining = {p: c - layer for p, c in remaining.items() if c - layer > 0}
    return _merge_uncontested(pots)


def _merge_uncontested(pots: list) -> list:
    """Merge any pot whose eligible set is empty into the nearest lower pot.

    Empty-eligible pots only arise in degenerate inputs; merging downward keeps
    chip conservation exact and avoids stranding chips. If the lowest pot itself
    has no eligible players (pathological), it is left as-is for the caller.
    """
    out = []
    for pot in pots:
        if not pot["eligible"] and out:
            out[-1]["amount"] += pot["amount"]
        else:
            out.append(pot)
    return out


def award_pots(pots: list, rank_of: dict) -> dict:
    """Award each pot to the highest-ranked eligible player(s), splitting ties.

    Args:
        pots: output of ``build_pots``.
        rank_of: {player: comparable hand rank} (higher wins). Must contain
            every eligible player across all pots.

    Returns:
        {player: chips won} (only players who win a share appear). Split
        remainders (from integer division) are assigned to the earliest eligible
        winner in list order, deterministically.
    """
    winnings: dict = {}
    for pot in pots:
        eligible = pot["eligible"]
        if not eligible:
            continue
        best = max(rank_of[p] for p in eligible)
        winners = [p for p in eligible if rank_of[p] == best]
        share, rem = divmod(pot["amount"], len(winners))
        for i, p in enumerate(winners):
            winnings[p] = winnings.get(p, 0) + share + (1 if i < rem else 0)
    return winnings
