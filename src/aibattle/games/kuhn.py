"""Kuhn Poker — the v0 reference game.

Standard 2-player Kuhn Poker:
- Deck {J, Q, K}, ranked J < Q < K. Each player dealt one card; one card unused.
- Each player antes 1 chip (pot starts at 2). player_0 acts first.
- Action sequences and outcomes:
    check, check        -> showdown, each contributed 1
    check, bet, fold    -> bettor (player_1) wins, folder contributed 1
    check, bet, call    -> showdown, each contributed 2
    bet, fold           -> bettor (player_0) wins, folder contributed 1
    bet, call           -> showdown, each contributed 2

Payoffs are zero-sum chips: the winner gains exactly what the loser put in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..types import Action, Observation, PlayerId
from .base import Game

_RANK = {"J": 0, "Q": 1, "K": 2}
_PLAYERS = ["player_0", "player_1"]
# Deterministic fallback priority used by the runner on invalid actions.
ACTION_PRIORITY = ["fold", "check", "call", "bet"]


@dataclass(frozen=True)
class KuhnState:
    cards: dict                       # {"player_0": "K", "player_1": "Q"}
    history: tuple = field(default_factory=tuple)  # ("check", "bet", ...)


class KuhnPoker(Game):
    name = "kuhn_poker"
    version = "1.0.0"
    players = list(_PLAYERS)

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> KuhnState:
        deck = ["J", "Q", "K"]
        rng.shuffle(deck)
        cards = {"player_0": deck[0], "player_1": deck[1]}
        return KuhnState(cards=cards, history=())

    # -- turn logic ---------------------------------------------------------
    def current_player(self, state: KuhnState) -> PlayerId:
        # Turns strictly alternate, player_0 first.
        return _PLAYERS[len(state.history) % 2]

    def legal_actions(self, state: KuhnState, player: PlayerId) -> list:
        if not state.history:
            return ["check", "bet"]
        if state.history[-1] == "bet":
            return ["call", "fold"]
        # last action was "check" (and we are not yet terminal)
        return ["check", "bet"]

    def is_terminal(self, state: KuhnState) -> bool:
        h = state.history
        if h and h[-1] in ("fold", "call"):
            return True
        if h == ("check", "check"):
            return True
        return False

    def step(self, state: KuhnState, action: Action) -> KuhnState:
        player = self.current_player(state)
        legal = self.legal_actions(state, player)
        assert action in legal, f"illegal action {action!r}; legal={legal}"
        return KuhnState(cards=state.cards, history=state.history + (action,))

    # -- payoffs ------------------------------------------------------------
    def _contributions(self, state: KuhnState) -> dict:
        """Chips each player put in: ante 1, plus 1 for a bet or call."""
        contrib = {p: 1 for p in _PLAYERS}
        for i, act in enumerate(state.history):
            if act in ("bet", "call"):
                contrib[_PLAYERS[i % 2]] += 1
        return contrib

    def returns(self, state: KuhnState) -> dict:
        assert self.is_terminal(state), "returns() called on non-terminal state"
        contrib = self._contributions(state)
        if state.history[-1] == "fold":
            folder = _PLAYERS[(len(state.history) - 1) % 2]
            winner = _PLAYERS[1 - _PLAYERS.index(folder)]
        else:  # showdown: higher card wins
            c0, c1 = state.cards["player_0"], state.cards["player_1"]
            winner = "player_0" if _RANK[c0] > _RANK[c1] else "player_1"
        loser = _PLAYERS[1 - _PLAYERS.index(winner)]
        amount = contrib[loser]
        return {winner: float(amount), loser: float(-amount)}

    # -- observation / rendering -------------------------------------------
    def observation(self, state: KuhnState, player: PlayerId) -> Observation:
        legal = self.legal_actions(state, player)
        pot = sum(self._contributions(state).values())
        history = [
            {"player": _PLAYERS[i % 2], "action": a}
            for i, a in enumerate(state.history)
        ]
        return Observation(
            player=player,
            private={"card": state.cards[player]},
            public={"pot": pot},
            history=history,
            legal_actions=legal,
            rendered=self._render_for(state, player, legal, pot),
        )

    def _render_for(self, state, player, legal, pot) -> str:
        if state.history:
            hist_str = ", ".join(
                f"{_PLAYERS[i % 2]} {a}" for i, a in enumerate(state.history)
            )
        else:
            hist_str = "(no actions yet)"
        return (
            f"You are {player} in Kuhn Poker. Your private card is "
            f"{state.cards[player]} (ranks: J<Q<K). Pot is {pot} chips. "
            f"Action history: {hist_str}. "
            f"Legal actions: {', '.join(legal)}."
        )

    def render(self, state: KuhnState, *, perspective: Optional[PlayerId] = None) -> str:
        if perspective is not None:
            obs = self.observation(state, perspective)
            return obs.rendered
        hist_str = ", ".join(
            f"{_PLAYERS[i % 2]} {a}" for i, a in enumerate(state.history)
        ) or "(no actions yet)"
        cards = ", ".join(f"{p}={c}" for p, c in state.cards.items())
        return f"KuhnPoker[cards: {cards}; history: {hist_str}]"
