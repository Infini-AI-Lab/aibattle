# Additional Game Designs for AI Battle Arena

This document summarizes four additional games proposed for AI Battle Arena: Leduc Poker, Independent Blackjack, Othello-lite 6x6, and Repeated Colonel Blotto.

The current goal is to define playable environments and collect clean trajectories. Detailed game-specific analysis and metrics can be added later after enough trajectories are collected.

## 1. Leduc Poker

### Type

Imperfect-information poker game. Two players.

### Purpose

Leduc Poker is an intermediate poker game between Kuhn Poker and Texas Hold'em. It is useful for collecting trajectories involving bluffing, hidden-information reasoning, public-card belief updates, and multi-round betting, while remaining much simpler than full Hold'em.

### Rules

- Deck: J, J, Q, Q, K, K.
- Card strength: K > Q > J.
- Each player antes and receives one private card.
- First betting round.
- One public card is revealed.
- Second betting round.
- If no player folds, showdown happens.

### Actions

The game supports fold, check, call, bet, and raise. For v0, each betting round can be limited to at most one raise to keep the betting logic simple.

### Showdown

- A private card that pairs with the public card beats any non-pair hand.
- If neither player has a pair, the higher private card wins.
- If both players have equivalent hand strength, the pot is split.

### Trajectory Content

A trajectory should record private observations, public state transitions, betting history, legal actions, raw agent outputs, parsed actions, fallback events if any, showdown results, and final chip movement.

## 2. Independent Blackjack

### Type

Risk and probability game. One agent plays independently against the dealer in each evaluation run.

### Purpose

Independent Blackjack evaluates risk calibration, probability reasoning, rule following, and basic strategy. Agents do not directly compete with one another. Each agent independently plays many hands against a fixed dealer policy, and the trajectories can later be analyzed across agents.

### Rules

- Each hand starts with a player hand and a dealer upcard.
- The agent acts until it stands, busts, or doubles.
- The dealer follows a fixed policy.
- Profit is recorded for the hand.
- Each new hand is independent.

### Actions

The v0 action set supports hit, stand, and double. Split, surrender, and insurance are out of scope for the first version.

### Dealer Policy

Dealer hits until 17 and stands on both hard 17 and soft 17.

### Scoring

- Normal win: +1.
- Normal loss: -1.
- Push: 0.
- Double win or loss: +/-2.
- Blackjack: +1.5.

### Trajectory Content

A trajectory should record the player hand, dealer upcard, hidden dealer card when revealed, legal actions, agent decisions, card draws, terminal outcome, and hand profit. Future analysis can be performed from these trajectories without requiring metrics to be built into the first implementation.

## 3. Othello-lite 6x6

### Type

Perfect-information board game. Two players.

### Purpose

Othello-lite is a 6x6 version of Othello/Reversi. It is intended to collect trajectories involving legal move generation, board evaluation, mobility, corner and edge control, and long-horizon planning. It is more strategic than Connect Four but simpler than Chess or Go.

### Board

The game uses a 6x6 board. Rows are numbered 1 to 6 from top to bottom, and columns are labeled A to F from left to right.

### Initial Board

```
. . . . . .
. . . . . .
. . W B . .
. . B W . .
. . . . . .
. . . . . .
```

Black moves first.

### Rules

- A player places one piece on a legal empty cell.
- A legal move must flip at least one opponent piece.
- Flips can happen in eight directions: horizontal, vertical, and diagonal.
- If a player has no legal move, that player passes.
- If both players cannot move, the game ends.
- The player with more pieces at the end wins.

### Trajectory Content

A trajectory should record the full board state at each turn, current player, legal moves, chosen move, flipped pieces, pass events, final board, and winner.

## 4. Repeated Colonel Blotto

### Type

Strategic allocation game. Two players. Simultaneous actions.

### Purpose

Repeated Colonel Blotto tests resource allocation, numerical reasoning, opponent modeling, adaptation, memory, and risk diversification. The repeated format is preferred because it allows agents to observe and respond to opponent allocation patterns over time.

### Default Setup

- Rounds: 20.
- Resources per round: 100.
- Battlefields: 5.
- Battlefield values: [1, 2, 3, 4, 5].

### Round Flow

- Each agent secretly allocates 100 resources across 5 battlefields.
- Allocations must use non-negative integers and sum to 100.
- For each battlefield, the higher allocation wins that battlefield value.
- Ties give no score in the v0 version.
- Scores accumulate across rounds.
- The player with the higher cumulative score after all rounds wins.

### Trajectory Content

A trajectory should record round number, battlefield values, both submitted allocations, battlefield-level outcomes, round scores, cumulative scores, invalid allocation or fallback events, and final winner.

## Summary

| Game | Category | Main Purpose | Interaction |
| --- | --- | --- | --- |
| Leduc Poker | Imperfect information | Bluffing, betting, belief update | Agent-vs-agent |
| Independent Blackjack | Risk / probability | Basic strategy and risk calibration | Agent-vs-environment |
| Othello-lite 6x6 | Perfect information | Board planning and mobility | Agent-vs-agent |
| Repeated Colonel Blotto | Strategic allocation | Resource allocation and adaptation | Simultaneous agent-vs-agent |

These games should first be implemented as clean trajectory-collection environments. Later analysis can use the collected trajectories to derive outcome, validity, behavior, and diagnostic metrics.
