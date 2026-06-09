# AI Battle Arena — New Games Four-Model Experiment

Models: kimi-k2p6, deepseek-v4-pro, glm-5p1, gpt-oss-120b  
Unavailable ids (minimax-m2p7, deepseek-flash) are out of scope.

## independent_blackjack
_coverage: 4/4 models — COMPLETE (structure: independent_vs_dealer)_
| model | hands | profit | mean_per_hand | win_rate | loss_rate | push_rate | invalid_rate |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 2 | 1.0 | 0.5 | 0.5 | 0.0 | 0.5 | 0.0 |
| kimi-k2p6 | 2 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 |
| glm-5p1 | 2 | -1.0 | -0.5 | 0.0 | 0.5 | 0.5 | 0.0 |
| deepseek-v4-pro | 2 | -2.0 | -1.0 | 0.0 | 1.0 | 0.0 | 0.0 |

## leduc_poker
_coverage: 6/6 model pairs, 6/6 seat-swapped pairs, 4/4 models — COMPLETE (structure: round_robin_seat_swap)_
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| gpt-oss-120b | 6 | 0.333 | 1.833 | 0.0 |
| glm-5p1 | 6 | 0.5 | 0.333 | 0.0 |
| deepseek-v4-pro | 6 | 0.5 | 0.167 | 0.0 |
| kimi-k2p6 | 6 | 0.5 | -2.333 | 0.0 |

## repeated_colonel_blotto
_coverage: 6/6 model pairs, 0/6 seat-swapped pairs, 4/4 models — PARTIAL (structure: round_robin_seat_swap)_
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| gpt-oss-120b | 3 | 0.667 | 0.333 | 0.0 |
| kimi-k2p6 | 3 | 0.667 | 0.333 | 0.0667 |
| glm-5p1 | 3 | 0.333 | -0.333 | 0.4 |
| deepseek-v4-pro | 3 | 0.333 | -0.333 | 0.0833 |

## othello_lite_6x6
_coverage: 3/6 model pairs, 0/6 seat-swapped pairs, 4/4 models — PARTIAL (structure: round_robin_seat_swap)_
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| deepseek-v4-pro | 1 | 1.0 | 1.0 | 0.0 |
| gpt-oss-120b | 3 | 0.667 | 0.333 | 0.0 |
| kimi-k2p6 | 1 | 0.0 | -1.0 | 0.0 |
| glm-5p1 | 1 | 0.0 | -1.0 | 0.0 |
