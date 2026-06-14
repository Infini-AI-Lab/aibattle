# AI Battle Arena — New Games Four-Model Experiment

Models: kimi-k2p6, deepseek-v4-pro, glm-5p1, minimax-m2p7, gpt-oss-120b, minimax-m3  
Unavailable ids (deepseek-flash) are out of scope.

## independent_blackjack
_coverage: 6/6 models — COMPLETE (structure: independent_vs_dealer)_
| model | hands | profit | mean_per_hand | win_rate | loss_rate | push_rate | invalid_rate |
|---|---|---|---|---|---|---|---|
| glm-5p1 | 100 | 14.0 | 0.14 | 0.49 | 0.43 | 0.08 | 0.0 |
| kimi-k2p6 | 100 | 7.5 | 0.075 | 0.47 | 0.42 | 0.11 | 0.0 |
| gpt-oss-120b | 100 | 5.5 | 0.055 | 0.47 | 0.44 | 0.09 | 0.0 |
| minimax-m2p7 | 100 | 2.5 | 0.025 | 0.46 | 0.45 | 0.09 | 0.0 |
| minimax-m3 | 100 | 1.5 | 0.015 | 0.46 | 0.47 | 0.07 | 0.0 |
| deepseek-v4-pro | 100 | -2.0 | -0.02 | 0.4 | 0.45 | 0.15 | 0.0 |

## leduc_poker
_coverage: 15/15 model pairs, 15/15 seat-swapped pairs, 6/6 models — COMPLETE (structure: round_robin_seat_swap)_
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| deepseek-v4-pro | 250 | 0.392 | 0.12 | 0.0 |
| glm-5p1 | 250 | 0.384 | 0.084 | 0.0 |
| kimi-k2p6 | 250 | 0.412 | 0.08 | 0.0 |
| gpt-oss-120b | 250 | 0.464 | -0.016 | 0.0 |
| minimax-m3 | 250 | 0.436 | -0.088 | 0.0 |
| minimax-m2p7 | 250 | 0.416 | -0.18 | 0.0 |

## repeated_colonel_blotto
_coverage: 15/15 model pairs, 15/15 seat-swapped pairs, 6/6 models — COMPLETE (structure: round_robin_seat_swap)_
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| deepseek-v4-pro | 87 | 0.793 | 0.609 | 0.0 |
| kimi-k2p6 | 90 | 0.722 | 0.467 | 0.0 |
| glm-5p1 | 90 | 0.667 | 0.356 | 0.0 |
| minimax-m3 | 37 | 0.351 | -0.27 | 0.0 |
| gpt-oss-120b | 84 | 0.179 | -0.643 | 0.0 |
| minimax-m2p7 | 86 | 0.128 | -0.733 | 0.0 |

## othello_lite_6x6
_coverage: 10/15 model pairs, 10/15 seat-swapped pairs, 5/6 models — PARTIAL (structure: round_robin_seat_swap)_
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| kimi-k2p6 | 200 | 0.705 | 0.44 | 0.0 |
| deepseek-v4-pro | 200 | 0.66 | 0.355 | 0.0 |
| glm-5p1 | 200 | 0.46 | -0.07 | 0.0 |
| gpt-oss-120b | 200 | 0.43 | -0.1 | 0.0 |
| minimax-m2p7 | 200 | 0.17 | -0.625 | 0.0 |
