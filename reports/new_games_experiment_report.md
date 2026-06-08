# AI Battle Arena — New Games Four-Model Experiment

Models: kimi-k2p6, deepseek-v4-pro, glm-5p1, gpt-oss-120b  
Unavailable ids (minimax-m2p7, deepseek-flash) are out of scope.

## leduc_poker
| model | games | win_rate | net_per_game | invalid_rate |
|---|---|---|---|---|
| gpt-oss-120b | 6 | 0.333 | 1.833 | 0.0 |
| glm-5p1 | 6 | 0.5 | 0.333 | 0.0 |
| deepseek-v4-pro | 6 | 0.5 | 0.167 | 0.0 |
| kimi-k2p6 | 6 | 0.5 | -2.333 | 0.0 |

## independent_blackjack
| model | hands | profit | mean_per_hand | win_rate | loss_rate | push_rate | invalid_rate |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 2 | 1.0 | 0.5 | 0.5 | 0.0 | 0.5 | 0.0 |
| kimi-k2p6 | 2 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 |
| glm-5p1 | 2 | -1.0 | -0.5 | 0.0 | 0.5 | 0.5 | 0.0 |
| deepseek-v4-pro | 2 | -2.0 | -1.0 | 0.0 | 1.0 | 0.0 | 0.0 |
