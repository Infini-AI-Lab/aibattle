# GPT/Claude Coached Tournament Summary

## Experiment Setting

- Date completed: 2026-06-08 UTC
- Output directory: `../aibattle-logs/gpt_claude_coached_tournament`
- Prompt template: coached
- Response format: action-only system instruction
- Reasoning effort: `medium`
- Output token cap: `8192`
- Max concurrency: `256`
- Seat swapping: enabled
- Error files: `0`

## Models

| Model | Model identifier | Region |
| --- | --- | --- |
| Claude Opus 4.8 | `us.anthropic.claude-opus-4-8` | `us-west-2` |
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | `us-east-1` |
| GPT-5.5 | `openai.gpt-5.5` | `us-east-2` |
| GPT-5.4 | `openai.gpt-5.4` | `us-east-2` |

## Games

| Game | Episodes per pair | Pairs | Total episodes | Notes |
| --- | ---: | ---: | ---: | --- |
| Connect Four | 50 | 6 | 300 | `random_open=2` |
| Gomoku-Lite | 50 | 6 | 300 | `random_open=2` |
| Hold'em 1-Hand Mode | 100 | 6 | 600 | `starting_stack=200` |
| Hold'em Match Mode | 20 | 6 | 120 | `starting_stack=200`, `max_hands=30` |

Total episodes: `1320`.

## Completion Audit

| Game | Episodes | Pair distribution | Error files |
| --- | ---: | --- | ---: |
| Connect Four | 300/300 | 50, 50, 50, 50, 50, 50 | 0 |
| Gomoku-Lite | 300/300 | 50, 50, 50, 50, 50, 50 | 0 |
| Hold'em 1-Hand Mode | 600/600 | 100, 100, 100, 100, 100, 100 | 0 |
| Hold'em Match Mode | 120/120 | 20, 20, 20, 20, 20, 20 | 0 |

Quality counters:

| Counter | Count |
| --- | ---: |
| Invalid actions | 320 |
| Incomplete response steps | 47 |
| Truncated response steps | 165 |

## Overall Results

Raw return is additive within each game, but game scales differ. The overall raw return table is useful as a run summary; per-game tables are the better comparison surface.

| Rank | Model | Wins | Raw return | Invalid actions |
| ---: | --- | ---: | ---: | ---: |
| 1 | GPT-5.5 | 386 | 440.0 | 17 |
| 2 | GPT-5.4 | 353 | 1.0 | 44 |
| 3 | Claude Opus 4.8 | 296 | -211.0 | 50 |
| 4 | Claude Sonnet 4.6 | 250 | -230.0 | 209 |

Total draws: 35.

## Results by Game

### Connect Four

| Rank | Model | Wins | Raw return | Invalid actions |
| ---: | --- | ---: | ---: | ---: |
| 1 | Claude Opus 4.8 | 93 | 45.0 | 34 |
| 2 | GPT-5.4 | 81 | 17.0 | 33 |
| 3 | GPT-5.5 | 69 | -5.0 | 2 |
| 4 | Claude Sonnet 4.6 | 43 | -57.0 | 56 |

Draws: 14.

### Gomoku-Lite

| Rank | Model | Wins | Raw return | Invalid actions |
| ---: | --- | ---: | ---: | ---: |
| 1 | GPT-5.5 | 99 | 55.0 | 9 |
| 2 | GPT-5.4 | 69 | -9.0 | 11 |
| 3 | Claude Opus 4.8 | 62 | -20.0 | 16 |
| 4 | Claude Sonnet 4.6 | 61 | -26.0 | 153 |

Draws: 9.

### Hold'em 1-Hand Mode

| Rank | Model | Wins | Raw return | Invalid actions |
| ---: | --- | ---: | ---: | ---: |
| 1 | GPT-5.5 | 180 | 374.0 | 1 |
| 2 | GPT-5.4 | 169 | -15.0 | 0 |
| 3 | Claude Sonnet 4.6 | 118 | -143.0 | 0 |
| 4 | Claude Opus 4.8 | 121 | -216.0 | 0 |

Draws: 12.

### Hold'em Match Mode

| Rank | Model | Wins | Raw return | Invalid actions |
| ---: | --- | ---: | ---: | ---: |
| 1 | GPT-5.5 | 38 | 16.0 | 5 |
| 2 | GPT-5.4 | 34 | 8.0 | 0 |
| 3 | Claude Sonnet 4.6 | 28 | -4.0 | 0 |
| 4 | Claude Opus 4.8 | 20 | -20.0 | 0 |

Draws: 0.

## Main Takeaways

- GPT-5.5 had the strongest overall raw return and led three of the four games by raw return.
- Claude Opus 4.8 led Connect Four.
- GPT-5.4 was consistently competitive, ranking second overall and second in three of the four per-game tables.
- Claude Sonnet 4.6 had the highest invalid-action count, mostly from Gomoku-Lite and Connect Four.
- The run completed without failed episode files, so all requested scheduled games are present in the final dataset.
