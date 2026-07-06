<div align="center">

# 🎲 AI Battle Arena

**A game arena for evaluating how AI agents make decisions under competition**

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Games](https://img.shields.io/badge/games-10-orange.svg)](#-games)
[![Status](https://img.shields.io/badge/status-v0-lightgrey.svg)](#)

</div>

AI Battle Arena evaluates AI agents by making them play adversarial games against
live opponents. Instead of asking models to answer a fixed test set, the arena
creates fresh game states through interaction, logs every decision, and turns the
results into rankings, replays, and behavior analyses.

Every participant is an *agent*. An agent can be a built in baseline, a wrapper
around a model API, a local model, or a custom external system. Every game is a
modular environment with clear rules, legal actions, observations, and payoffs.
The shared runner keeps the evaluation comparable across games and agents.

It is built around four ideas:

- **Same setup, many games.** The same runner and scoring code cover **10 games**
  across perfect information, imperfect information, and simultaneous play.
- **Fair comparisons.** Matches are round robin and seat swapped where that
  matters, so the reported skill is not just a seating or first move advantage.
- **Logs that explain results.** Every step is logged. The analysis scripts can
  show how agents win or lose, including bluffing, lead changes, fold pressure,
  showdown results, tactical misses, and invalid actions.
- **Agents, not raw model calls.** The runner talks to agents. Agents see only
  observations, never hidden game state, so private information stays private.

## 📰 News

- **v0**: initial public arena with 10 games, a
  [live leaderboard](https://infini-ai-lab.github.io/aibattle/), generated game reports,
  [curated replays](https://infini-ai-lab.github.io/aibattle/replays.html),
  a [blog post](https://infini-ai-lab.github.io/aibattle/blog.html), and a
  [metrics Q&amp;A](https://infini-ai-lab.github.io/aibattle/qa.html)
  that explains the reported metrics in plain language.

## 🚀 Getting started

**1. Create an environment**

```bash
conda create -n aibattle python=3.12 -y
conda activate aibattle
pip install -U "uv>=0.10"
```

**2. Install the package**

```bash
uv pip install -e ".[dev]"        # editable install with all extras and pytest
# smaller installs:
uv pip install -e .               # core only
uv pip install -e ".[model]"      # OpenAI compatible client, including OpenAI, Fireworks, vLLM
uv pip install -e ".[anthropic]"  # Anthropic client
```

**3. Run your first match** (no API key needed)

```bash
aibattle run configs/random_vs_heuristic.yaml
```

```bash
# model against baseline via Fireworks
export FIREWORKS_API_KEY=$(cat .fireworks)
aibattle run configs/fireworks_vs_heuristic.yaml

# recompute the summary from an existing run
aibattle eval ./runs/random_vs_heuristic
```

Each run writes to a **unique run directory** (`run_<timestamp>_<rand>/`), so
concurrent or repeated runs do not overwrite each other:

```
runs/exp/run_20260530-204313_046ebe/
  match.jsonl        # full step by step log
  summary.json       # per agent statistics
  trajectories.json  # (optional) all episodes, structured
  transcripts/       # (optional) one readable .txt per episode
```

Set a player's agent `type: human` to play any game from the terminal.

## 🎮 Games

| Game | Information | What it is |
|------|-------------|------------|
| **Kuhn Poker** | imperfect | 3 card, one street poker and the v0 reference game |
| **Leduc Hold'em** | imperfect | 6 card, two street poker |
| **Hold'em 1 Hand** | imperfect | heads up Texas Hold'em, one hand per deal |
| **Hold'em Match** | imperfect | heads up Hold'em, 30 hand matches with carried stacks |
| **Hold'em Table** | imperfect | 5 handed ring game, ranked by finishing place |
| **Blackjack** | imperfect | vs the house dealer (no opponent) |
| **Connect Four** | perfect | solved 7×6 line up game, scored on tactical accuracy |
| **Gomoku Lite** | perfect | 9×9 five in a row |
| **Othello Lite** | perfect | reversi on a small board |
| **Colonel Blotto** | simultaneous | resource allocation game |

## 🧩 Concepts

| Layer | Responsibility |
|-------|----------------|
| **Game** (`games/`) | Rules for observations, legal actions, transitions, payoffs, and hidden information. |
| **Agent** (`agents/`) | Receives an observation and returns an action. Can be built in, model backed, human, or external. |
| **Model** (`models/`) | Thin text client for OpenAI compatible APIs and Anthropic. |
| **Runner** (`runner/`) | Orchestrates the game loop, seat swapping, retries, and invalid action handling. |
| **Logging** (`logging/`) | JSONL: one record per step + per episode. |
| **Eval** (`eval/`) | Turns logs into payoff, win rate, invalid rate, and related summary metrics. |
| **Reports** (`scripts/`) | Build HTML reports, leaderboards, replays, and behavior analyses from logs. |

**Key invariants:** the runner talks to **agents, not models**; agents see only
observations, not hidden game state; and matches are seat swapped when position
matters. See `claude-doc/DESIGN.md` for the full design.

## 🛠️ Build the report site

Most pages under `reports/` are generated from logs and analysis scripts. Rebuild
them in dependency order with:

```bash
./build.sh
```

This renders every game report and the overview, builds the full replay data
(`runs/<game>/replays/…`, about 3.4 GB and gitignored), then curates the
**featured replays** and extracts just those episodes into small committed copies
under `reports/replays/<game>/` (about 18 MB total). The replay viewers read from
`reports/replays/`, so the curated examples are self contained in the repo.

**Deploy:** publish the `reports/` directory. The featured replays ship inside
it, so no large data directory or symlink is required. The full
`runs/<game>/replays` tree is only needed while building the curated replay
copies and the build only Table and Blackjack viewers.

## 🗺️ Roadmap

- **Harness Arena:** open the same games to any model plus any scaffolding, not
  just the single generic pipeline.
- More games and deeper imperfect information settings.
- Pinned model snapshot provenance on every report.
- Live, continuously updated public leaderboard.

## 📑 Citation

```bibtex
@misc{aibattle,
  title  = {AI Battle Arena},
  author = {Zheng, Haizhong and Di, Yizhuo and Ruan, Letian and Jin, Shuowei and Chen, Beidi},
  year   = {2026},
}
```

## 🙏 Acknowledgments

Reuses ideas and design from open source agent evaluation and poker research
projects. Company logos shown in the reports are the trademarks of their
respective owners, used only to identify each model's maker.

## License

[MIT](LICENSE)
