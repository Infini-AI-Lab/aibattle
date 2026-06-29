<div align="center">

# 🎲 AI Battle Arena

**An evaluation-first arena for adversarial games between AI agents**

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Games](https://img.shields.io/badge/games-10-orange.svg)](#-games)
[![Status](https://img.shields.io/badge/status-v0-lightgrey.svg)](#)

</div>

---

AI Battle Arena is an **evaluation-first, multi-agent arena** where AI agents play
adversarial games head-to-head under one identical pipeline — so results are
directly comparable. Every participant is an *agent* (built-in baseline,
model-backed wrapper, or external pipeline); every game is a modular environment;
a unified runner records structured JSONL logs that turn into opponent-adjusted
rankings and interactive HTML leaderboards.

It is built around four ideas:

- **One pipeline, many games.** The same prompts, harness, and scoring run every
  model through **10 games** spanning perfect- and imperfect-information play.
- **Position-neutral, opponent-adjusted.** Matches are **round-robin and
  seat-swapped**, and models are rated by a **bootstrap Elo** so models that faced
  different opponents stay comparable — with honest error bars.
- **Logs that explain, not just score.** Every step is logged; the analyzers turn
  raw episodes into *why-win / why-lose* deep-dives (bluffing, gear-shifting,
  lead trajectories, pressure-vs-showdown, …), not just a number.
- **Agents, not models.** The runner talks to agents; agents only ever see
  observations, never game state — so hidden information stays hidden.

## 📰 News

- **v0** — initial public arena: 10 games, bootstrap-Elo rankings, auto-generated
  interactive reports, and a [metrics Q&amp;A](reports/qa.html)
  page that defines every metric in a plain-language Q&amp;A.

## 🚀 Getting started

**1. Create an environment**

```bash
conda create -n aibattle python=3.12 -y
conda activate aibattle
pip install -U "uv>=0.10"
```

**2. Install**

```bash
uv pip install -e ".[dev]"        # editable install + all extras + pytest
# lighter footprints:
uv pip install -e .               # core only
uv pip install -e ".[model]"      # + OpenAI-compatible client (OpenAI / Fireworks / vLLM)
uv pip install -e ".[anthropic]"  # + Anthropic client
```

**3. Run your first match** (no API key needed)

```bash
aibattle run configs/random_vs_heuristic.yaml
```

```bash
# model vs baseline via Fireworks
export FIREWORKS_API_KEY=$(cat .fireworks)
aibattle run configs/fireworks_vs_heuristic.yaml

# recompute the summary from an existing log
aibattle eval ./runs/random_vs_heuristic
```

Each run writes to a **unique per-run subdirectory** (`run_<timestamp>_<rand>/`)
so concurrent or repeated runs never overwrite each other:

```
runs/exp/run_20260530-204313_046ebe/
  match.jsonl        # full step-by-step log
  summary.json       # per-agent statistics
  trajectories.json  # (optional) all episodes, structured
  transcripts/       # (optional) one readable .txt per episode
```

Set a player's agent `type: human` to turn any game into an interactive terminal
match.

## 🎮 Games

| Game | Information | What it is |
|------|-------------|------------|
| **Kuhn Poker** | imperfect | 3-card, one-street poker — the v0 reference game and GTO sanity check |
| **Leduc Hold'em** | imperfect | 6-card, two-street poker |
| **Hold'em 1-Hand** | imperfect | heads-up Texas Hold'em, one hand per deal |
| **Hold'em Match** | imperfect | heads-up Hold'em, 30-hand carry-stack matches (win or bust) |
| **Hold'em Table** | imperfect | 5-handed ring game, ranked by finishing place |
| **Blackjack** | imperfect | vs the house dealer (no opponent) |
| **Connect Four** | perfect | solved 7×6 line-up game; scored on tactical accuracy |
| **Gomoku-Lite** | perfect | 9×9 five-in-a-row |
| **Othello-Lite** | perfect | reversi on a small board |
| **Colonel Blotto** | simultaneous | resource-allocation game |

## 🧩 Concepts

| Layer | Responsibility |
|-------|----------------|
| **Game** (`games/`) | Pure, immutable rules: observations, legal actions, transitions, payoffs. Enforces hidden information. |
| **Agent** (`agents/`) | Receives an observation, returns an action. Builtin / model / external. |
| **Model** (`models/`) | Thin `str → str` client. OpenAI-compatible (Fireworks) + Anthropic. |
| **Runner** (`runner/`) | Orchestrates the loop, seat-swapping, and the invalid-action policy. |
| **Logging** (`logging/`) | JSONL: one record per step + per episode. |
| **Eval** (`eval/`) | Pure function of logs → mean payoff/hand, win rate, invalid rate. |
| **Reports** (`scripts/`) | Turn episode logs into interactive HTML leaderboards + deep-dive analyses. |

**Key invariants:** the runner talks to **agents, not models**; agents never see
game state, only observations; matches are **seat-swapped** so reported skill is
position-neutral. See `claude-doc/DESIGN.md` for the full design.

## 🛠️ Build the report site

The site under `reports/` is generated. Rebuild it in dependency order with:

```bash
./build.sh
```

This renders every game report + the overview, builds the full replay data
(`runs/<game>/replays/…`, ~3.4 GB, gitignored), then curates the per-game
**featured replays** and extracts just those episodes into small, committed
copies under `reports/replays/<game>/` (~18 MB total). The replay viewers read
from `reports/replays/`, so the curated examples are self-contained in the repo.

**Deploy:** publish the `reports/` directory — that's it. The featured replays
ship inside it, so no large data or symlink is required. (The full
`runs/<game>/replays` tree is only a build-time input for the extraction and
for the build-only Table/Blackjack viewers.)

## 🗺️ Roadmap

- **Harness Arena** — open the same games to any model + any scaffolding, not just
  the single generic pipeline.
- More games and more imperfect-information depth.
- Pinned model-snapshot provenance on every report.
- Live, continuously updated public leaderboard.

## 📑 Citation

```bibtex
@misc{aibattle,
  title  = {AI Battle Arena},
  author = {},
  year   = {2026},
}
```

## 🙏 Acknowledgments

Reuses ideas and design from open-source agent-evaluation and poker-research
projects. Company logos shown in the reports are the trademarks of their
respective owners, used only to identify each model's maker.

## License

[MIT](LICENSE)
