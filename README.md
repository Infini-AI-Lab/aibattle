# aibattle

**AI Battle Arena v0** — an evaluation-first multi-agent arena for adversarial
games between AI agents. Every participant is an *agent* (built-in baseline,
default model-backed wrapper, or external pipeline); every game is a modular
environment; a unified runner records structured JSONL logs for comparison.

The v0 reference game is **Kuhn Poker**. See `claude-doc/DESIGN.md` for the full
design.

## Installation

**Step 1: Create a conda environment**

```bash
conda create -n aibattle python=3.12 -y
conda activate aibattle
```

**Step 2: Install uv**

```bash
pip install -U "uv>=0.10"
```

**Step 3: Install aibattle**

```bash
uv pip install -e ".[dev]"      # editable install with all extras + pytest
```

Other dependency groups (use instead of `dev` for a lighter footprint):

```bash
uv pip install -e .             # core only (pyyaml)
uv pip install -e ".[model]"    # + openai client (OpenAI / Fireworks / vLLM)
uv pip install -e ".[anthropic]"  # + anthropic client
```

**Step 4: Verify**

```bash
aibattle run configs/random_vs_heuristic.yaml   # no API key needed
```

## Quick start

Run a match from a YAML config:

```bash
# Baseline calibration — no API key needed
aibattle run configs/random_vs_heuristic.yaml

# Model vs baseline via Fireworks (reads FIREWORKS_API_KEY, or a .fireworks file)
export FIREWORKS_API_KEY=$(cat .fireworks)
aibattle run configs/fireworks_vs_heuristic.yaml

# Recompute the summary from an existing log (resolves the latest run)
aibattle eval ./runs/random_vs_heuristic
```

Each run writes its outputs to a **unique per-run subdirectory** under the
configured `output.dir` — `run_<timestamp>_<rand>/` — so repeated or concurrent
runs can never overwrite or corrupt each other's logs:

```
runs/exp/
  run_20260530-204313_046ebe/
    match.jsonl        # full step-by-step log
    summary.json       # per-agent statistics
    trajectories.json  # (optional) all episodes, structured
    transcripts/       # (optional) one readable .txt per episode
  run_20260530-204821_979e29/
    ...
```

`aibattle eval <dir>` accepts a `match.jsonl` file, a single run directory, or a
parent directory (in which case it evaluates the most recent run).

### Saving trajectories

Two optional, human- and machine-friendly outputs are controlled in the
`output:` block of the config:

```yaml
output:
  dir: ./runs/exp
  save_trajectories: true     # -> trajectories.json : ALL episodes in one structured JSON file
  save_transcripts: true      # -> transcripts/episode_NNNN.txt : one plain-text file per episode
```

- **`trajectories.json`** — a single file containing every episode with its
  nested steps (observation, raw agent/model output, extracted action, result).
  Good for programmatic analysis and as a future training-data source.
- **`transcripts/`** — one readable `.txt` per episode showing, step by step,
  what each agent saw, the model's raw output, and the action that was
  extracted. Good for eyeballing how an agent actually played a hand.

## Concepts

| Layer | Responsibility |
|-------|----------------|
| **Game** (`games/`) | Pure, immutable rules: observations, legal actions, transitions, payoffs. Enforces hidden information. |
| **Agent** (`agents/`) | Receives an observation, returns an action. Builtin / model / external. |
| **Model** (`models/`) | Thin `str -> str` client. OpenAI-compatible (Fireworks) + Anthropic. |
| **Runner** (`runner/`) | Orchestrates the loop, seat-swapping, and the invalid-action policy. |
| **Logging** (`logging/`) | JSONL: one record per step + per episode. |
| **Eval** (`eval/`) | Pure function of logs → mean payoff/hand, win rate, invalid rate. |

Key invariants: the runner talks to **agents, not models**; agents never see
game state, only observations; matches are **seat-swapped** so reported skill is
position-neutral, with mean payoff per hand (chips) as the primary metric.
