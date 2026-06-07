# Plan: LocalAgent Reasoning Harnesses (CoT / Vote / Two-Stage / Self-Refine)

## Context

`aibattle` (AI Battle Arena) is an evaluation-first multi-agent game arena. The current `ModelAgent` in the agent layer is a "single-generation" LLM wrapper: `observation → GameTemplate → prompt → model → text → parse → action`.

Goal of this batch: build **a set of lightweight in-house reasoning harnesses / inference-time scaffolding** on top of `ModelAgent` — improving a single agent's decision quality through multiple LLM calls and structured intermediate steps. **Note**: this is NOT integrating external frameworks like LangChain/AutoGen (that was a rejected first interpretation); it is implementing prompt-engineering and multi-step reasoning capabilities inside the framework itself.

### The first four harnesses

1. **Structured CoT** — a single generation, but the model is forced to emit structured reasoning (state, immediate opportunities/threats, the opponent's likely plan) before giving the action. The lightest one and the basis for the other three.
2. **Self-Consistency (majority vote)** — run the same prompt N times at `temperature > 0` and take a majority vote over the actions. Trades compute for stability.
3. **Two-Stage: assess → decide** — Gen-1 assesses the opponent/situation → spliced into the prompt → Gen-2 makes the final decision.
4. **Self-Refine (self-critique)** — Gen-1 initial action + rationale → Gen-2 critique (is it the best option?) → Gen-3 revision.

### Academic support (verified online)

- **CoT**: [Chain-of-Thought Prompting Elicits Reasoning in LLMs (Wei et al., 2022, arXiv:2201.11903)](https://arxiv.org/abs/2201.11903) — intermediate reasoning steps significantly improve complex reasoning.
- **Self-Consistency**: [Self-Consistency Improves CoT Reasoning (Wang et al., 2022, arXiv:2203.11171)](https://arxiv.org/abs/2203.11171) — sample multiple reasoning paths then vote, GSM8K 56.5%→74.4%.
- **Two-stage / opponent-range estimation**: [How Far Are LLMs from Professional Poker Players? (arXiv:2602.00528, 2026)](https://arxiv.org/abs/2602.00528) decomposes poker into "act under hidden info, estimate opponent ranges, anticipate the future"; [PokerSkill (arXiv:2605.30094)](https://arxiv.org/html/2605.30094v1) reaches expert level with three-stage pure-prompt scaffolding, no training or solver required.
- **Self-Refine**: [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023, arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — the same LLM acts as generator/critic/reviser, ~20% average improvement (NeurIPS 2023).
- **(Next step) opponent modeling needs memory**: [Readable Minds: Emergent ToM in LLM Poker Agents (arXiv:2604.04157)](https://arxiv.org/abs/2604.04157) — LLMs only develop opponent models when equipped with persistent memory, confirming that "opponent modeling needs cross-hand state"; doing stateless first this round is the right call.

### Key decisions (confirmed with the user)

1. **Scope**: build all four harnesses; **stateless** (each `act()` is independent); cross-hand memory such as opponent modeling/reflection is deferred.
2. **Abstraction**: **generic harness + game-agnostic** — intermediate prompts use generic concatenation, and the final decision step reuses `GameTemplate.parse`; write once, works for all games (Kuhn/Holdem/Connect4/Gomoku).
3. **Code organization**: **shared base class + composable steps** (to make future composition easy, e.g. CoT + voting).
4. **Configurable**: N (number of votes), iteration rounds, temperature, custom intermediate prompt text, etc. are **overridable** in YAML (each with a sensible default).
5. **Keystone**: first extract `ModelAgent`'s render/generate/parse/repair loop into a shared helper, then have the harnesses reuse it.
6. **Config type**: add `type: local`, select a concrete harness via `harness: <name>`.

---

## Implementation

### Step 1 (keystone): extract shared primitives

**Create [src/aibattle/agents/template_loop.py](src/aibattle/agents/template_loop.py)** providing the primitives shared by the harnesses:

- `GenerateResult` (dataclass) — normalizes the output of one generation: `content` (for parse), `full_text` (for logging, may include thinking), `meta` (provider fields, merged into metadata).
- `async def run_template_loop(template, generate, request, *, max_retries=2) -> AgentResponse` — replicates verbatim the existing render→generate→parse→repair→INVALID logic and metadata assembly from [model_agent.py](src/aibattle/agents/model_agent.py). `generate: Callable[[str], Awaitable[GenerateResult]]`.
- A **vote/parse helper** `parse_or_none(template, text, request) -> Optional[Move]` (thin wrapper over `template.parse`), used by harnesses in intermediate steps.

**Refactor [src/aibattle/agents/model_agent.py](src/aibattle/agents/model_agent.py)**: `act()` maps `ModelOutput → GenerateResult` (keeping the `has_reasoning`/`finish_reason`/`truncated`/`completion_tokens`/`prompt_tokens` metadata keys unchanged) and `return await run_template_loop(...)`.

> **Risk (highest)**: logs/replay depend on `ModelAgent`'s metadata keys. The loop must emit identical keys. After implementing, run one game via `scripts/smoke_v2.py` and diff the before/after logs to confirm.

### Step 2: LocalAgent base class + composable steps

**Create [src/aibattle/agents/local/base.py](src/aibattle/agents/local/base.py)**:

`HarnessAgent(Agent)` (`agent_type = "local"`) — a shared base class holding `client: ModelClient`, `template: GameTemplate`, `name`, `max_retries`, and providing composable step primitives reused by every harness:

```python
class HarnessAgent(Agent):
    agent_type = "local"
    def __init__(self, *, client, template, name, max_retries=2, **harness_cfg): ...

    # Primitives (for subclasses to orchestrate):
    async def _generate(self, prompt: str) -> GenerateResult: ...      # call client, normalize
    def _final_prompt(self, request) -> str: ...                       # template.render_prompt
    def _parse(self, text, request) -> Optional[Move]: ...             # template.parse
    def _vote(self, moves: list[Move]) -> Move: ...                    # majority vote (tie -> first legal)
    def _compose(self, request, *, extra_context: str) -> str: ...     # generic mid-prompt concatenation

    # Abstract: subclass implements the orchestration, returns AgentResponse
    # (stashing intermediate artifacts into metadata)
    @abstractmethod
    async def act(self, request) -> AgentResponse: ...
```

Intermediate artifacts (assessment text / candidate list / critique) are stashed into `AgentResponse.metadata["harness"]`, so logs/replay can audit whether the harness actually helped the decision.

### Step 3: the four harness subclasses

Each in its own file, all `agent_type="local"`, reusing the Step 2 primitives + the Step 1 keystone:

- **[src/aibattle/agents/local/cot.py](src/aibattle/agents/local/cot.py) `StructuredCoTAgent`** — appends a generic, game-agnostic structured-reasoning instruction to `template.render_prompt` ("First analyze, item by item: the current state and your objective, the immediate opportunities and threats, the opponent's likely plan inferred from the visible history, and your legal options; then give the action on the last line"), a single generate, with `run_template_loop`'s parse/repair as the backstop. Param: `cot_instructions` (overridable).
- **[src/aibattle/agents/local/self_consistency.py](src/aibattle/agents/local/self_consistency.py) `SelfConsistencyAgent`** — run the same prompt `n` times concurrently (`temperature` configurable), parse each into a Move, `_vote` takes the majority; if all fail, fall back to a single + repair. Params: `n` (default 5), `temperature` (default 0.7). metadata records the vote distribution.
- **[src/aibattle/agents/local/two_stage.py](src/aibattle/agents/local/two_stage.py) `TwoStageAgent`** — Gen-1 uses a generic "assessment prompt" (default: "Based only on the public information and action history so far, assess the opponent's likely position, plan, pressure, or available threats, and briefly explain your reasoning") → `_compose` splices the assessment in → Gen-2 runs the final decision step + parse/repair. Param: `estimate_prompt` (overridable). metadata records the Gen-1 assessment text.
- **[src/aibattle/agents/local/self_refine.py](src/aibattle/agents/local/self_refine.py) `SelfRefineAgent`** — Gen-1 initial action + rationale → Gen-2 critique (generic "is this the best legal action, or is there a better one?" prompt) → Gen-3 revision + parse/repair. Params: `rounds` (default 1 critique round), `critique_prompt` (overridable). metadata records each round's draft/critique.

All intermediate prompts are **game-agnostic** (they only reference `observation.rendered` / `history` / `legal_actions`), and the final step always reuses `GameTemplate`, so they automatically apply to all games.

### Step 4: registry + loader + default params

- **Edit [src/aibattle/agents/registry.py](src/aibattle/agents/registry.py)**: add `_build_local_agent(cfg, game_name, seed)` selecting a class from the sub-registry `{cot, self_consistency, two_stage, self_refine}` by `cfg["harness"]`; construct via `make_client(cfg["model"])` + `make_template(game_name)`; pass `cfg.get("harness_args", {})` through to the harness. Add a `local` branch to `make_agent`.
- **Edit [src/aibattle/config/loader.py](src/aibattle/config/loader.py) (line 83)**: add `"local"` to the type whitelist, update the error message accordingly.
- Each harness's configurable params (`n`/`temperature`/`rounds`/`*_prompt`) have sensible defaults; YAML only overrides what it needs.

### Step 5: pyproject.toml

**Edit [pyproject.toml](pyproject.toml)**: add `pytest-asyncio>=0.23` to `dev`; add `[tool.pytest.ini_options]` (`asyncio_mode="auto"`, `testpaths=["tests"]`, `markers`). The harnesses reuse the existing `openai`/`anthropic` clients, so no new runtime dependency.

### Implementation order

1. `template_loop.py` + refactor `model_agent.py` → smoke-verify metadata unchanged.
2. `local/base.py` (`HarnessAgent` + primitives).
3. The four harness subclasses (cot → self_consistency → two_stage → self_refine).
4. registry `local` dispatch + loader whitelist.
5. `pyproject.toml` + pytest config.
6. TDD test suite.

### Example YAML

```yaml
players:
  player_0:
    agent:
      type: local
      harness: two_stage
      name: deepseek-twostage
      model: { provider: fireworks, model_id: accounts/fireworks/models/deepseek-v4-pro,
               api_key_env: FIREWORKS_API_KEY, temperature: 0.0, max_tokens: 16384 }
      harness_args: { estimate_prompt: "First assess the opponent's most likely position given the current action line" }
  player_1:
    agent:
      type: local
      harness: self_consistency
      model: { ... }
      harness_args: { n: 7, temperature: 0.8 }
```

---

## Testing (TDD — write tests first to pin the contract)

Create the `tests/` layout:

```
tests/
  conftest.py
  agents/
    test_template_loop.py     # shared keystone
    test_harness_cot.py
    test_harness_self_consistency.py
    test_harness_two_stage.py
    test_harness_self_refine.py
  config/
    test_registry_and_loader.py
  integration/
    test_runner_e2e.py        # end-to-end through the real Runner (offline)
```

### conftest.py shared fixtures

- **`make_request`** (factory) — builds a minimal `AgentRequest`+`Observation` without starting a game, with discrete/numeric variants, overridable `legal_actions`/`decision_seed`/`match`.
- **`FakeModelClient`** — subclasses `ModelClient`; `generate()` **returns a real `ModelOutput`** (strings auto-wrapped), records each `prompt`/`temperature`/`max_tokens` into `self.calls`, **supports scripting multiple outputs in call order** (needed for multi-step harnesses), and raises `AssertionError` when the script is exhausted (catches over-calling).
  > Key: `ModelAgent`/harnesses depend on `out.full_text()`, so the fake must return `ModelOutput`.
- **`real_kuhn`** — `make_game("kuhn_poker")` + `Runner` + `MatchLogger(None)` (zero file writes), offline end-to-end.

### Per-test focus

**`test_template_loop.py`** (fully offline, `FakeModelClient` + real `KuhnTemplate`/`HoldemTemplate`):
- First-try success → `attempts==1`, exactly one call and the prompt is the rendered result.
- Success after one repair → second prompt equals `repair_prompt(request, bad)`, `attempts==2`.
- Exhausted → `INVALID`, `attempts==max_retries+1`, `invalid==True`.
- Numeric amount parsing, missing-amount triggers repair; metadata passes through token/`truncated`/`has_reasoning`; `raw_output==full_text()`.
- **`ModelAgent` delegation consistency**: the real `ModelAgent` result == calling the helper directly (anti-regression anchor).

**`test_harness_cot.py`**:
- The rendered prompt contains the structured instruction; the final output parses to an action; garbage→repair→INVALID path works.
- `cot_instructions` override takes effect (custom text appears in the prompt sent to the client).

**`test_harness_self_consistency.py`**:
- Script `n` outputs (e.g. `["bet","bet","check","bet","check"]`) → vote yields `bet`; `metadata["harness"]` contains the vote distribution.
- `n`/`temperature` pass-through (assert `FakeModelClient.calls` was called n times with the right temperature).
- All unparseable → fall back to repair → finally INVALID.
- Tie (2 vs 2 vs ...) takes the first legal Move (deterministic, reproducible).

**`test_harness_two_stage.py`**:
- Two calls: the first prompt contains the "assessment" instruction, the second prompt **contains the assessment text from the first** (assert the concatenation); the final output parses to an action.
- `metadata["harness"]["estimate"]` records the Gen-1 text.
- `estimate_prompt` override takes effect.
- Gen-2 garbage→repair.

**`test_harness_self_refine.py`**:
- Correct three-step call order: draft→critique→revise; the third prompt contains the critique content (assert the concatenation).
- `rounds` controls the number of critique rounds (rounds=2 → call count increases accordingly).
- `metadata["harness"]` records each round's draft/critique.
- If the revision still doesn't parse → repair/INVALID backstop.

**`test_registry_and_loader.py`**:
- `make_agent({"type":"local","harness":"two_stage","model":{...}}, game_name="kuhn_poker")` constructs the corresponding harness; `harness_args` pass-through; `game_name` resolves to the correct template.
- Unknown `harness` name / missing `model` → clear error.
- `load_config` (temp YAML via `tmp_path`) accepts `type: local`, rejects unknown types, still requires `agent.type`.

**`test_runner_e2e.py`** (`@pytest.mark.integration`, offline):
- Drive a harness agent with `FakeModelClient` through a full Kuhn match (`MatchLogger(None)`, `episode_dir=None`) → correct `episodes` count, `failures==0`, zero-sum `returns`.
- When the harness returns `INVALID`, the `fallback` policy applies and the match completes.
- The harness is a drop-in `Agent`: plays through against builtin/random.

### Marker / CI

- **Default fast tier** (only `pytest-asyncio` added): template_loop, the four harnesses, registry/loader — all offline, using `FakeModelClient`.
- **`@pytest.mark.integration`**: runner e2e, offline but slightly slower.
- No test writes to the repo filesystem except the loader's use of `tmp_path`; every match uses `MatchLogger(None)`.

---

## Verification (end-to-end)

1. **Tests**: `uv pip install -e ".[dev]"` → `pytest` (all green, offline).
2. **metadata regression**: `python scripts/smoke_v2.py` (needs `.fireworks`) to confirm `ModelAgent`'s log metadata keys are unchanged after the refactor.
3. **Manual harness test**: write two `type: local` configs (e.g. `two_stage` vs baseline `model`), `aibattle run` a Kuhn/Holdem config, and check that `metadata["harness"]` in `trajectories.json` has intermediate-reasoning traces.
4. **Controlled experiment**: run a small tournament of the same model baseline (`model`) vs each harness (reusing the `scripts/*_tournament.py` pattern) and see whether win rate / mean payoff improves — validating the harnesses' actual effect.

---

## Key files

**New**:
- [src/aibattle/agents/template_loop.py](src/aibattle/agents/template_loop.py) — shared keystone
- [src/aibattle/agents/local/base.py](src/aibattle/agents/local/base.py) — `HarnessAgent` + composable primitives
- [src/aibattle/agents/local/cot.py](src/aibattle/agents/local/cot.py) / [self_consistency.py](src/aibattle/agents/local/self_consistency.py) / [two_stage.py](src/aibattle/agents/local/two_stage.py) / [self_refine.py](src/aibattle/agents/local/self_refine.py) — the four example harnesses
- [scripts/smoke_harness.py](scripts/smoke_harness.py) — run the four harnesses live (one hand each) and print their intermediate reasoning
- [configs/harness_cot_vs_self_consistency.yaml](configs/harness_cot_vs_self_consistency.yaml) — example config wiring two harnesses head-to-head

**Modified**:
- [src/aibattle/agents/model_agent.py](src/aibattle/agents/model_agent.py) — delegates to `run_template_loop`
- [src/aibattle/agents/registry.py](src/aibattle/agents/registry.py) — `local` dispatch (sub-registry + `harness_args` pass-through + `game_name`)
- [src/aibattle/config/loader.py](src/aibattle/config/loader.py) — add `local` to the type whitelist

---

## Experiment results (ablation: bare model vs each harness)

A controlled ablation was run with `scripts/harness_ablation.py`: the SAME model
(`gpt-oss-120b` via Fireworks) plays against itself, one seat bare (`type: model`)
and one seat wrapped in a harness, seats swapped to be position-neutral. Any
payoff difference therefore comes from the harness scaffolding alone. Numbers are
from the harness's perspective; a harness "beats" the bare model only when
`mean/hand > 0` **and** the 95% CI does not cross 0.

**Kuhn Poker (8 hands per matchup):**

| harness          | mean/hand | ±ci95 | win% | invalid% | verdict        |
|------------------|----------:|------:|-----:|---------:|----------------|
| cot              |    +0.000 | 0.741 |  50% |       0% | no sig. diff   |
| self_consistency |    +0.000 | 0.741 |  50% |       0% | no sig. diff   |
| two_stage        |    +0.000 | 1.171 |  50% |       0% | no sig. diff   |
| self_refine      |    −0.125 | 1.076 |  50% |       0% | no sig. diff   |

**Heads-Up Hold'em (6 hands per matchup):**

| harness          | mean/hand | ±ci95 | win% | invalid% | verdict        |
|------------------|----------:|------:|-----:|---------:|----------------|
| cot              |    +0.667 | 3.306 |  67% |       0% | no sig. diff   |
| self_consistency |    +0.000 | 1.753 |  50% |       0% | no sig. diff   |
| two_stage        |    +0.000 | 1.753 |  50% |       0% | no sig. diff   |
| self_refine      |    +0.333 | 3.843 |  67% |       0% | no sig. diff   |

### Findings

1. **No harness beats the bare model with statistical significance** in either
   game. Every CI crosses 0. (Hold'em's apparent +0.667 / 67% for cot is noise —
   the CI is ±3.3, far wider than the mean; Hold'em single-hand variance is huge.)
2. **The harnesses run cleanly on a complex game**: 0% invalid-action rate on
   Hold'em across all four harnesses confirms the prompt/parse/repair/numeric-bet
   paths are correct end-to-end — a genuine result, independent of skill effect.
3. **Structured CoT is essentially redundant for reasoning models.** Inspecting
   raw outputs, the *bare* model already emits a full chain-of-thought (it is a
   reasoning model and cites near-optimal strategy unprompted). CoT only adds
   structure/length to reasoning the model already does — it does not create it.
   The original CoT result (Wei 2022) assumed *non-reasoning* models; that premise
   no longer holds for 2026-era reasoning models. **The real value must come from
   what a single forward pass cannot give** (cross-hand memory, exact tool
   computation, enforced information structure) — see Follow-up.

### Caveats

- **Sample sizes are tiny** (6–8 hands) — these runs show *direction and that the
  harnesses operate*, not a verdict. Hold'em especially needs 50–100+ hands to
  narrow the CI enough to conclude anything about skill.
- **Kuhn is too simple** (a solved game with a known optimal policy), so a single
  forward pass already plays near-optimally and leaves no room for scaffolding.

Reproduce: pit a `type: local` harness against a `type: model` baseline of the
SAME model in a config (see `configs/harness_cot_vs_self_consistency.yaml` for the
shape) and `aibattle run` it, then `aibattle eval` the run dir. The throwaway
ablation driver used for the numbers above was removed during cleanup.
(Note: `kimi-k2p6` over-thinks trivial decisions — ~84 s and >4096 tokens for one
Kuhn move — so it is impractical here; use a lighter model such as `gpt-oss-120b`.)

---

## Follow-up (out of scope for this batch, already planned)

- **Cross-hand memory harnesses**: opponent modeling (track an opponent's bet/bluff frequency per state), post-game reflection (Reflexion-style lesson memory). Requires the agent to hold state across `act()` calls — the next architectural step, supported by [Readable Minds (arXiv:2604.04157)](https://arxiv.org/abs/2604.04157).
- **Tool augmentation**: wire in a deterministic equity calculator / pot-odds tool to compensate for LLMs' weak arithmetic (a killer app for poker).
- **Harness composition**: e.g. CoT + voting, two-stage + self-refine (the shared base class's composable steps already reserve room for this).
