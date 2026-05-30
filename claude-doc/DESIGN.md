# AI Battle Arena v0 — Concrete Design Plan

This document turns the high-level design into implementable specifications:
concrete interfaces, data shapes, module layout, the YAML schema, the log
format, and the policy decisions the high-level doc left open (invalid actions,
reproducibility, evaluation). It is the contract the v0 code is built against.

The high-level vision lives in the project design doc; this is the engineering
companion. Where the two disagree, this document wins for v0 implementation
details.

---

## 0. Decisions locked for v0

These are the questions the high-level doc raised but did not answer. They are
settled here so implementation can proceed without re-litigating them.

| # | Question | Decision |
|---|----------|----------|
| 1 | Agent call sync or async? | **`async def act(...)`**. Costs nothing for local agents; avoids a painful migration for model/remote agents. |
| 2 | What happens on an illegal/unparseable action? | **Retry policy → fallback**. Configurable: retry `max_retries` times, then apply `on_invalid_action` (`fallback`/`forfeit`). Default `fallback` to a deterministic legal action. Every invalid event is logged. |
| 3 | Is a config reproducible? | **Game dealing + agent ordering are seeded and reproducible. Model stochasticity is logged but NOT reproducible.** Stated explicitly so nobody expects bit-identical LLM runs. |
| 4 | How are 2-player zero-sum games scored fairly? | **Seat-swapped episodes**: each pair plays both seats on the same dealt cards. Primary metric is **mean payoff per hand (in chips)**, not win rate. |
| 5 | Log format? | **JSONL**: one record per step + one summary record per episode. Debuggable now, trivially convertible to trajectories later. |
| 6 | Provider abstraction? | A thin `ModelClient.generate(...)` interface. Provider quirks stay inside client adapters and never leak into the agent template. |

---

## 1. Module layout

```
src/aibattle/
  __init__.py
  types.py              # core dataclasses: Observation, Action, AgentRequest, AgentResponse, StepRecord, ...
  games/
    __init__.py
    base.py             # Game protocol/ABC
    kuhn.py             # KuhnPoker implementation
    registry.py         # name -> Game class
  agents/
    __init__.py
    base.py             # Agent protocol/ABC
    random_agent.py     # built-in baseline
    heuristic_agent.py  # Kuhn-specific baseline
    model_agent.py      # default model-backed agent (generic wrapper + game template)
    templates/
      kuhn.py           # Kuhn prompt template + output parser
    registry.py         # name -> Agent factory
  models/
    __init__.py
    base.py             # ModelClient protocol
    openai_client.py    # also covers OpenAI-compatible endpoints (Fireworks, local)
    anthropic_client.py
  runner/
    __init__.py
    runner.py           # match/episode loop
  logging/
    __init__.py
    logger.py           # JSONL writer
    schema.py           # record schemas / validation
  eval/
    __init__.py
    evaluator.py        # reads logs -> summary statistics
  config/
    __init__.py
    schema.py           # YAML config dataclasses + validation
    loader.py           # load + resolve env vars
  cli.py                # thin wrapper: `aibattle run config.yaml`
```

Dependency direction (no cycles): `runner` depends on `games`, `agents`,
`logging`. `agents/model_agent` depends on `models`. Nothing depends on
`runner` except `cli`. `eval` depends only on log files, never on live objects.

---

## 2. Core types (`types.py`)

All cross-layer data is plain, serializable dataclasses. No layer passes live
game-state objects across the agent boundary.

```python
PlayerId = str            # e.g. "player_0", "player_1"
Action = str              # game-defined token, e.g. "check", "bet", "call", "fold"

@dataclass(frozen=True)
class Observation:
    player: PlayerId
    private: dict          # info only this player sees, e.g. {"card": "K"}
    public: dict           # info all players see, e.g. {"pot": 2}
    history: list[dict]    # public action log: [{"player","action"}, ...]
    legal_actions: list[Action]
    rendered: str          # human/agent-facing text rendering of the above

@dataclass(frozen=True)
class AgentRequest:
    game: str              # "kuhn_poker"
    game_version: str
    player: PlayerId
    observation: Observation
    instructions: str      # output-format expectations for model agents
    step_index: int

@dataclass
class AgentResponse:
    action: Action                  # REQUIRED — the only field the runner needs
    message: str | None = None      # optional natural-language rationale
    raw_output: str | None = None   # optional unparsed model output
    metadata: dict = field(default_factory=dict)  # latency, tokens, retries, etc.
```

`Observation.rendered` exists so model agents don't each reinvent
serialization, and so logs are human-readable. The structured fields exist so
non-LLM agents and the evaluator never have to parse text.

---

## 3. Game Layer (`games/base.py`)

A `Game` is a **pure, immutable-state** module. `step` returns a new state; it
never mutates. State is opaque to everyone except the game itself.

```python
class Game(ABC):
    name: str
    version: str
    players: list[PlayerId]          # fixed roster for v0

    @abstractmethod
    def initial_state(self, rng: random.Random) -> State: ...

    @abstractmethod
    def current_player(self, state: State) -> PlayerId: ...

    @abstractmethod
    def observation(self, state: State, player: PlayerId) -> Observation: ...
    # MUST expose only what `player` is allowed to see — enforces hidden info.

    @abstractmethod
    def legal_actions(self, state: State, player: PlayerId) -> list[Action]: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State: ...
    # Precondition: action in legal_actions(state, current_player(state)).
    # Caller (runner) guarantees this; game may assert.

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    @abstractmethod
    def returns(self, state: State) -> dict[PlayerId, float]: ...
    # Terminal payoffs (chips). Zero-sum for Kuhn: sum == 0.

    @abstractmethod
    def render(self, state: State, *, perspective: PlayerId | None = None) -> str: ...
```

Key invariant: **the runner is the only caller of `step`, and it only calls it
with a validated legal action.** Agents never receive `State`.

### 3.1 Kuhn Poker (`games/kuhn.py`)

Standard 2-player Kuhn Poker:
- Deck: `{J, Q, K}` (ranked J < Q < K). Each player dealt one card; one card unused.
- Ante: each player puts in 1 chip (pot starts at 2).
- Action sequences (P0 acts first):
  - `check, check` → showdown, pot 2.
  - `check, bet, fold` → bettor wins pot 2 (bettor +1).
  - `check, bet, call` → showdown, pot 4 (winner +2).
  - `bet, fold` → bettor wins pot 2 (bettor +1).
  - `bet, call` → showdown, pot 4 (winner +2).
- Action tokens: `check`, `bet`, `call`, `fold`. `legal_actions` returns the
  subset valid at the current node.
- `returns`: winner gains the loser's net contribution; zero-sum.
- `version = "1.0.0"`.

State is a small frozen dataclass: `(cards: dict, history: tuple, to_act: PlayerId)`.

---

## 4. Agent Layer (`agents/base.py`)

```python
class Agent(ABC):
    name: str          # instance label, e.g. "gpt-4o-default"
    agent_type: str    # "builtin" | "model" | "external"

    @abstractmethod
    async def act(self, request: AgentRequest) -> AgentResponse: ...
    # MUST return an action. May return an illegal one — the runner handles that.
```

Agents are stateless across episodes by default. If an external agent keeps
memory, that is its own business; the arena does not manage it.

### 4.1 Built-in baselines
- **`RandomAgent`**: uniform over `legal_actions`, seeded from the request via
  an injected `random.Random` (so baseline behavior is reproducible).
- **`KuhnHeuristicAgent`**: simple rule-based policy (e.g. always bet/call with
  K, fold Q to a bet some fraction, etc.) — a non-trivial calibration baseline.

### 4.2 Default model-backed agent (`agents/model_agent.py`)
A generic wrapper = `ModelClient` + a **game-specific template**.

```python
class ModelAgent(Agent):
    agent_type = "model"
    def __init__(self, client: ModelClient, template: GameTemplate, *, max_retries=2):
        ...
    async def act(self, request) -> AgentResponse:
        prompt = self.template.render_prompt(request)
        for attempt in range(self.max_retries + 1):
            raw = await self.client.generate(prompt)
            action = self.template.parse(raw, request.observation.legal_actions)
            if action is not None:
                return AgentResponse(action=action, raw_output=raw,
                                     metadata={"attempts": attempt + 1})
            prompt = self.template.repair_prompt(request, raw)  # nudge toward format
        # exhausted retries → return last raw with no valid action; runner decides.
        return AgentResponse(action=INVALID, raw_output=raw,
                             metadata={"attempts": self.max_retries + 1, "invalid": True})
```

`GameTemplate` (per game, `agents/templates/kuhn.py`):
- `render_prompt(request)` → rules + private card + public history + legal
  actions + strict output format ("Respond with exactly one of: check, bet").
- `parse(raw, legal_actions)` → `Action | None` (tolerant: case-insensitive,
  extracts the action token, rejects anything not in `legal_actions`).
- `repair_prompt(request, bad_output)` → a reminder appended for the retry.

### 4.3 External custom agents
Two supported integration modes for v0:
1. **In-process**: a Python class implementing `Agent`, referenced by import
   path in YAML (`module:ClassName`).
2. **HTTP**: `HttpAgent` POSTs the `AgentRequest` (JSON) to a user endpoint and
   reads back an `AgentResponse`. The protocol is exactly the dataclasses above,
   JSON-serialized. This is the documented contract for remote pipelines.

---

## 5. Model Layer (`models/base.py`)

```python
class ModelClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str | list[dict], *,
                       temperature: float = 0.0,
                       max_tokens: int = 256) -> str: ...
```

- `OpenAIClient` covers OpenAI and any OpenAI-compatible endpoint (Fireworks,
  vLLM, local) via a configurable `base_url`. The repo already has a
  `.fireworks` config file, so Fireworks-via-OpenAI-compat is the near-term path.
- `AnthropicClient` for Claude.
- API keys come from env vars referenced in YAML, never inlined.
- Provider-specific request/response shaping stays inside the client. The
  agent template sees only `str in → str out`.

---

## 6. Runner Layer (`runner/runner.py`)

One match = one or more episodes. Episode loop:

```
state = game.initial_state(rng)
logger.episode_start(meta)
step_index = 0
while not game.is_terminal(state):
    p = game.current_player(state)
    obs = game.observation(state, p)
    request = build_request(game, p, obs, step_index)
    response = await agents[p].act(request)
    action, invalid_info = resolve_action(response, obs.legal_actions, policy)
    logger.step(StepRecord(step_index, p, obs, response, action, invalid_info))
    state = game.step(state, action)
    step_index += 1
returns = game.returns(state)
logger.episode_end(returns, winner, length)
```

`resolve_action` implements the **invalid-action policy** (Decision #2):
- If `response.action` is legal → use it.
- Else, per config `on_invalid_action`:
  - `fallback` (default): pick a deterministic legal action (first in a fixed
    priority order, e.g. `fold > check > call > bet`), record an invalid event.
  - `forfeit`: end the episode, award the opponent the win, record forfeit.
- The model agent already retried internally; the runner-level policy is the
  final backstop after retries are exhausted.

### 6.1 Seat-swapping (Decision #4)
The runner supports `seat_swap: true`. When enabled, episodes are run in pairs:
the same RNG seed deals identical cards, and the two agents swap `player_0` /
`player_1`. This isolates skill from positional/dealing luck. Logs tag each
episode with `pair_id` and `seat_assignment`.

The runner knows **nothing** about strategy, model providers, or agent
internals. It only orchestrates the protocol.

---

## 7. Logging Layer (`logging/`)

One JSONL file per match (configurable directory). Record types share a
`record_type` discriminator.

**Step record:**
```json
{"record_type":"step","episode":3,"pair_id":1,"step":2,"player":"player_0",
 "observation":{"private":{"card":"Q"},"public":{"pot":2},
   "history":[{"player":"player_0","action":"check"}],
   "legal_actions":["check","bet"],"rendered":"You hold Q. Pot is 2..."},
 "response":{"action":"bet","message":"Bluffing a weak hand",
   "raw_output":"I'll bet.","metadata":{"attempts":1,"latency_ms":640}},
 "selected_action":"bet","invalid":false}
```

**Episode summary record:**
```json
{"record_type":"episode","episode":3,"pair_id":1,"seat_assignment":{"player_0":"agentA","player_1":"agentB"},
 "returns":{"player_0":2,"player_1":-2},"winner":"player_0","length":3,
 "invalid_count":{"player_0":0,"player_1":0},"seed":12345}
```

**Match header record** (first line): game name/version, agent identities and
types, full resolved config, seed, timestamp.

JSONL is chosen so logs stream as they're produced, are greppable, and convert
to RL trajectories later without a schema migration.

---

## 8. Evaluation Layer (`eval/evaluator.py`)

Pure function of the log file(s) → summary. Never touches live objects.

Metrics (per agent, aggregated across episodes and seat assignments):
- Episodes played.
- **Mean payoff per hand (chips)** — the primary signal for zero-sum Kuhn.
- 95% confidence interval on mean payoff (episodes are noisy; report variance).
- Win / loss / tie rate (secondary).
- **Invalid action rate** (invalid steps / total steps).
- Average episode length.
- Action distribution (how often each action chosen) — useful for spotting
  degenerate or always-fold model behavior.

Seat-swapped pairs are aggregated so reported skill is position-neutral.

Output: a JSON summary + a printed table.

---

## 9. YAML Configuration (`config/`)

```yaml
game:
  name: kuhn_poker
  version: "1.0.0"        # optional; pinned for reproducibility
  params: {}              # game-specific knobs (none for Kuhn v0)

players:
  player_0:
    agent:
      type: model         # builtin | model | external
      name: gpt4o
      model:
        provider: openai  # openai | anthropic | (openai-compatible via base_url)
        model_id: gpt-4o
        base_url: null     # set for Fireworks/local
        api_key_env: OPENAI_API_KEY
        temperature: 0.0
        max_tokens: 256
      max_retries: 2
  player_1:
    agent:
      type: builtin
      name: kuhn_heuristic

run:
  episodes: 200
  seed: 12345
  seat_swap: true         # play each pair in both seats
  on_invalid_action: fallback   # fallback | forfeit
  concurrency: 1          # episodes run sequentially in v0; field reserved

output:
  dir: ./runs/exp1
  save_full_log: true
  save_summary: true
```

External agent examples:
```yaml
# in-process
agent: {type: external, name: my_agent, entrypoint: "mypkg.agents:MyAgent"}
# remote HTTP
agent: {type: external, name: my_agent, http: {url: "http://localhost:8080/act", timeout_s: 30}}
```

Loader resolves `*_env` references against the environment, validates against
dataclasses in `config/schema.py`, and fails fast with clear messages on
unknown game/agent names or missing keys.

CLI is a thin wrapper:
```
aibattle run config.yaml          # run a match
aibattle eval ./runs/exp1         # (re)compute summary from logs
```

---

## 10. Mapping to the high-level success criteria

| Criterion | Where satisfied |
|-----------|-----------------|
| Define Kuhn experiment in YAML | §9 |
| Run one or many episodes | §6 runner loop, `run.episodes` |
| Built-in baseline player | §4.1 |
| Default model-backed player | §4.2 + §5 |
| External custom player | §4.3 (in-process + HTTP) |
| Runner agnostic to agent internals | §6 — only the protocol crosses the boundary |
| Game enforces legal actions & outcomes | §3 `legal_actions`/`step`/`returns` + §6 `resolve_action` |
| Structured logs per match | §7 JSONL |
| Basic evaluation summaries | §8 |
| Architecture stays extensible | §1 layout, no reverse dependencies, JSONL → trajectory path |

---

## 11. Build order (suggested)

1. `types.py` — freeze the dataclasses (the whole system keys off these).
2. `games/kuhn.py` + `games/base.py` — testable in isolation against known
   Kuhn payoffs.
3. `agents/random_agent.py` + `runner.py` + `logging` — get random-vs-random
   episodes producing valid logs. **This is the first end-to-end milestone.**
4. `eval/evaluator.py` — summarize those logs.
5. `agents/heuristic_agent.py` — calibration baseline; sanity-check eval numbers.
6. `models/` + `agents/model_agent.py` + `templates/kuhn.py` — first model run.
7. `config/` + `cli.py` — wire YAML over the top.
8. `agents/HttpAgent` — external-agent contract last (least core risk).

Each step is independently testable; the framework is exercisable end-to-end
after step 3 with zero model dependencies.

---

## 12. Open questions deferred past v0 (explicitly out of scope)

- Multi-player (>2) games and non-zero-sum payoffs.
- Concurrency / parallel episode execution (`run.concurrency` reserved but =1).
- Trajectory/RL export format (JSONL is forward-compatible; no exporter yet).
- Caching model responses for partial reproducibility.
- Tournament scheduling and rating systems.
