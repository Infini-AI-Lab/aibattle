# Remote Agent Design

This document defines the high-level design for Remote Agents in AI Battle
Arena. It focuses on the role, boundaries, and interaction contract, not on a
large implementation plan.

## 1. Purpose

AI Battle Arena should support two broad ways to bring agents into the arena:

1. **Local agents**: Python implementations that run inside the same process as
   the framework.
2. **Remote agents**: external decision services that expose an agent API and
   return game actions.

A Remote Agent lets users participate without putting their agent code inside
this repository or even using Python. This is important for private pipelines,
company-internal agents, RL services, multi-step planners, or agents that need
their own infrastructure.

The core promise is:

> A Remote Agent receives a standardized, player-scoped decision request and
> returns a game action under the AI Battle Arena agent protocol.

## 2. Key Boundary: Remote Agent vs Model Agent

Remote Agent and Model Agent are different abstractions.

### ModelAgent

`ModelAgent` is the framework-controlled LLM wrapper.

Its semantic flow is:

```text
Observation -> GameTemplate -> prompt -> model provider -> text -> parser -> action
```

OpenAI, Fireworks, Anthropic, and OpenAI-compatible local servers belong on this
path. They are model provider APIs. They return text, not game actions.

This path is useful for official model benchmarks because AI Battle Arena can
control the prompt, parser, retry behavior, and comparison setup.

### RemoteAgent

`RemoteAgent` is an externally controlled decision-maker.

Its semantic flow is:

```text
AgentRequest -> external service -> AgentResponse
```

The remote service already acts as an agent. It may internally call Fireworks,
OpenAI, a local model, a planner, an RL policy, or anything else. AI Battle
Arena does not inspect that implementation.

The key distinction is not whether the call is remote. The key distinction is
API semantics:

```text
Model provider API: prompt -> text
Remote agent API:  observation -> action
```

Therefore Fireworks and OpenAI should remain under `ModelAgent`, not
`RemoteAgent`.

## 3. Runtime Shape

Remote Agent should remain an implementation of the normal `Agent` interface,
not a separate runner path.

```text
Runner
  -> Agent.act(request)
  -> AgentResponse
  -> Game.validate_action(...)
  -> Runner invalid-action policy if needed
```

For a remote implementation:

```text
Runner
  -> RemoteAgent.act(request)
      -> serialize AgentRequest
      -> POST to endpoint
      -> parse response JSON
      -> return AgentResponse
  -> Runner validates action
```

This keeps the runner agent-agnostic. The runner should not know whether an
agent is local, model-backed, human, or remote.

The current code has an HTTP-based adapter named `HttpAgent`. Conceptually, this
is the current implementation of the Remote Agent path. Future naming can either
keep `HttpAgent` as the transport-specific class or introduce `RemoteAgent` as
the public concept and make HTTP one transport.

## 4. Protocol Contract

The Remote Agent Protocol defines what AI Battle Arena sends to the remote
service and what the service must return.

### Request

The request should be player-scoped. It must contain only what the acting player
is allowed to see.

Recommended fields:

```json
{
  "protocol_version": "0.1",
  "request_id": "optional stable decision id",
  "game": "kuhn_poker",
  "game_version": "1.0.0",
  "player": "player_0",
  "step_index": 0,
  "instructions": "Respond with exactly one legal action token.",
  "observation": {
    "player": "player_0",
    "private": {},
    "public": {},
    "history": [],
    "legal_actions": ["check", "bet"],
    "rendered": "..."
  },
  "match": {
    "episode": 0,
    "total_episodes": 100,
    "you": "agent_name",
    "standing": {}
  },
  "decision_seed": 123
}
```

Minimum required fields:

- `game`
- `game_version`
- `player`
- `step_index`
- `observation`
- `observation.legal_actions`

Recommended but optional fields:

- `protocol_version`
- `request_id`
- `instructions`
- `match`
- `decision_seed`

### Response

The response should return a game action, not model text.

Recommended fields:

```json
{
  "action": "bet",
  "amount": null,
  "message": "optional explanation",
  "raw_output": "optional original model/pipeline output",
  "metadata": {
    "model": "private-model-v1",
    "latency_ms": 120
  }
}
```

Required field:

- `action`

Optional fields:

- `amount`
- `message`
- `raw_output`
- `metadata`

For numeric games, `amount` is used when the game action requires it, such as a
Hold'em bet or raise. The game remains responsible for validating whether the
action and amount are legal.

## 5. Responsibility Split

### Framework responsibilities

AI Battle Arena is responsible for:

- constructing the player-scoped `AgentRequest`;
- ensuring the remote agent receives only the current player's observation;
- serializing the request;
- applying timeouts;
- parsing the response into `AgentResponse`;
- logging the response and metadata;
- asking the game to validate the returned move;
- applying the configured invalid-action policy.

### Remote service responsibilities

The remote service is responsible for:

- understanding the protocol fields it receives;
- choosing an action from the current observation;
- returning valid JSON;
- returning an `action` field;
- including `amount` when required by the game;
- optionally returning metadata useful for debugging or reproducibility.

The remote service is not required to reveal its internal implementation.

### Game responsibilities

The game is responsible for:

- defining legal actions;
- validating action and amount;
- transitioning state;
- enforcing game rules;
- defining terminal returns.

The Remote Agent adapter should not encode game-specific legality rules.

## 6. Failure Handling

Remote services can fail in normal ways:

- timeout;
- network error;
- malformed JSON;
- missing `action`;
- illegal action;
- illegal amount;
- service crash.

The Remote Agent adapter should convert transport/protocol failures into an
invalid `AgentResponse`, for example:

```python
AgentResponse(
    action=INVALID,
    metadata={"error": "...", "error_type": "timeout"}
)
```

The adapter should not decide fallback actions itself. Fallback or forfeit is a
runner-level policy, using the normal invalid-action flow.

This keeps the responsibility boundary clear:

- Remote Agent adapter handles transport/protocol conversion.
- Game handles legality.
- Runner handles invalid-action policy.

## 7. Reproducibility

Remote agents are inherently harder to reproduce than local deterministic
agents. The framework can provide reproducibility inputs, but it cannot force the
remote service to use them.

Recommended framework behavior:

- include `decision_seed` when available;
- include stable decision identifiers when possible;
- log the endpoint agent name and response metadata;
- encourage remote services to return model/version/config metadata.

Recommended remote service behavior:

- use the provided seed when stochastic behavior should be reproducible;
- return enough metadata to identify the agent version;
- avoid depending on hidden global state when claiming reproducibility.

The protocol should be honest: AI Battle Arena can make the request stream
reproducible, but cannot guarantee that an external service is deterministic.

## 8. Versioning

Remote Agent Protocol should be versioned because request/response schemas will
evolve.

For v0, versioning can stay simple:

- include `protocol_version` in requests;
- document the fields expected in responses;
- treat unknown response fields as metadata or ignore them;
- avoid breaking existing required fields.

The first stable protocol should optimize for clarity, not feature breadth.

## 9. Non-Goals For v0

Remote Agent v0 should not try to solve:

- plugin marketplace or discovery;
- authentication standards;
- multi-endpoint orchestration;
- streaming actions;
- arbitrary structured action schemas;
- exposing full hidden game state;
- standardized reasoning traces;
- model-provider integration through RemoteAgent.

Model-provider integration remains the responsibility of `ModelAgent` and
`ModelClient`.

## 10. Design Summary

Remote Agent is the path for external systems that already behave like agents.
It should share the same core `AgentRequest -> AgentResponse` semantics as local
agents, with HTTP as the transport.

The framework should define the protocol, not the remote implementation.

The most important boundary is:

```text
ModelAgent:  framework-controlled prompt/text/parser path
RemoteAgent: externally controlled observation/action path
```

Keeping that boundary clear lets AI Battle Arena support both official model
benchmarks and private agent submissions without mixing their semantics.
