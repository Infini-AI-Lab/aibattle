"""Coached tournament for Claude plus Fireworks-hosted models.

Requested default matrix:
  * Models:
      - claude-opus-4.8
      - claude-sonnet-4.6
      - deepseek-v4-pro
      - gpt-oss-120b
      - kimi-k2p6
      - glm-5p1
      - minimax-m2p7
  * Prompting: coached templates for every model
  * Games/counts per model pair:
      - Connect Four: 50
      - Gomoku-Lite: 50
      - Hold'em 1-Hand Mode: 100
      - Hold'em Match Mode: 20
  * Logs: ../aibattle-logs/gpt_claude_fireworks_coached_tournament by default
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import re
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    REPO_ROOT.parent / "aibattle-logs" / "gpt_claude_fireworks_coached_tournament"
)
REASONING_EFFORT = "medium"
ACTION_ONLY_SYSTEM_PROMPT = (
    "You are a competitive game-playing agent. Use the rules, state, and "
    "coaching in the user prompt privately. The reply format is mandatory: "
    "return only the final legal action and nothing else. Any extra word, "
    "label, punctuation, explanation, or board restatement is invalid. Valid "
    "examples: 3 ; 7 7 ; call ; check ; fold ; raise 40. Invalid examples: "
    "Action: 3 ; I choose 3 ; 3 because center is best."
)


MODEL_SPECS = [
    {
        "name": "claude-opus-4.8",
        "provider": "bedrock_anthropic",
        "model_id_env": "CLAUDE_OPUS_4_8_MODEL_ID",
        "default_model_id": "us.anthropic.claude-opus-4-8",
        "region_env": "CLAUDE_OPUS_4_8_REGION",
        "default_region": "us-west-2",
    },
    {
        "name": "claude-sonnet-4.6",
        "provider": "bedrock_anthropic",
        "model_id_env": "CLAUDE_SONNET_4_6_MODEL_ID",
        "default_model_id": "us.anthropic.claude-sonnet-4-6",
        "region_env": "CLAUDE_SONNET_4_6_REGION",
        "default_region": "us-east-1",
    },
    {
        "name": "deepseek-v4-pro",
        "provider": "fireworks",
        "model_id_env": "DEEPSEEK_V4_PRO_MODEL_ID",
        "default_model_id": "accounts/fireworks/models/deepseek-v4-pro",
    },
    {
        "name": "gpt-oss-120b",
        "provider": "fireworks",
        "model_id_env": "GPT_OSS_120B_MODEL_ID",
        "default_model_id": "accounts/fireworks/models/gpt-oss-120b",
    },
    {
        "name": "kimi-k2p6",
        "provider": "fireworks",
        "model_id_env": "KIMI_K2P6_MODEL_ID",
        "default_model_id": "accounts/fireworks/models/kimi-k2p6",
    },
    {
        "name": "glm-5p1",
        "provider": "fireworks",
        "model_id_env": "GLM_5P1_MODEL_ID",
        "default_model_id": "accounts/fireworks/models/glm-5p1",
    },
    {
        "name": "minimax-m2p7",
        "provider": "fireworks",
        "model_id_env": "MINIMAX_M2P7_MODEL_ID",
        "default_model_id": "accounts/fireworks/models/minimax-m2p7",
    },
]


GAME_SPECS = [
    {
        "label": "connect4",
        "title": "Connect Four",
        "game_name": "connect4",
        "episodes_env": "CONNECT4_EPISODES",
        "default_episodes": 50,
        "params": {"random_open": 2},
        "seat_swap": True,
    },
    {
        "label": "gomoku",
        "title": "Gomoku-Lite",
        "game_name": "gomoku",
        "episodes_env": "GOMOKU_EPISODES",
        "default_episodes": 50,
        "params": {"random_open": 2},
        "seat_swap": True,
    },
    {
        "label": "holdem_1hand",
        "title": "Hold'em 1-Hand Mode",
        "game_name": "holdem",
        "episodes_env": "HOLDEM_1HAND_EPISODES",
        "default_episodes": 100,
        "params": {"starting_stack": 200},
        "seat_swap": True,
    },
    {
        "label": "holdem_match",
        "title": "Hold'em Match Mode",
        "game_name": "holdem_match",
        "episodes_env": "HOLDEM_MATCH_EPISODES",
        "default_episodes": 20,
        "params": {"starting_stack": 200, "max_hands": 30},
        "seat_swap": True,
    },
]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _episode_json_paths(pair_dir: Path) -> list[Path]:
    return sorted(
        p for p in pair_dir.glob("ep*.json")
        if not p.name.endswith(".error.json")
    )


def _load_episode_jsons(pair_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _episode_json_paths(pair_dir)
    ]


def _resolved_models() -> list[dict]:
    models = []
    for spec in MODEL_SPECS:
        m = dict(spec)
        m["model_id"] = os.environ.get(m["model_id_env"], m["default_model_id"])
        if m["provider"] == "bedrock_anthropic":
            m["aws_region"] = os.environ.get(m["region_env"], m["default_region"])
        models.append(m)
    return models


def _selected(items: list[dict], wanted: str | None, *, key: str) -> list[dict]:
    if not wanted:
        return items
    keep = {x.strip() for x in wanted.split(",") if x.strip()}
    return [x for x in items if x[key] in keep or x.get("name") in keep]


class Heartbeat:
    def __init__(self, root: Path):
        self.root = root / "_heartbeat"
        self.root.mkdir(parents=True, exist_ok=True)
        self._files = {}

    def cb(self, game_label: str, pair_label: str):
        path = self.root / f"{game_label}_{_safe_name(pair_label)}.log"
        fh = self._files.get(path)
        if fh is None:
            fh = path.open("a", buffering=1, encoding="utf-8")
            self._files[path] = fh

        def _write(ev):
            pub = ev.get("public") or {}
            hand = pub.get("match_hand") or ev.get("episode")
            max_hands = pub.get("match_max_hands") or "?"
            street = pub.get("street") or ""
            amount = ev.get("amount")
            raw = ev.get("raw_output") or ""
            fh.write(
                f"{time.strftime('%H:%M:%S')} [{game_label} {pair_label}] "
                f"ep={ev['episode']} hand={hand}/{max_hands} {street} "
                f"s{ev['step']:02d} {ev['agent_name']} -> {ev['action']}"
                f"{(' ' + str(amount)) if amount is not None else ''} "
                f"(out_len={len(raw)})\n"
            )

        return _write

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()


def _kimi_concurrency_key(game_label: str) -> str:
    bucket = "holdem" if game_label.startswith("holdem") else "board"
    return f"fireworks:kimi-k2p6:{bucket}"


def _agent_cfg(model: dict, args: argparse.Namespace, *, game_label: str) -> dict:
    model_block = {
        "provider": model["provider"],
        "model_id": model["model_id"],
        "max_tokens": (
            args.fireworks_max_tokens
            if model["provider"] == "fireworks"
            else args.anthropic_max_tokens
        ),
        "timeout_s": args.timeout_s,
        "system_prompt": ACTION_ONLY_SYSTEM_PROMPT,
    }
    if model["provider"] == "bedrock_anthropic":
        model_block["aws_region"] = model["aws_region"]
        model_block["reasoning_effort"] = REASONING_EFFORT
        model_block["thinking_budget_tokens"] = args.thinking_budget_tokens
        if args.temperature is not None:
            model_block["temperature"] = args.temperature
    else:
        model_block["api_key_env"] = "FIREWORKS_API_KEY"
        if args.fireworks_temperature is not None:
            model_block["temperature"] = args.fireworks_temperature
        if model["name"] == "kimi-k2p6":
            model_block["global_concurrency_limit"] = args.kimi_global_concurrency_limit
            model_block["concurrency_key"] = _kimi_concurrency_key(game_label)
    return {
        "type": "model",
        "name": model["name"],
        "coached": True,
        "model": model_block,
        "max_retries": args.max_retries,
    }


def _trim_episode(e: dict) -> dict:
    e2 = dict(e)
    steps = []
    for step in e.get("steps", []):
        s2 = dict(step)
        response = dict(s2.get("response") or {})
        response.pop("raw_output", None)
        response.pop("prompt", None)
        s2["response"] = response
        steps.append(s2)
    e2["steps"] = steps
    return e2


def _pair_summary(episodes: list[dict]) -> dict:
    wins = defaultdict(int)
    total_returns = defaultdict(float)
    invalids = defaultdict(int)
    draws = 0
    for ep in episodes:
        winner = ep.get("winner_name")
        if winner:
            wins[winner] += 1
        else:
            draws += 1
        for seat, model_name in ep.get("seat_assignment", {}).items():
            total_returns[model_name] += float(ep.get("returns", {}).get(seat, 0.0))
            invalids[model_name] += int(ep.get("invalid_count", {}).get(seat, 0))
    return {
        "wins": dict(wins),
        "draws": draws,
        "returns": dict(total_returns),
        "invalids": dict(invalids),
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{path.name}."
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=prefix,
        suffix=".tmp",
        delete=False,
    ) as fh:
        fh.write(json.dumps(payload, indent=2))
        tmp = Path(fh.name)
    tmp.replace(path)


async def _run_pair(
    *,
    game: dict,
    model_a: dict,
    model_b: dict,
    pair_id: int,
    out_root: Path,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
    heartbeat: Heartbeat,
) -> dict:
    pair_label = f"{model_a['name']}__vs__{model_b['name']}"
    pair_dir = out_root / game["label"] / pair_label
    pair_dir.mkdir(parents=True, exist_ok=True)
    episodes = _env_int(game["episodes_env"], game["default_episodes"])
    runner = Runner(
        lambda: make_game(game["game_name"], dict(game["params"])),
        on_invalid_action="fallback",
    )
    started = time.perf_counter()
    try:
        failures = 0
        completed = len(_episode_json_paths(pair_dir))
        while completed < episodes:
            target = min(episodes, completed + args.pair_batch_size)
            with MatchLogger(None) as logger:
                result = await runner.run_match(
                    make_agent(
                        _agent_cfg(model_a, args, game_label=game["label"]),
                        game_name=game["game_name"],
                    ),
                    make_agent(
                        _agent_cfg(model_b, args, game_label=game["label"]),
                        game_name=game["game_name"],
                    ),
                    episodes=target,
                    seat_swap=bool(game.get("seat_swap", False)),
                    seed=args.seed + pair_id,
                    logger=logger,
                    semaphore=sem,
                    episode_dir=str(pair_dir),
                    on_step=heartbeat.cb(game["label"], pair_label),
                )
            failures += result.failures
            completed = len(_episode_json_paths(pair_dir))

        trimmed = [_trim_episode(ep) for ep in _load_episode_jsons(pair_dir)]
        pair_summary = _pair_summary(trimmed)
        elapsed = time.perf_counter() - started
        print(
            f"[{pair_id + 1}] {game['label']} {model_a['name']} vs {model_b['name']} "
            f"done in {elapsed:.0f}s episodes={len(trimmed)}/{episodes} "
            f"failures={failures} summary={pair_summary}",
            flush=True,
        )
        return {
            "a": model_a["name"],
            "b": model_b["name"],
            "seed": args.seed + pair_id,
            "episode_dir": str(pair_dir),
            "episodes": trimmed,
            "failures": failures,
            "summary": pair_summary,
        }
    except Exception as exc:  # noqa: BLE001
        err_path = pair_dir / "pair.error.json"
        _save_json(
            err_path,
            {
                "game": game["label"],
                "pair": pair_label,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"[{pair_id + 1}] {game['label']} {model_a['name']} vs {model_b['name']} "
            f"FAILED: {exc}",
            flush=True,
        )
        raise


async def _run_game(
    game: dict,
    *,
    models: list[dict],
    out_root: Path,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
    heartbeat: Heartbeat,
) -> dict:
    pairs = list(itertools.combinations(models, 2))
    print(
        f"{game['title']}: {len(pairs)} pairs x "
        f"{_env_int(game['episodes_env'], game['default_episodes'])} episodes, "
        f"coached=True, reasoning_effort={REASONING_EFFORT}",
        flush=True,
    )
    tasks = [
        _run_pair(
            game=game,
            model_a=a,
            model_b=b,
            pair_id=i,
            out_root=out_root,
            args=args,
            sem=sem,
            heartbeat=heartbeat,
        )
        for i, (a, b) in enumerate(pairs)
    ]
    results = await asyncio.gather(*tasks)
    data = {
        "title": game["title"],
        "game": game["game_name"],
        "label": game["label"],
        "episodes_per_pair": _env_int(game["episodes_env"], game["default_episodes"]),
        "coached": True,
        "reasoning_effort": REASONING_EFFORT,
        "seat_swap": bool(game.get("seat_swap", False)),
        "models": [m["name"] for m in models],
        "pairs": results,
    }
    _save_json(out_root / game["label"] / f"{game['label']}_data.json", data)
    return data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_temperature = os.environ.get("TEMPERATURE")
    default_fireworks_temperature = os.environ.get("FIREWORKS_TEMPERATURE")
    p.add_argument("--out", default=os.environ.get("OUT", str(DEFAULT_OUT)))
    p.add_argument("--games", default=os.environ.get("GAMES", ""))
    p.add_argument("--models", default=os.environ.get("MODELS", ""))
    p.add_argument("--max-concurrency", type=int,
                   default=_env_int("MAX_CONCURRENCY", 16))
    p.add_argument("--anthropic-max-tokens", type=int,
                   default=_env_int("ANTHROPIC_MAX_TOKENS", 128000))
    p.add_argument("--fireworks-max-tokens", type=int,
                   default=_env_int("FIREWORKS_MAX_TOKENS", 128000))
    p.add_argument("--thinking-budget-tokens", type=int,
                   default=_env_int("THINKING_BUDGET_TOKENS", 1024))
    p.add_argument("--temperature", type=float,
                   default=(float(default_temperature)
                            if default_temperature is not None else None))
    p.add_argument("--fireworks-temperature", type=float,
                   default=(float(default_fireworks_temperature)
                           if default_fireworks_temperature is not None else 0.0))
    p.add_argument("--kimi-global-concurrency-limit", type=int,
                   default=_env_int("KIMI_GLOBAL_CONCURRENCY_LIMIT", 4))
    p.add_argument("--timeout-s", type=float,
                   default=float(os.environ.get("TOURNAMENT_TIMEOUT", "900")))
    p.add_argument("--max-retries", type=int, default=_env_int("MAX_RETRIES", 2))
    p.add_argument("--pair-batch-size", type=int,
                   default=_env_int("PAIR_BATCH_SIZE", 4))
    p.add_argument("--seed", type=int, default=_env_int("SEED", 20260607))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


async def main_async() -> None:
    args = _parse_args()
    out_root = Path(args.out).resolve()
    models = _selected(_resolved_models(), args.models or None, key="name")
    games = _selected(GAME_SPECS, args.games or None, key="label")
    if len(models) < 2:
        raise ValueError("At least two models are required for a tournament.")
    if not games:
        raise ValueError("No games selected.")

    print(f"Output root: {out_root}", flush=True)
    print("Models:", flush=True)
    for m in models:
        extra = f" region={m['aws_region']}" if m.get("aws_region") else ""
        print(
            f"  - {m['name']}: provider={m['provider']} "
            f"model_id={m['model_id']}{extra}",
            flush=True,
        )
    print("Games:", flush=True)
    for g in games:
        ep = _env_int(g["episodes_env"], g["default_episodes"])
        print(f"  - {g['title']} ({g['label']}): {ep} per pair", flush=True)
    if args.dry_run:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(out_root),
        "coached": True,
        "reasoning_effort": REASONING_EFFORT,
        "max_concurrency": args.max_concurrency,
        "anthropic_max_tokens": args.anthropic_max_tokens,
        "fireworks_max_tokens": args.fireworks_max_tokens,
        "timeout_s": args.timeout_s,
        "thinking_budget_tokens": args.thinking_budget_tokens,
        "system_prompt": ACTION_ONLY_SYSTEM_PROMPT,
        "temperature": args.temperature,
        "fireworks_temperature": args.fireworks_temperature,
        "kimi_global_concurrency_limit": args.kimi_global_concurrency_limit,
        "pair_batch_size": args.pair_batch_size,
        "models": models,
        "games": games,
    }
    _save_json(out_root / "manifest.json", manifest)

    heartbeat = Heartbeat(out_root)
    sem = asyncio.Semaphore(args.max_concurrency)
    start = time.perf_counter()
    try:
        results = await asyncio.gather(
            *[
                _run_game(
                    game,
                    models=models,
                    out_root=out_root,
                    args=args,
                    sem=sem,
                    heartbeat=heartbeat,
                )
                for game in games
            ]
        )
        _save_json(out_root / "summary.json", {"games": results})
        print(f"DONE in {time.perf_counter() - start:.0f}s", flush=True)
    finally:
        heartbeat.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
