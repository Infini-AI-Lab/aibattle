"""Coached Bedrock tournament for Claude Opus/Sonnet and GPT 5.x models.

Requested default matrix:
  * Models: opus 4.8, sonnet 4.6, gpt 5.5, gpt 5.4
  * Prompting: coached templates for every model
  * Reasoning: medium effort
  * Games/counts per model pair:
      - Connect Four: 50
      - Gomoku-Lite: 50
      - Hold'em 1-Hand Mode: 100
      - Hold'em Match Mode: 20
  * Logs: ../aibattle-logs/bedrock_coached_tournament by default
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path

from aibattle.agents.registry import make_agent
from aibattle.games.registry import make_game
from aibattle.logging.logger import MatchLogger
from aibattle.runner.runner import Runner


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT.parent / "aibattle-logs" / "bedrock_coached_tournament"
REASONING_EFFORT = "medium"


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
        "name": "gpt-5.5",
        "provider": "bedrock_openai",
        "model_id_env": "GPT_5_5_MODEL_ID",
        "default_model_id": "openai.gpt-5.5",
        "region_env": "GPT_5_5_REGION",
        "default_region": "us-east-2",
    },
    {
        "name": "gpt-5.4",
        "provider": "bedrock_openai",
        "model_id_env": "GPT_5_4_MODEL_ID",
        "default_model_id": "openai.gpt-5.4",
        "region_env": "GPT_5_4_REGION",
        "default_region": "us-east-2",
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


def _resolved_models() -> list[dict]:
    models = []
    for spec in MODEL_SPECS:
        m = dict(spec)
        m["model_id"] = os.environ.get(m["model_id_env"], m["default_model_id"])
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


def _agent_cfg(model: dict, args: argparse.Namespace) -> dict:
    model_block = {
        "provider": model["provider"],
        "model_id": model["model_id"],
        "aws_region": model["aws_region"],
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "reasoning_effort": REASONING_EFFORT,
    }
    if args.temperature is not None:
        model_block["temperature"] = args.temperature
    if model["provider"] == "bedrock_anthropic":
        model_block["thinking_budget_tokens"] = args.thinking_budget_tokens
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp.replace(path)


async def _run_game(
    game_spec: dict,
    *,
    models: list[dict],
    args: argparse.Namespace,
    out_root: Path,
    semaphore: asyncio.Semaphore,
    heartbeat: Heartbeat,
) -> dict:
    episodes = _env_int(game_spec["episodes_env"], game_spec["default_episodes"])
    pairs = list(itertools.combinations(models, 2))
    random.Random(args.launch_seed + len(game_spec["label"])).shuffle(pairs)
    game_dir = out_root / game_spec["label"]
    game_dir.mkdir(parents=True, exist_ok=True)
    data_path = game_dir / f"{game_spec['label']}_data.json"
    data = {
        "title": game_spec["title"],
        "game": game_spec["game_name"],
        "label": game_spec["label"],
        "episodes_per_pair": episodes,
        "coached": True,
        "reasoning_effort": REASONING_EFFORT,
        "seat_swap": game_spec["seat_swap"],
        "models": [
            {
                "name": m["name"],
                "provider": m["provider"],
                "model_id": m["model_id"],
                "aws_region": m["aws_region"],
            }
            for m in models
        ],
        "pairs": [],
    }

    done = 0
    total = len(pairs)
    print(
        f"{game_spec['title']}: {total} pairs x {episodes} episodes, "
        f"coached=True, reasoning_effort={REASONING_EFFORT}",
        flush=True,
    )

    async def play(pair_index: int, a: dict, b: dict):
        nonlocal done
        a_name, b_name = a["name"], b["name"]
        pair_label = f"{a_name}__vs__{b_name}"
        pair_dir = game_dir / _safe_name(pair_label)
        pair_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + (1000 * GAME_SPECS.index(game_spec)) + pair_index
        runner = Runner(
            lambda: make_game(game_spec["game_name"], dict(game_spec["params"])),
            on_invalid_action="fallback",
        )
        started = time.perf_counter()
        try:
            with MatchLogger(None) as logger:
                res = await runner.run_match(
                    make_agent(_agent_cfg(a, args), game_name=game_spec["game_name"]),
                    make_agent(_agent_cfg(b, args), game_name=game_spec["game_name"]),
                    episodes=episodes,
                    seed=seed,
                    seat_swap=game_spec["seat_swap"],
                    logger=logger,
                    semaphore=semaphore,
                    episode_dir=str(pair_dir),
                    on_step=heartbeat.cb(game_spec["label"], pair_label),
                )
            pair_data = {
                "a": a_name,
                "b": b_name,
                "seed": seed,
                "episode_dir": str(pair_dir),
                "episodes": [_trim_episode(e) for e in res.episodes],
                "summary": _pair_summary(res.episodes),
                "failures": res.failures,
            }
            data["pairs"].append(pair_data)
            _save_json(data_path, data)
            done += 1
            print(
                f"[{done}/{total}] {game_spec['label']} {a_name} vs {b_name} "
                f"done in {time.perf_counter() - started:.0f}s "
                f"episodes={len(res.episodes)}/{episodes} failures={res.failures} "
                f"summary={pair_data['summary']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            done += 1
            print(
                f"[{done}/{total}] {game_spec['label']} {a_name} vs {b_name} "
                f"FAILED: {exc}",
                flush=True,
            )
            traceback.print_exc()

    await asyncio.gather(*(play(i, a, b) for i, (a, b) in enumerate(pairs)))
    _save_json(data_path, data)
    return data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_temperature = os.environ.get("TEMPERATURE")
    p.add_argument("--out", default=os.environ.get("OUT", str(DEFAULT_OUT)))
    p.add_argument("--games", default=os.environ.get("GAMES", ""))
    p.add_argument("--models", default=os.environ.get("MODELS", ""))
    p.add_argument("--max-concurrency", type=int,
                   default=_env_int("MAX_CONCURRENCY", 16))
    p.add_argument("--max-tokens", type=int, default=_env_int("MAX_TOKENS", 1024))
    p.add_argument("--thinking-budget-tokens", type=int,
                   default=_env_int("THINKING_BUDGET_TOKENS", 1024))
    p.add_argument("--temperature", type=float,
                   default=(float(default_temperature)
                            if default_temperature is not None else None))
    p.add_argument("--timeout-s", type=float,
                   default=float(os.environ.get("BEDROCK_TIMEOUT", "900")))
    p.add_argument("--max-retries", type=int, default=_env_int("MAX_RETRIES", 2))
    p.add_argument("--seed", type=int, default=_env_int("SEED", 20260607))
    p.add_argument("--launch-seed", type=int, default=_env_int("LAUNCH_SEED", 9876))
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
        print(
            f"  - {m['name']}: provider={m['provider']} "
            f"model_id={m['model_id']} region={m['aws_region']}",
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
        "max_tokens": args.max_tokens,
        "thinking_budget_tokens": args.thinking_budget_tokens,
        "temperature": args.temperature,
        "models": models,
        "games": games,
    }
    _save_json(out_root / "manifest.json", manifest)

    heartbeat = Heartbeat(out_root)
    sem = asyncio.Semaphore(args.max_concurrency)
    start = time.perf_counter()
    try:
        results = []
        for game in games:
            results.append(
                await _run_game(
                    game,
                    models=models,
                    args=args,
                    out_root=out_root,
                    semaphore=sem,
                    heartbeat=heartbeat,
                )
            )
        _save_json(out_root / "summary.json", {"games": results})
    finally:
        heartbeat.close()
    print(f"DONE in {time.perf_counter() - start:.0f}s", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
