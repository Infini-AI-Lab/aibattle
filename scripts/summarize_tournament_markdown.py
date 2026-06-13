"""Build a markdown summary from a tournament log directory.

Reads `manifest.json` plus per-pair `ep*.json` files and writes a human-readable
markdown report with setup details, completion audit, and result tables.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


GAME_DEFAULTS = {
    "connect4": {
        "label": "connect4",
        "title": "Connect Four",
        "default_episodes": 50,
        "params": {"random_open": 2},
    },
    "gomoku": {
        "label": "gomoku",
        "title": "Gomoku-Lite",
        "default_episodes": 50,
        "params": {"random_open": 2},
    },
    "holdem_1hand": {
        "label": "holdem_1hand",
        "title": "Hold'em 1-Hand Mode",
        "default_episodes": 100,
        "params": {"starting_stack": 200},
    },
    "holdem_match": {
        "label": "holdem_match",
        "title": "Hold'em Match Mode",
        "default_episodes": 20,
        "params": {"starting_stack": 200, "max_hands": 30},
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", help="Tournament log directory with manifest.json")
    p.add_argument(
        "--output",
        help="Markdown output path. Defaults to reports/<logdir>_summary.md",
    )
    p.add_argument("--title", help="Report title override")
    p.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Optional request timeout to include in the setup section",
    )
    return p.parse_args()


def _completed_episode_paths(pair_dir: Path) -> list[Path]:
    return sorted(
        p for p in pair_dir.glob("ep*.json") if not p.name.endswith(".error.json")
    )


def _pair_dirs(game_dir: Path) -> list[Path]:
    return sorted(p for p in game_dir.glob("*__vs__*") if p.is_dir())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def _expected_pairs(models: list[str]) -> list[str]:
    return [f"{a}__vs__{b}" for a, b in combinations(models, 2)]


def _collect_game_stats(
    game_dir: Path,
    *,
    models: list[str],
) -> dict:
    stats = {
        name: {"wins": 0, "returns": 0.0, "invalids": 0, "episodes": 0}
        for name in models
    }
    draws = 0
    truncated_steps = 0
    reasoning_steps = 0
    completion_tokens = defaultdict(int)
    prompt_tokens = defaultdict(int)
    latencies = defaultdict(list)

    for pair_dir in _pair_dirs(game_dir):
        for ep_path in _completed_episode_paths(pair_dir):
            ep = _load_json(ep_path)
            winner = ep.get("winner_name")
            if not winner:
                draws += 1
            for seat, model_name in ep.get("seat_assignment", {}).items():
                stats[model_name]["episodes"] += 1
                stats[model_name]["returns"] += float(
                    ep.get("returns", {}).get(seat, 0.0)
                )
                stats[model_name]["invalids"] += int(
                    ep.get("invalid_count", {}).get(seat, 0)
                )
                if winner == model_name:
                    stats[model_name]["wins"] += 1
            for step in ep.get("steps", []):
                meta = (step.get("response") or {}).get("metadata", {})
                if meta.get("truncated"):
                    truncated_steps += 1
                if meta.get("has_reasoning"):
                    reasoning_steps += 1
                if isinstance(meta.get("completion_tokens"), (int, float)):
                    completion_tokens[step["agent_name"]] += int(meta["completion_tokens"])
                if isinstance(meta.get("prompt_tokens"), (int, float)):
                    prompt_tokens[step["agent_name"]] += int(meta["prompt_tokens"])
                if isinstance(meta.get("latency_ms"), (int, float)):
                    latencies[step["agent_name"]].append(float(meta["latency_ms"]))

    ranking = sorted(
        stats.items(),
        key=lambda kv: (-kv[1]["returns"], -kv[1]["wins"], kv[0]),
    )
    return {
        "stats": stats,
        "ranking": ranking,
        "draws": draws,
        "truncated_steps": truncated_steps,
        "reasoning_steps": reasoning_steps,
        "completion_tokens": dict(completion_tokens),
        "prompt_tokens": dict(prompt_tokens),
        "latencies": {
            model: (sum(vals) / len(vals) if vals else None)
            for model, vals in latencies.items()
        },
    }


def _collect_error_types(root: Path) -> Counter:
    counts = Counter()
    for path in root.glob("**/*.error.json"):
        try:
            payload = _load_json(path)
        except Exception:
            counts["Unreadable"] += 1
            continue
        key = payload.get("error_type") or "Unknown"
        counts[key] += 1
    return counts


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _report_title(args: argparse.Namespace, root: Path) -> str:
    if args.title:
        return args.title
    name = root.name.replace("_", " ").strip()
    return f"{name.title()} Summary"


def _load_manifest(root: Path) -> dict:
    for name in ("manifest.full.json", "manifest.json"):
        path = root / name
        if path.exists():
            return _load_json(path)
    raise FileNotFoundError(f"No manifest found under {root}")


def _resolve_games(root: Path, manifest: dict) -> list[dict]:
    games = list(manifest.get("games") or [])
    found_labels = sorted(
        label
        for label in GAME_DEFAULTS
        if (root / label).is_dir()
    )
    if len(games) >= len(found_labels):
        return games
    by_label = {g["label"]: dict(g) for g in games if g.get("label")}
    merged = []
    for label in found_labels:
        merged.append({**GAME_DEFAULTS[label], **by_label.get(label, {})})
    return merged


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    manifest = _load_manifest(out_dir)
    models = [m["name"] for m in manifest["models"]]
    games = _resolve_games(out_dir, manifest)
    expected_pair_count = len(_expected_pairs(models))
    title = _report_title(args, out_dir)

    output = (
        Path(args.output).resolve()
        if args.output
        else Path("reports").resolve() / f"{out_dir.name}_summary.md"
    )

    total_requested = 0
    total_completed = 0
    total_invalids = 0
    total_truncated = 0
    total_reasoning = 0
    overall = {
        name: {"wins": 0, "returns": 0.0, "invalids": 0, "episodes": 0}
        for name in models
    }
    total_draws = 0
    completion_rows = []
    game_sections = []

    for game in games:
        label = game["label"]
        game_dir = out_dir / label
        target = int(game["default_episodes"])
        requested = expected_pair_count * target
        pair_dirs = _pair_dirs(game_dir)
        pair_counts = {
            pair_dir.name: len(_completed_episode_paths(pair_dir)) for pair_dir in pair_dirs
        }
        counts = [pair_counts.get(name, 0) for name in _expected_pairs(models)]
        completed = sum(counts)
        errors = len(list(game_dir.glob("**/*.error.json")))
        stats = _collect_game_stats(game_dir, models=models)
        total_requested += requested
        total_completed += completed
        total_draws += stats["draws"]
        total_truncated += stats["truncated_steps"]
        total_reasoning += stats["reasoning_steps"]

        for model_name, model_stats in stats["stats"].items():
            overall[model_name]["wins"] += model_stats["wins"]
            overall[model_name]["returns"] += model_stats["returns"]
            overall[model_name]["invalids"] += model_stats["invalids"]
            overall[model_name]["episodes"] += model_stats["episodes"]
            total_invalids += model_stats["invalids"]

        completion_rows.append(
            [
                game["title"],
                f"{completed}/{requested}",
                ", ".join(str(n) for n in counts) if counts else "-",
                str(errors),
            ]
        )

        rank_rows = []
        for rank, (model_name, model_stats) in enumerate(stats["ranking"], 1):
            rank_rows.append(
                [
                    str(rank),
                    model_name,
                    str(model_stats["episodes"]),
                    str(model_stats["wins"]),
                    _fmt_num(model_stats["returns"]),
                    str(model_stats["invalids"]),
                ]
            )
        note = ""
        if completed < requested:
            note = (
                f"\nThis table is provisional: `{completed}/{requested}` episodes are "
                f"present for {game['title']}."
            )
        game_sections.append(
            "\n".join(
                [
                    f"### {game['title']}",
                    "",
                    _markdown_table(
                        ["Rank", "Model", "Episodes", "Wins", "Raw return", "Invalid actions"],
                        rank_rows,
                    ),
                    "",
                    f"Draws: {stats['draws']}.{note}",
                ]
            )
        )

    overall_ranking = sorted(
        overall.items(),
        key=lambda kv: (-kv[1]["returns"], -kv[1]["wins"], kv[0]),
    )
    overall_rows = []
    for rank, (model_name, model_stats) in enumerate(overall_ranking, 1):
        overall_rows.append(
            [
                str(rank),
                model_name,
                str(model_stats["episodes"]),
                str(model_stats["wins"]),
                _fmt_num(model_stats["returns"]),
                str(model_stats["invalids"]),
            ]
        )

    model_rows = []
    for model in manifest["models"]:
        model_rows.append(
            [
                model["name"],
                model["model_id"],
                model.get("aws_region", ""),
            ]
        )

    game_rows = []
    for game in games:
        target = int(game["default_episodes"])
        total = expected_pair_count * target
        notes = ", ".join(
            f"{k}={v}" for k, v in sorted((game.get("params") or {}).items())
        )
        game_rows.append(
            [
                game["title"],
                str(target),
                str(expected_pair_count),
                str(total),
                notes or "-",
            ]
        )

    error_types = _collect_error_types(out_dir)
    error_rows = [[name, str(count)] for name, count in error_types.most_common()]

    quality_rows = [
        ["Invalid actions", str(total_invalids)],
        ["Truncated response steps", str(total_truncated)],
        ["Reasoning-tagged steps", str(total_reasoning)],
    ]

    settings = [
        f"- Created at: `{manifest['created_at']}`",
        f"- Output directory: `{manifest['output_root']}`",
        f"- Prompt template: `coached`" if manifest.get("coached") else "- Prompt template: `plain`",
        "- Response format: action-only system instruction",
        f"- Claude reasoning effort: `{manifest.get('reasoning_effort')}`",
        f"- Claude max output tokens: `{manifest.get('anthropic_max_tokens')}`",
        f"- Fireworks max output tokens: `{manifest.get('fireworks_max_tokens')}`",
        f"- Fireworks temperature: `{manifest.get('fireworks_temperature')}`",
        f"- Max concurrency: `{manifest.get('max_concurrency')}`",
        f"- Pair batch size: `{manifest.get('pair_batch_size')}`",
        f"- Thinking budget tokens: `{manifest.get('thinking_budget_tokens')}`",
    ]
    timeout_s = args.timeout_s
    if timeout_s is None and manifest.get("timeout_s") is not None:
        timeout_s = float(manifest["timeout_s"])
    if timeout_s is not None:
        settings.append(f"- Request timeout: `{_fmt_num(timeout_s)}s`")

    complete = total_completed == total_requested and not error_types
    summary_label = "Overall Results" if complete else "Current Results"
    top_model = overall_ranking[0][0] if overall_ranking else "n/a"
    max_invalid_model = max(
        overall.items(), key=lambda kv: (kv[1]["invalids"], kv[0])
    )[0] if overall else "n/a"
    leader_lines = [
        f"- Overall raw-return leader: {top_model}.",
    ]
    for game, section in zip(games, game_sections):
        del section
        game_dir = out_dir / game["label"]
        stats = _collect_game_stats(game_dir, models=models)
        if stats["ranking"]:
            leader_lines.append(
                f"- {game['title']} leader by raw return: {stats['ranking'][0][0]}."
            )
    leader_lines.append(
        f"- Highest invalid-action count so far: {max_invalid_model}."
    )
    if complete:
        leader_lines.append(
            "- The dataset is complete: every scheduled episode is present and no error files remain."
        )
    else:
        leader_lines.append(
            f"- The run is still in progress: `{total_completed}/{total_requested}` completed episodes, `{sum(error_types.values())}` error files."
        )

    lines = [
        f"# {title}",
        "",
    ]
    if not complete:
        lines.extend(
            [
                "This report is provisional. The tournament is still running, so the results below can move as the remaining episodes land.",
                "",
            ]
        )
    lines.extend(
        [
            "## Experiment Setting",
            "",
            *settings,
            "",
            "## Models",
            "",
            _markdown_table(["Model", "Model identifier", "Region"], model_rows),
            "",
            "## Games",
            "",
            _markdown_table(
                ["Game", "Episodes per pair", "Pairs", "Total episodes", "Notes"],
                game_rows,
            ),
            "",
            f"Total scheduled episodes: `{total_requested}`.",
            "",
            "## Completion Audit",
            "",
            _markdown_table(
                ["Game", "Episodes", "Pair distribution", "Error files"],
                completion_rows,
            ),
            "",
            f"Total completed episodes: `{total_completed}`.",
            "",
            "Quality counters:",
            "",
            _markdown_table(["Counter", "Count"], quality_rows),
            "",
        ]
    )
    if error_rows:
        lines.extend(
            [
                "Outstanding error types:",
                "",
                _markdown_table(["Error type", "Count"], error_rows),
                "",
            ]
        )
    lines.extend(
        [
            f"## {summary_label}",
            "",
            (
                "Raw return is additive within each game, but the game scales differ. "
                "Use the per-game tables as the main comparison surface."
            ),
            "",
            _markdown_table(
                ["Rank", "Model", "Episodes", "Wins", "Raw return", "Invalid actions"],
                overall_rows,
            ),
            "",
            f"Total draws: {total_draws}.",
            "",
            "## Results by Game",
            "",
            *game_sections,
            "",
            "## Main Takeaways",
            "",
            *leader_lines,
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
