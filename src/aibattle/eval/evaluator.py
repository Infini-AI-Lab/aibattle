"""Evaluation layer: a pure function of log file(s) -> summary.

Reads a match JSONL log and produces per-agent statistics. Never touches live
game/agent objects. Metrics are aggregated by agent *identity* (name) across
both seat assignments so reported skill is position-neutral.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Callable, Optional


def _parse_records(path: str):
    """Parse every non-empty JSONL record once."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def evaluate(path: str, progress: Optional[Callable] = None) -> dict:
    header = None
    episodes = []
    steps = []
    # per agent name -> accumulators
    payoffs = defaultdict(list)
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    steps_total = defaultdict(int)
    steps_invalid = defaultdict(int)
    action_counts = defaultdict(lambda: defaultdict(int))
    lengths = []

    # Single read of the log. Progress is reported per *episode* (the unit the
    # user cares about): the denominator is the episode count from the match
    # header, and the bar ticks as each episode's summary record is parsed.
    total_eps = None
    seen_eps = 0
    for rec in _parse_records(path):
        rt = rec.get("record_type")
        if rt == "match":
            header = rec
            total_eps = rec.get("episodes")
        elif rt == "step":
            steps.append(rec)
        elif rt == "episode":
            episodes.append(rec)
            seen_eps += 1
            if progress is not None:
                progress(seen_eps, total_eps or seen_eps)
    if progress is not None and not episodes:
        progress(1, 1)  # nothing to do; close the bar cleanly

    # seat -> agent name per episode (steps precede the episode summary record).
    seat_maps = {ep["episode"]: ep["seat_assignment"] for ep in episodes}

    for rec in steps:
        smap = seat_maps.get(rec["episode"], {})
        name = smap.get(rec["player"], rec["player"])
        steps_total[name] += 1
        if rec.get("invalid"):
            steps_invalid[name] += 1
        action_counts[name][rec["selected_action"]] += 1

    for ep in episodes:
        lengths.append(ep["length"])
        smap = ep["seat_assignment"]
        returns = ep["returns"]
        for seat, payoff in returns.items():
            name = smap.get(seat, seat)
            payoffs[name].append(payoff)
            if payoff > 0:
                wins[name] += 1
            elif payoff < 0:
                losses[name] += 1
            else:
                ties[name] += 1

    agents = sorted(payoffs)
    per_agent = {}
    for name in agents:
        ps = payoffs[name]
        n = len(ps)
        mean = sum(ps) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in ps) / (n - 1) if n > 1 else 0.0
        stderr = math.sqrt(var / n) if n else 0.0
        ci95 = 1.96 * stderr
        st = steps_total[name]
        per_agent[name] = {
            "episodes": n,
            "mean_payoff_per_hand": round(mean, 4),
            "payoff_ci95": round(ci95, 4),
            "total_payoff": round(sum(ps), 2),
            "wins": wins[name],
            "losses": losses[name],
            "ties": ties[name],
            "win_rate": round(wins[name] / n, 4) if n else 0.0,
            "invalid_action_rate": round(steps_invalid[name] / st, 4) if st else 0.0,
            "steps": st,
            "action_distribution": dict(action_counts[name]),
        }

    return {
        "match": header,
        "num_episodes": len(episodes),
        "avg_episode_length": round(sum(lengths) / len(lengths), 4) if lengths else 0.0,
        "per_agent": per_agent,
    }


def format_summary(summary: dict) -> str:
    lines = []
    m = summary.get("match") or {}
    lines.append(
        f"Game: {m.get('game')} v{m.get('game_version')}  |  "
        f"episodes: {summary['num_episodes']}  |  "
        f"avg length: {summary['avg_episode_length']}"
    )
    lines.append("")
    header = (
        f"{'agent':<22}{'eps':>5}{'mean/hand':>12}{'±ci95':>9}"
        f"{'win%':>8}{'invalid%':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, s in summary["per_agent"].items():
        lines.append(
            f"{name:<22}{s['episodes']:>5}{s['mean_payoff_per_hand']:>12.4f}"
            f"{s['payoff_ci95']:>9.4f}{s['win_rate'] * 100:>7.1f}%"
            f"{s['invalid_action_rate'] * 100:>9.1f}%"
        )
    lines.append("")
    for name, s in summary["per_agent"].items():
        lines.append(f"  {name} actions: {s['action_distribution']}")
    return "\n".join(lines)
