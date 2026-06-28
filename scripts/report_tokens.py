"""Shared helpers for the tokens/dec + $/1K dec leaderboard columns.

tokens/dec = average completion (reasoning) tokens generated per decision.
$/1K dec   = estimated cost per 1,000 decisions = tokens/dec x the model's
             Fireworks serverless decode price (output $/1M tokens).

Closed models (Claude / GPT-5.x) hide their chain-of-thought, so their token
count — and thus cost — is not observable; both columns render as "—".
"""
from __future__ import annotations

from collections import defaultdict

from model_names import strip_coached, output_price

# Header cells (paired) and the explanatory note, so every page reads the same.
TOKEN_HEADERS = "<th>tokens/dec</th><th>$/1K dec</th>"
TOKEN_NOTE = (
    '<div class="note"><b>tokens/dec</b> = avg completion (reasoning) tokens per decision. '
    '<b>$/1K dec</b> = estimated cost per 1,000 decisions = tokens/dec × the model\'s Fireworks '
    'serverless decode price (output $/1M tokens). Both are <b>—</b> for Claude and GPT-5.x, which '
    'hide their chain-of-thought, so their token count (and cost) is not observable here.</div>'
)


def _closed(model: str) -> bool:
    return model.startswith("claude") or model.startswith("gpt-5")


def tokens_from_episodes(episodes) -> dict:
    """Average completion tokens per model over an iterable of episode dicts."""
    tot: dict = defaultdict(int)
    n: dict = defaultdict(int)
    for e in episodes:
        for s in (e.get("steps") or []):
            m = s.get("agent_name")
            if not m:
                continue
            m = strip_coached(m)
            ct = ((s.get("response") or {}).get("metadata") or {}).get("completion_tokens")
            if isinstance(ct, (int, float)):
                tot[m] += ct
                n[m] += 1
    return {m: round(tot[m] / n[m]) for m in tot if n[m]}


def token_cost_cells(model: str, avg) -> str:
    """Two <td> cells (tokens/dec, $/1K dec); "—" for closed models / no price."""
    m = strip_coached(model)
    if _closed(m) or not avg:
        return "<td>—</td><td>—</td>"
    price = output_price(m)
    cost = f"${avg * price / 1e6 * 1000:.2f}" if price else "—"
    return f"<td>{round(avg):,}</td><td>{cost}</td>"
