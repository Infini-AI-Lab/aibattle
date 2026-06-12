"""Estimate the model-generation TRUNCATION ratio across all tournaments.

Truncation isn't logged directly (the OpenAI client keeps only content/reasoning,
not ``finish_reason``). But these are reasoning models given a 16,384-token
budget, and a generation that exhausts that budget mid-chain-of-thought never
emits a final answer — so ``content`` comes back empty, parsing fails, and after
``max_retries`` the move is recorded ``invalid``. Empirically EVERY invalid move
in these runs has the same signature: empty message, retries exhausted,
``has_reasoning=True``. That is the truncation fingerprint.

We report two ratios per model:
  - decision truncation : decisions that ended invalid with empty content +
                          reasoning present (a fully-truncated decision)
  - generation estimate : truncated generations / total generations, where a
                          decision that needed ``attempts`` tries and ended
                          invalid contributes ``attempts`` truncated generations,
                          and a valid decision contributes ``attempts-1`` failed
                          (assumed-truncated) generations + 1 good one.

Reads every <name>_data.json under runs/ tournament dirs; prints a table and
writes reports/truncation_analysis.json.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

# (label, path-to-data-json)
SOURCES = [
    ("connect4", "runs/connect4/connect4_data.json"),
    ("gomoku", "runs/gomoku/gomoku_data.json"),
    ("holdem", "runs/holdem_1hand/tournament_data.json"),
]
REPORT_DIR = "reports"


def _episodes(data):
    for g in data["games"]:
        for e in g["episodes"]:
            yield e


def _blank():
    return {
        "decisions": 0,
        "generations": 0,          # sum of attempts
        "trunc_decisions": 0,      # invalid + empty content + reasoning
        "trunc_generations": 0,    # estimated truncated generations
        "invalid_decisions": 0,
        "invalid_nontrunc": 0,     # invalid but NOT the truncation signature
    }


def analyze(label, path):
    data = json.load(open(path))
    per_model = defaultdict(_blank)
    for e in _episodes(data):
        for s in e["steps"]:
            nm = s["agent_name"]
            md = (s.get("response") or {}).get("metadata", {}) or {}
            msg = (s.get("response") or {}).get("message") or ""
            attempts = int(md.get("attempts", 1))
            has_reasoning = bool(md.get("has_reasoning"))
            invalid = bool(s.get("invalid"))

            st = per_model[nm]
            st["decisions"] += 1
            st["generations"] += attempts

            # Truncation fingerprint: model produced reasoning but no final answer.
            truncated_decision = invalid and not msg.strip() and has_reasoning
            if invalid:
                st["invalid_decisions"] += 1
            if truncated_decision:
                st["trunc_decisions"] += 1
                # all `attempts` generations ran out of budget
                st["trunc_generations"] += attempts
            else:
                if invalid:
                    st["invalid_nontrunc"] += 1
                # a valid (or non-trunc-invalid) decision: the attempts before the
                # last one failed to parse — for reasoning models that is almost
                # always truncation too, so count attempts-1 as truncated.
                st["trunc_generations"] += max(attempts - 1, 0)

    out = {}
    for m, st in per_model.items():
        out[m] = {
            **st,
            "decision_trunc_rate": round(st["trunc_decisions"] / max(st["decisions"], 1), 4),
            "generation_trunc_rate": round(st["trunc_generations"] / max(st["generations"], 1), 4),
            "invalid_rate": round(st["invalid_decisions"] / max(st["decisions"], 1), 4),
        }
    return out


def main():
    results = {}
    for label, path in SOURCES:
        if not os.path.exists(path):
            print(f"skip {label}: no data at {path}")
            continue
        results[label] = analyze(label, path)

    if not results:
        print("No tournament data found.")
        return

    os.makedirs(REPORT_DIR, exist_ok=True)
    json.dump(results, open(os.path.join(REPORT_DIR, "truncation_analysis.json"), "w"),
              indent=2)

    # console table
    all_models = sorted({m for r in results.values() for m in r})
    for label in results:
        r = results[label]
        print(f"\n=== {label} ===")
        print(f"  {'model':<16} {'decisions':>9} {'gens':>7} "
              f"{'dec-trunc%':>10} {'gen-trunc%':>10} {'invalid%':>9} {'nontrunc-inv':>12}")
        for m in sorted(r, key=lambda x: r[x]["decision_trunc_rate"], reverse=True):
            s = r[m]
            print(f"  {m:<16} {s['decisions']:>9} {s['generations']:>7} "
                  f"{s['decision_trunc_rate']*100:>9.2f}% {s['generation_trunc_rate']*100:>9.2f}% "
                  f"{s['invalid_rate']*100:>8.2f}% {s['invalid_nontrunc']:>12}")

    # cross-tournament per-model rollup
    print("\n=== per-model rollup (all tournaments) ===")
    print(f"  {'model':<16} {'decisions':>9} {'gens':>7} {'dec-trunc%':>10} {'gen-trunc%':>10}")
    for m in all_models:
        dec = sum(results[l][m]["decisions"] for l in results if m in results[l])
        gen = sum(results[l][m]["generations"] for l in results if m in results[l])
        td = sum(results[l][m]["trunc_decisions"] for l in results if m in results[l])
        tg = sum(results[l][m]["trunc_generations"] for l in results if m in results[l])
        print(f"  {m:<16} {dec:>9} {gen:>7} {td/max(dec,1)*100:>9.2f}% {tg/max(gen,1)*100:>9.2f}%")

    print(f"\nWrote {os.path.join(REPORT_DIR, 'truncation_analysis.json')}")


if __name__ == "__main__":
    main()
