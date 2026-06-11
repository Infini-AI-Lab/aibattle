"""Token-usage + truncation report for a tournament run (any game).

Usage:
    python scripts/analyze_tokens.py runs/connect4
    python scripts/analyze_tokens.py runs/holdem_1hand
    python scripts/analyze_tokens.py runs/kuhn_poker

Reads per-episode files and writes a board-style HTML report
(<run_dir>/token_report.html and reports/<name>_tokens.html) plus a console
table: per-model output-token distribution (exact when logged, else estimated
~chars/4) and truncation rate (finish_reason == "length").
"""

from __future__ import annotations

import json
import os
import sys

import eval_stats

REPORT_DIR = "reports"

# Shared client-side navbar (reports/nav.css + nav.js); injected by JS.
NAV_HEAD = '<link rel="stylesheet" href="nav.css"><script defer src="nav.js"></script>'

_STYLE = """
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1117;color:#e6e6e6;}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 22px 80px;}
  h1{font-size:23px;} .sub{color:#8b93a7;}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px;}
  th,td{padding:6px 8px;text-align:center;border-bottom:1px solid #20242e;}
  th{color:#9aa3b5;} td.model,th.model{text-align:left;font-weight:600;color:#cdd6f4;}
  .warn{color:#f87171;} .ok{color:#4ade80;}
"""


def render(name: str, rows: list) -> str:
    exact = all(r["exact"] for r in rows) if rows else False
    src = "exact (logged completion_tokens)" if exact else "estimated (~chars/4; older run without token logging)"
    trs = ""
    for r in rows:
        tr = r["trunc_rate"] * 100
        cls = "warn" if tr > 0 else "ok"
        trs += (f"<tr><td class='model'>{r['model']}</td><td>{r['decisions']}</td>"
                f"<td class='{cls}'>{tr:.2f}%</td><td>{r['tok_mean']}</td>"
                f"<td>{r['tok_p50']}</td><td>{r['tok_p90']}</td><td>{r['tok_p99']}</td>"
                f"<td>{r['tok_max']}</td></tr>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Token usage — {name}</title>{NAV_HEAD}<style>{_STYLE}</style></head>
<body><div class="wrap">
<h1>🪙 Token usage &amp; truncation — {name}</h1>
<div class="sub">Output tokens per decision (reasoning + answer). Source: {src}.
Truncation = finish_reason "length".</div>
<table><tr><th class='model'>model</th><th>decisions</th><th>truncated%</th>
<th>mean</th><th>p50</th><th>p90</th><th>p99</th><th>max</th></tr>{trs}</table>
</div></body></html>"""


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/connect4"
    name = os.path.basename(run_dir.rstrip("/"))
    per = eval_stats.collect(run_dir)
    rows = eval_stats.rows(per)
    if not rows:
        print(f"No per-episode files under {run_dir}/*/ep*.json")
        return
    html = render(name, rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    for p in (os.path.join(run_dir, "token_report.html"),
              os.path.join(REPORT_DIR, f"{name}_tokens.html")):
        open(p, "w", encoding="utf-8").write(html)
    json.dump(rows, open(os.path.join(REPORT_DIR, f"{name}_tokens.json"), "w"), indent=2)
    exact = all(r["exact"] for r in rows)
    print(f"Wrote {run_dir}/token_report.html and {REPORT_DIR}/{name}_tokens.html")
    print(f"token source: {'EXACT' if exact else 'ESTIMATED (~chars/4)'}\n")
    print(f"{'model':<18}{'dec':>5}{'trunc%':>8}{'mean':>7}{'p50':>7}{'p90':>7}{'p99':>8}{'max':>8}")
    for r in rows:
        print(f"{r['model']:<18}{r['decisions']:>5}{r['trunc_rate']*100:>7.2f}%"
              f"{r['tok_mean']:>7}{r['tok_p50']:>7}{r['tok_p90']:>7}{r['tok_p99']:>8}{r['tok_max']:>8}")


if __name__ == "__main__":
    main()
