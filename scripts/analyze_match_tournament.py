"""Analyze the Heads-Up Match-mode tournament.

Reads runs/holdem_match/match_data.json and reports match win rate (the
primary metric), head-to-head grid, and match-shape stats (bust vs max-hands,
avg hands/match, avg final-stack margin). Writes a Chart.js HTML report to
runs/holdem_match/match_report.html and reports/match_tournament_report.html
plus the raw numbers to reports/match_tournament_analysis.json. Styling matches
the board-game tournament report.
"""

from __future__ import annotations

import json
import os
import html as html_lib
import math
import glob
from collections import defaultdict

import poker_behavior as pb
from model_names import strip_coached, display_name, model_cell
from report_tokens import token_cost_cells, TOKEN_HEADERS, TOKEN_NOTE
from elo_util import bradley_terry, elo_key, bootstrap_elo, wld_from_records
from report_theme import BASE_CSS, CHART_SETUP
from report_legends import legend as _legend

# Coached is now the canonical (and only) run set; data lives in per-game folders.
DATA = "runs/holdem_match/match_data.json"
DATA_DIRS = [
    "runs/holdem_match",
    "runs/source_logs/aibattle-logs/*/holdem_match",
]
EP_GLOBS = [
    "runs/holdem_match/*__vs__*/ep*.json",
    "runs/source_logs/aibattle-logs/*/holdem_match/*__vs__*/ep*.json",
]
OUT_HTML = "runs/holdem_match/match_report.html"
REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# Models dropped from the Hold'em reports for incomplete schedules (see main()).
EXCLUDE_HOLDEM = {"gpt-oss-120b"}


def _pair_models(pair: dict) -> set:
    """The two model names a pair is between (robust to missing a/b keys)."""
    if pair.get("a") and pair.get("b"):
        return {strip_coached(pair["a"]), strip_coached(pair["b"])}
    ms = set()
    for e in pair.get("episodes", []):
        ms |= {strip_coached(v) for v in (e.get("seat_assignment") or {}).values()}
    return ms

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# pages include those two files in <head> via NAV_HEAD and the bar is injected
# by JS, so the nav markup lives in one place.
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=30"></script>'

# Page-specific styles that used to ride along with the nav CSS.
EXTRA_CSS = ""

_STYLE = BASE_CSS + """
  /* Prominent dividers for the three top-level sections (Results / Why / More). */
  h2.section { font-size:23px; margin:56px 0 18px; padding-top:16px;
    border-top:3px solid var(--red); color:var(--red); letter-spacing:.01em; }
  h2.section:first-of-type { margin-top:32px; }
  td.hh { font-weight:700; }
  td.hh .rec { display:block; font-weight:400; font-size:11px; color:var(--dim); margin-top:1px; }
  .strategy-intro { background:var(--faint); border:1px solid var(--line);
    border-left:3px solid var(--red); padding:12px 14px; margin:12px 0 18px;
    font-size:13px; line-height:1.55; }
  .strategy-glossary { display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
    gap:10px 16px; margin-top:12px; }
  .strategy-gloss { border-top:1px solid var(--line); padding-top:7px; }
  .strategy-gloss b { display:block; color:var(--fg); }
  .strategy-gloss span { display:block; color:var(--fg); }
  .strategy-gloss em { display:block; color:var(--dim); font-style:normal; font-size:11px; }
  .strategy-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
    gap:18px; align-items:start; }
  .strategy-card { border:1px solid var(--line); background:var(--panel);
    padding:16px; }
  .strategy-head { display:flex; justify-content:space-between; gap:12px;
    align-items:flex-start; margin-bottom:8px; }
  .strategy-head h3 { margin-bottom:4px; }
  .strategy-kpi { text-align:right; font-weight:700; font-size:18px; white-space:nowrap; }
  .strategy-kpi span { display:block; color:var(--dim); font-size:10px; font-weight:400; }
  .strategy-card canvas { max-height:230px; margin:4px 0 10px; }
  .strategy-cases { margin:6px 0 0; padding-left:22px; font-size:12px; line-height:1.45; }
  .strategy-cases li { margin:8px 0; }
  .case-title { font-weight:700; }
  .case-signal { color:var(--fg); margin-top:2px; }
  .case-meta { color:var(--dim); font-size:11px; margin-top:2px; }
  .case-link { display:inline-block; margin-top:5px; border:1px solid var(--line);
    padding:3px 8px; color:#4338ca; text-decoration:none; }
  .case-link:hover { border-color:var(--red); color:var(--fg); }
  .pl-verdict { margin:2px 0 8px; font-size:12.5px; line-height:1.45; color:var(--fg); }
  .pl-attr { margin:0 0 10px; font-size:11px; }
  .pl-row { display:flex; align-items:center; gap:7px; margin:4px 0; }
  .pl-lbl { width:84px; color:var(--dim); white-space:nowrap; }
  .pl-track { position:relative; flex:1; height:13px; background:var(--faint);
    border:1px solid var(--line); border-radius:2px; overflow:hidden; }
  .pl-val { width:40px; text-align:right; font-weight:600; font-variant-numeric:tabular-nums; }
  .metric-profile-layout { margin-top:4px; }
  .metric-profile-layout canvas { width:100% !important; max-height:230px; }
  /* win/loss-outcome bar (replaces the radar): one stacked bar per model */
  .wl-bar { display:flex; height:26px; border:1px solid var(--line); border-radius:3px;
    overflow:hidden; margin:2px 0 8px; background:var(--faint); }
  .wl-bar i { display:block; height:100%; }
  .wl-legend { display:flex; flex-wrap:wrap; gap:3px 12px; font-size:11px; color:var(--dim); }
  .wl-legend i, .wl-key i { display:inline-block; width:10px; height:10px; border-radius:2px;
    margin-right:4px; vertical-align:-1px; }
  .wl-key { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px 18px;
    margin:10px 0; font-size:12px; }
  /* lead-trajectory heatmap: one row per model, 30 hand-blocks coloured by lead% */
  .lt-heat { display:flex; flex-direction:column; gap:2px; margin-top:10px; }
  .lt-row { display:grid; grid-template-columns:188px repeat(30,1fr); gap:1px; align-items:center; }
  .lt-name { display:flex; align-items:center; white-space:nowrap; overflow:hidden;
    text-overflow:ellipsis; padding-right:8px; font-size:12px; }
  .lt-wr { margin-left:auto; padding-left:8px; font-variant-numeric:tabular-nums; }
  .lt-cell { height:20px; border-radius:1px; }
  .lt-h { text-align:center; color:var(--dim); font-size:9px; align-self:end; }
  .lt-head .lt-name { color:var(--dim); font-size:10px; }
  .lt-scale { display:flex; align-items:center; gap:8px; margin:8px 0 2px; font-size:11px; color:var(--dim); }
  .lt-grad { display:inline-block; width:180px; height:12px; border:1px solid var(--line); border-radius:2px;
    background:linear-gradient(to right,#c0392b,#f7f4ee,#1a7f37); }
  .lt-mid { margin-left:4px; }
  @media (max-width:860px) { .lt-row { grid-template-columns:120px repeat(30,1fr); } }
  @media (max-width:860px) { .strategy-grid, .strategy-glossary, .wl-key { grid-template-columns:1fr; } }
"""


def _percentile(vals, q):
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _replay_files() -> set[str]:
    manifest_path = os.path.join(REPORT_DIR, "runs/holdem_match/replays/match/manifest.json")
    try:
        data = json.load(open(manifest_path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {p.get("file") for p in data.get("pairs", []) if p.get("file")}


_KEEP_EP = ("episode", "seat_assignment", "returns", "winner", "winner_name",
            "length", "hands_played", "final_stacks", "stack_diff", "reason",
            "hand_summaries")


def _pair_dirs() -> list[str]:
    dirs = []
    for root in DATA_DIRS:
        dirs.extend(d for d in glob.glob(os.path.join(root, "*__vs__*"))
                    if os.path.isdir(d))
    return sorted(set(dirs))


def _replay_file_for_pair_dir(pair_dir: str, a: str, b: str) -> str:
    base = f"match__{a}__vs__{b}"
    norm = os.path.normpath(pair_dir)
    parts = norm.split(os.sep)
    if len(parts) >= 3 and parts[0] == "runs" and parts[1] == "holdem_match":
        return f"{base}.json"
    wave = "source"
    if "aibattle-logs" in parts:
        idx = parts.index("aibattle-logs")
        if idx + 1 < len(parts):
            wave = parts[idx + 1]
    safe_wave = "".join(c if c.isalnum() or c in "._-" else "_" for c in wave)
    return f"{base}__{safe_wave}.json"


def _load_all_match_data() -> dict:
    """Build the match aggregate from every synced run directory.

    The legacy `runs/holdem_match/match_data.json` only covers one tournament
    wave. The report should cover all synced logs, including the gpt/claude
    source waves under `runs/source_logs/aibattle-logs`.
    """
    models = []
    pairs = []
    max_hands = 0
    starting_stack = 0
    counts = []

    for pd in _pair_dirs():
        a, b = os.path.basename(pd).split("__vs__")
        episodes = []
        for ep_path in sorted(glob.glob(os.path.join(pd, "ep*.json"))):
            try:
                e = strip_coached(json.load(open(ep_path, encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            max_hands = max(max_hands, e.get("max_hands") or 0)
            fs = e.get("final_stacks") or {}
            if fs:
                starting_stack = max(starting_stack, sum(fs.values()) // len(fs))
            episodes.append({k: e[k] for k in _KEEP_EP if k in e})
        if not episodes:
            continue
        a = strip_coached(a); b = strip_coached(b)
        for m in (a, b):
            if m not in models:
                models.append(m)
        counts.append(len(episodes))
        pairs.append({
            "a": a, "b": b, "episodes": episodes,
            "source_dir": pd,
            "replay_file": _replay_file_for_pair_dir(pd, a, b),
        })

    if not pairs and os.path.exists(DATA):
        data = strip_coached(json.load(open(DATA)))
        counts = [len(p.get("episodes", [])) for p in data.get("pairs", [])]
        data["episode_counts"] = counts
        data["episode_count_range"] = [min(counts), max(counts)] if counts else [0, 0]
        data["pair_count"] = len(data.get("pairs", []))
        data["total_matches"] = sum(counts)
        return data

    return {
        "mode": "match",
        "models": models,
        "episodes_per_pair": max(counts) if counts else 0,
        "episode_counts": counts,
        "episode_count_range": [min(counts), max(counts)] if counts else [0, 0],
        "pair_count": len(pairs),
        "total_matches": sum(counts),
        "max_hands": max_hands,
        "starting_stack": starting_stack,
        "pairs": pairs,
    }


def _pair_file(a: str, b: str, available: set[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in available:
        return preferred
    direct = f"match__{a}__vs__{b}.json"
    rev = f"match__{b}__vs__{a}.json"
    if direct in available:
        return direct
    if rev in available:
        return rev
    return None


def _match_strategy(data: dict, models: list[str]) -> dict:
    """Diagnostic match-mode signals.

    Higher radar scores mean more review evidence, not better play.
    """
    available = _replay_files()
    s = {m: {
        "hands": 0, "net": 0.0, "btn_hands": 0, "btn_net": 0.0, "bb_hands": 0, "bb_net": 0.0,
        "big_loss": 0, "big_win": 0, "matches": 0, "wins": 0, "losses": 0, "bust_outs": 0,
        "bust_wins": 0, "lead_opp": 0, "lead_fail": 0, "def_opp": 0, "def_fail": 0,
        "late_net": 0.0, "early_net": 0.0, "late_hands": 0, "early_hands": 0,
        "after_big_loss": 0, "after_big_loss_net": 0.0, "lead_changes": 0,
        "max_drawdown": 0.0, "cases": [], "pairs": set(),
    } for m in models}

    def add_case(model, dimension, title, signal, opponent, episode, hand=None, step=0, pair_file=None):
        href = None
        if pair_file:
            href = f"match_replay.html?pair={pair_file}&match={episode}"
            if hand is not None:
                href += f"&hand={hand}"
            step_param = "end" if hand is not None and step == 0 else step
            href += f"&step={step_param}&cacheBust=18"
        s[model]["cases"].append({
            "dimension": dimension, "title": title, "signal": signal,
            "opponent": opponent, "episode": episode, "hand": hand, "step": step, "href": href,
        })

    for pair in data["pairs"]:
        a = strip_coached(pair["a"]); b = strip_coached(pair["b"])
        pair_file = _pair_file(a, b, available, pair.get("replay_file"))
        for e in pair["episodes"]:
            seat = {k: strip_coached(v) for k, v in e["seat_assignment"].items()}
            inv = {v: k for k, v in seat.items()}
            winner = strip_coached(e.get("winner_name")) if e.get("winner_name") else None
            fs = e.get("final_stacks", {})
            hands = e.get("hand_summaries", [])
            for m in (seat["player_0"], seat["player_1"]):
                if m not in s:
                    continue
                s[m]["matches"] += 1
                s[m]["pairs"].add(pair_file or "")
                if winner == m:
                    s[m]["wins"] += 1
                    if e.get("reason") == "bust":
                        s[m]["bust_wins"] += 1
                else:
                    s[m]["losses"] += 1
                    if e.get("reason") == "bust":
                        s[m]["bust_outs"] += 1

            # Checkpoint: after hand 10, or one third into shorter matches.
            if hands:
                cp_idx = min(len(hands), max(1, min(10, len(hands) // 3 or 1))) - 1
                cp = hands[cp_idx].get("stacks_after", {})
                for m in (seat["player_0"], seat["player_1"]):
                    if m not in s:
                        continue
                    st = cp.get(inv[m])
                    opp_seat = "player_1" if inv[m] == "player_0" else "player_0"
                    opp = seat[opp_seat]
                    if st is not None and st >= 240:
                        s[m]["lead_opp"] += 1
                        if winner != m:
                            s[m]["lead_fail"] += 1
                            add_case(m, "Stack pressure", "lead did not convert",
                                f"ahead {st} chips by hand {cp_idx + 1}, but lost the match",
                                opp, e["episode"], cp_idx + 1, 0, pair_file)
                    if st is not None and st <= 160:
                        s[m]["def_opp"] += 1
                        if winner != m:
                            s[m]["def_fail"] += 1

            prev_delta_by_seat = {}
            stack_path = {seat_name: [200] for seat_name in ("player_0", "player_1")}
            for idx, h in enumerate(hands):
                deltas = h.get("deltas", {})
                button = h.get("button")
                for seat_name, m in seat.items():
                    if m not in s:
                        continue
                    delta = float(deltas.get(seat_name, 0))
                    s[m]["hands"] += 1
                    s[m]["net"] += delta
                    if button == seat_name:
                        s[m]["btn_hands"] += 1; s[m]["btn_net"] += delta
                    else:
                        s[m]["bb_hands"] += 1; s[m]["bb_net"] += delta
                    if delta <= -30:
                        s[m]["big_loss"] += 1
                        opp = seat["player_1" if seat_name == "player_0" else "player_0"]
                        add_case(m, "Volatility control", "large single-hand loss",
                            f"lost {abs(int(delta))} chips in hand {idx + 1}",
                            opp, e["episode"], idx + 1, 0, pair_file)
                    if delta >= 30:
                        s[m]["big_win"] += 1
                    if idx < max(1, len(hands) // 3):
                        s[m]["early_net"] += delta; s[m]["early_hands"] += 1
                    if idx >= max(0, len(hands) * 2 // 3):
                        s[m]["late_net"] += delta; s[m]["late_hands"] += 1
                    prev_delta = prev_delta_by_seat.get(seat_name)
                    if prev_delta is not None and prev_delta <= -25:
                        s[m]["after_big_loss"] += 1
                        s[m]["after_big_loss_net"] += delta
                        if delta < 0:
                            opp = seat["player_1" if seat_name == "player_0" else "player_0"]
                            add_case(m, "Adaptation", "loss followed by another loss",
                                f"after losing {abs(int(prev_delta))}, next hand lost {abs(int(delta))}",
                                opp, e["episode"], idx + 1, 0, pair_file)
                    prev_delta_by_seat[seat_name] = delta
                for seat_name in ("player_0", "player_1"):
                    stack_path[seat_name].append(h.get("stacks_after", {}).get(seat_name, stack_path[seat_name][-1]))
            for seat_name, path in stack_path.items():
                m = seat[seat_name]
                if m not in s or not path:
                    continue
                peak = path[0]; dd = 0.0; last_sign = 0; changes = 0
                for v in path:
                    peak = max(peak, v)
                    dd = max(dd, peak - v)
                    diff = v - 200
                    sign = 1 if diff > 0 else -1 if diff < 0 else 0
                    if sign and last_sign and sign != last_sign:
                        changes += 1
                    if sign:
                        last_sign = sign
                s[m]["max_drawdown"] += dd
                s[m]["lead_changes"] += changes

    for m in models:
        s[m]["pairs"] = sorted(x for x in s[m]["pairs"] if x)
    return s


def _match_verdict(st, fac, lbwin_m, bo_hi, bo_lo):
    """One-line verdict from well-sampled signals; short-stack only if n>=30."""
    mt = st.get("matches", 1) or 1
    bo = st.get("bust_outs", 0) / mt
    dp = (fac.get("by_depth", {}).get("deep") or {}).get("agg")
    sh = (fac.get("by_depth", {}).get("short") or {})
    s, sn = sh.get("agg"), sh.get("n", 0)
    ah = (fac.get("by_lead", {}).get("ahead") or {}).get("agg")
    bh = (fac.get("by_lead", {}).get("behind") or {}).get("agg")
    tag = "Strong" if lbwin_m >= 0.55 else ("Weak" if lbwin_m <= 0.43 else "Mid")
    parts = []
    if dp is not None and dp <= 0.22:
        parts.append("very passive (rarely raises)")
    elif dp is not None and dp >= 0.38:
        parts.append("aggressive")
    if bo >= bo_hi:
        parts.append("high-variance (busts out a lot)")
    elif bo <= bo_lo:
        parts.append("steady (rarely busts out)")
    if ah is not None and bh is not None and bh - ah >= 0.12:
        parts.append("fights back when behind")
    if s is not None and sn >= 30:
        if s >= 0.45:
            parts.append(f"shoves when short ({s*100:.0f}%)")
        elif s <= 0.18:
            parts.append(f"freezes when short ({s*100:.0f}%)")
    return f"<b>{tag}</b>" + (" — " + "; ".join(parts) if parts else "") + "."


def _match_cases_html(report):
    """Replay-linked per-model evidence (kept as a Section 3 deep-dive)."""
    models = [r["model"] for r in report["leaderboard"]]
    stats = report["strategy"]
    cards = []
    for rank, m in enumerate(models, 1):
        seen = set(); picked = []
        for c in sorted(stats[m]["cases"], key=lambda x: 0 if x.get("href") else 1):
            k = (c["dimension"], c["title"], c["opponent"], c["episode"], c.get("hand"))
            if k in seen:
                continue
            seen.add(k); picked.append(c)
            if len(picked) == 3:
                break
        lis = ""
        for c in picked:
            link = (f"<a class='case-link' href=\"{html_lib.escape(c['href'], quote=True)}\">watch replay</a>"
                    if c.get("href") else "<div class='case-meta'>no replay for this pair</div>")
            hand = f" · hand {c.get('hand')}" if c.get("hand") else ""
            lis += (f"<li><div class='case-title'>{html_lib.escape(c['dimension'])}: {html_lib.escape(c['title'])}</div>"
                    f"<div class='case-signal'>{html_lib.escape(c['signal'])}</div>"
                    f"<div class='case-meta'>vs {html_lib.escape(str(c['opponent']))} · match {html_lib.escape(str(c['episode']))}{hand}</div>{link}</li>")
        cards.append(f"<article class='strategy-card'><div class='strategy-head'><div>"
                     f"<h3>{rank}. {model_cell(m)}</h3></div></div>"
                     f"<ol class='strategy-cases'>{lis}</ol></article>")
    return ("<div class=\"note\">Specific match moments behind each model's profile, linked into the "
            "replay viewer.</div><div class=\"strategy-grid\">" + "".join(cards) + "</div>")


def _match_strategy_html(report):
    """Section 2 cards: verdict + a win/loss-outcome bar (how it wins AND loses).

    Match poker is a different, more layered game than 1-Hand, so instead of a
    radar we decompose every match into its four outcomes — won by busting the
    opponent, won at the cap (out-chipped them), lost at the cap (ground down),
    lost by busting out (blown up) — which is the most direct "why win / why
    lose" view: it separates grinders from gamblers and the ground-down from the
    blown-up.
    """
    models = [r["model"] for r in report["leaderboard"]]
    stats = report["strategy"]
    lbwin = {r["model"]: r["win_rate"] for r in report["leaderboard"]}
    try:
        F = json.load(open(os.path.join(REPORT_DIR, "match_factors.json")))
    except (OSError, json.JSONDecodeError):
        F = {"win_type": {}, "by_depth": {}, "by_lead": {}}

    def rate(n, d): return n / d if d else 0.0
    bo_rates = [rate(stats[m]["bust_outs"], stats[m]["matches"]) for m in models]
    sb = sorted(bo_rates); bo_hi = sb[min(len(sb)-1, int(.75*len(sb)))]; bo_lo = sb[int(.25*len(sb))]

    # Four match outcomes (+ rare draw), in left-to-right order: wins then losses,
    # so the green→red boundary sits exactly at the model's win rate.
    SEGS = [
        ("bust_win", "won by bust",            "#1a7f37", "busted the opponent"),
        ("cap_win",  "won at cap",             "#5fae6a", "led on chips at the 30-hand cap"),
        ("draw",     "draw",                   "#c9c4b8", "tied at the cap"),
        ("lost_cap", "lost at cap (cap-lose)", "#d99a2b", "behind on chips at the cap — slowly ground down"),
        ("lost_bust","lost · bust out", "#c0392b", "busted out — lost the whole stack"),
    ]
    cards = []
    for rank, m in enumerate(models, 1):
        st = stats[m]
        w = F.get("win_type", {}).get(m, {}) or {}
        mt = w.get("matches") or st.get("matches") or 1
        segs_html = "".join(
            f"<i style='width:{w.get(k,0)/mt*100:.2f}%;background:{col}' "
            f"title='{lbl}: {w.get(k,0)} ({w.get(k,0)/mt*100:.0f}%) — {desc}'></i>"
            for k, lbl, col, desc in SEGS if w.get(k, 0))
        def pc(k): return w.get(k, 0) / mt * 100
        legend = (
            f"<div class='wl-legend'>"
            f"<span><i style='background:#1a7f37'></i>bust-win {pc('bust_win'):.0f}%</span>"
            f"<span><i style='background:#5fae6a'></i>cap-win {pc('cap_win'):.0f}%</span>"
            f"<span><i style='background:#d99a2b'></i>cap-lose {pc('lost_cap'):.0f}%</span>"
            f"<span><i style='background:#c0392b'></i>bust-out {pc('lost_bust'):.0f}%</span>"
            f"</div>")
        fac = {"by_depth": F["by_depth"].get(m, {}), "by_lead": F["by_lead"].get(m, {})}
        verdict = _match_verdict(st, fac, lbwin[m], bo_hi, bo_lo)
        wr = lbwin[m]
        cards.append(f"""
    <article class="strategy-card">
      <div class="strategy-head"><div><h3>{rank}. {model_cell(m)}</h3></div>
        <div class="strategy-kpi {'pos' if wr >= 0.5 else 'neg'}">{wr*100:.0f}%<span>match win</span></div></div>
      <div class="pl-verdict">{verdict}</div>
      <div class="wl-bar">{segs_html}</div>
      {legend}
    </article>""")

    html = f"""
  <div class="strategy-intro">
    <b>How each model wins — and how it loses.</b> Every match ends one of four ways; this bar
    splits each model's matches into them, wins on the left, losses on the right (so the green→red
    edge is its win rate):
    <div class="wl-key">
      <span><i style="background:#1a7f37"></i><b>bust-win</b> — busted the opponent</span>
      <span><i style="background:#5fae6a"></i><b>cap-win</b> — out-chipped them over 30 hands</span>
      <span><i style="background:#d99a2b"></i><b>cap-lose</b> — behind on chips at the cap (ground down)</span>
      <span><i style="background:#c0392b"></i><b>bust-out</b> — busted out</span>
    </div>
    A wide <span style="color:#5fae6a;font-weight:700">cap-win</span> block = a grinder that out-chips
    opponents; a wide <span style="color:#d99a2b;font-weight:700">cap-lose</span> block = it gets
    slowly ground down; a wide <span style="color:#c0392b;font-weight:700">bust-out</span>
    block = it busts out a lot (high variance).
  </div>
  <div class="strategy-grid">{''.join(cards)}</div>"""
    return html, ""


def _lead_traj_html(report):
    """Heatmap: each model a row of 30 hand-blocks, coloured by its ahead-on-chips
    share after that hand (green = ahead, red = behind). Cleaner than 12 lines."""
    try:
        T = json.load(open(os.path.join(REPORT_DIR, "match_factors.json"))).get("lead_trajectory", {})
    except (OSError, json.JSONDecodeError):
        T = {}
    rows_data = [r for r in report["leaderboard"] if T.get(r["model"])]
    if not rows_data:
        return "", ""

    def heat(v):
        if v is None:
            return "#efece6"
        d = max(-20.0, min(20.0, v - 50.0)); t = abs(d) / 20.0
        a = (247, 244, 238)                       # faint base at 50%
        b = (26, 127, 55) if d >= 0 else (192, 57, 43)
        r, g, bl = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
        return f"#{r:02x}{g:02x}{bl:02x}"

    head = "".join(f"<span class='lt-h'>{h if h in (1,5,10,15,20,25,30) else ''}</span>"
                   for h in range(1, 31))
    rows = [f"<div class='lt-row lt-head'><span class='lt-name'>hand →</span>{head}</div>"]
    for r in rows_data:
        m = r["model"]
        cells = "".join(
            f"<span class='lt-cell' style='background:{heat(v)}' "
            f"title='hand {i+1}: ahead in {v:.0f}% of matches'></span>"
            for i, v in enumerate(T[m]))
        rows.append(f"<div class='lt-row'><span class='lt-name'>{model_cell(m)}"
                    f"<b class='lt-wr'>{r['win_rate']*100:.0f}%</b></span>{cells}</div>")
    html = f"""
  <h2>📊 Lead trajectory <span class="note">(each row = a model; 30 blocks = hands 1→30; block colour =
  share of matches it is <span style="color:#1a7f37;font-weight:700">ahead</span> /
  <span style="color:#c0392b;font-weight:700">behind</span> on chips after that hand)</span></h2>
  <div class="callout">A row that stays <span style="color:#1a7f37;font-weight:700">green</span> across =
  wire-to-wire leader; green that reddens left→right = front-runs then gets ground down; green in the
  middle that <span style="color:#c0392b;font-weight:700">reddens at the end</span> = builds a lead but
  can't close it; all <span style="color:#c0392b;font-weight:700">red</span> = behind the whole match.
  The last block ≈ the model's match win rate.</div>
  <div class="lt-scale"><span>behind ≤35%</span><i class="lt-grad"></i><span>ahead ≥65%</span>
    <span class="lt-mid">50% = even</span></div>
  <div class="lt-heat">{''.join(rows)}</div>"""
    return html, ""


def analyze(data: dict) -> dict:
    models = data["models"]
    played = defaultdict(int); won = defaultdict(int); drew = defaultdict(int)
    stack_margin = defaultdict(float); hands = defaultdict(int); busts = defaultdict(int)
    h2h = {a: {b: 0 for b in models} for a in models}
    h2h_played = {a: {b: 0 for b in models} for a in models}
    elo_records = []  # per-match (a, b, result) for the Elo bootstrap

    for pair in data["pairs"]:
        for e in pair["episodes"]:
            seat = e["seat_assignment"]
            a, b = seat["player_0"], seat["player_1"]
            wname = e.get("winner_name")
            fs = e.get("final_stacks", {})
            for p in ("player_0", "player_1"):
                played[seat[p]] += 1
                hands[seat[p]] += e.get("hands_played", 0)
            if wname is None:
                drew[a] += 1; drew[b] += 1
                elo_records.append((a, b, 0))
            else:
                won[wname] += 1
                loser = b if wname == a else a
                h2h[wname][loser] += 1
                elo_records.append((a, b, 1 if wname == a else -1))
                if fs:
                    stack_margin[wname] += abs(fs.get("player_0", 0) - fs.get("player_1", 0))
                if e.get("reason") == "bust":
                    busts[loser] += 1
            h2h_played[a][b] += 1; h2h_played[b][a] += 1

    # Match mode is win-or-lose by design (chips don't carry meaning past the
    # match outcome), so the Elo is a Bradley-Terry fit over match W/L/D —
    # opponent-adjusted, fair when models faced different opponents.
    wld = {a: {b: (h2h[a][b], h2h[b][a],
                   h2h_played[a][b] - h2h[a][b] - h2h[b][a])
               for b in models if b != a} for a in models}
    _, elo = bradley_terry(models, wld)
    elo_ci = bootstrap_elo(models, elo_records, lambda s: wld_from_records(models, s))

    rows = []
    for m in models:
        n = played[m] or 1
        rows.append({
            "model": m, "matches": played[m], "elo": elo[m], "elo_sd": elo_ci[m]["sd"],
            "win_rate": round(won[m] / n, 3), "wins": won[m], "draws": drew[m],
            "busted_out_rate": round(busts[m] / n, 3),
            "avg_hands_per_match": round(hands[m] / n, 1),
            "avg_win_margin": round(stack_margin[m] / (won[m] or 1), 1),
        })
    rows.sort(key=lambda r: (elo_key(elo, r["model"]), r["win_rate"]), reverse=True)
    return {"models": models, "max_hands": data.get("max_hands"),
            "episodes_per_pair": data.get("episodes_per_pair"),
            "episode_count_range": data.get("episode_count_range"),
            "pair_count": data.get("pair_count") or len(data.get("pairs", [])),
            "total_matches": data.get("total_matches") or sum(len(p.get("episodes", [])) for p in data.get("pairs", [])),
            "elo": elo,
            "leaderboard": rows, "h2h_wins": h2h, "h2h_played": h2h_played}


def _match_decision_quality_html(rep: dict) -> str:
    """Experimental, equity-based block (reads match_decision_quality.json):
    ① is the aggression backed by cards, ② clear all-in blunders, ③ opponent
    adaptation. Equity is vs a random hand (opponent range unknown), so these
    are gross, range-free proxies — labelled experimental on the page."""
    try:
        D = json.load(open(os.path.join(REPORT_DIR, "match_decision_quality.json")))
    except (OSError, json.JSONDecodeError):
        return ""
    order = [r["model"] for r in rep["leaderboard"]]
    wr = {r["model"]: r["win_rate"] for r in rep["leaderboard"]}
    fe, bl, ad = D.get("fire_equity", {}), D.get("blunders", {}), D.get("adaptation", {})

    def corr(vals):
        pts = [(v, wr[m]) for m, v in zip(order, vals) if v is not None]
        if len(pts) < 3:
            return "—"
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; n = len(xs)
        mx = sum(xs) / n; my = sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** .5; vy = sum((b - my) ** 2 for b in ys) ** .5
        return f"{cov/(vx*vy):+.2f}" if vx and vy else "—"

    def pc(v, fmt="{:.0%}"):
        return fmt.format(v) if v is not None else "—"

    r1 = "".join(
        f"<tr><td class='model'>{model_cell(m)}</td>"
        f"<td>{pc((fe.get(m,{}).get('mix') or {}).get('lt40'))}</td>"
        f"<td>{pc(fe.get(m,{}).get('bluff_success'))}</td>"
        f"<td>{pc(fe.get(m,{}).get('fire_eq'))}</td>"
        f"<td class='small'>{fe.get(m,{}).get('n_agg','—')}</td></tr>" for m in order)

    return f"""
  <h2>🧪 Decision quality <span class="note">(experimental — uses each model's hole-card
    <b>equity vs a random hand</b>; the opponent's range is never known, so read this as a gross,
    range-free signal, not exact EV)</span></h2>

  <h3>Does it bluff, or only bet strong hands?</h3>
  <div class="note"><b>bluff rate</b> = share of its bets/raises made with weak cards (&lt;40% equity);
    <b>bluff success</b> = of those bluffs, how often the opponent actually folds; <b>avg equity when
    betting</b> = how strong its cards are, on average, when it fires. A model that <b>only bets strong
    hands</b> (low bluff rate, high avg equity) is predictable and easy to fold to; <b>mixing in
    bluffs</b> is a sign of skill.
    <span class="small">corr with win%: <b>bluff rate {corr([(fe.get(m,{}).get('mix') or {}).get('lt40') for m in order])}</b>
    (more bluffing ↔ winning), avg-equity-when-betting {corr([fe.get(m,{}).get('fire_eq') for m in order])}
    (only-bet-strong ↔ losing); bluff success {corr([fe.get(m,{}).get('bluff_success') for m in order])}
    (not a skill signal).</span></div>
  <table>
    <tr><th class='model'>model</th><th>bluff rate (&lt;40% eq)</th><th>bluff success</th>
        <th>avg equity when betting</th><th>n</th></tr>{r1}
  </table>
"""


def _match_factors_html(rep: dict) -> str:
    """Match-specific "why win/lose" block: how matches resolve (bust vs cap),
    aggression by stack depth (push/fold gear-change), and ahead-vs-behind."""
    path = os.path.join(REPORT_DIR, "match_factors.json")
    try:
        F = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return ('  <div class="note">Match factors not generated yet — run '
                '<code>python3 scripts/analyze_match_factors.py</code>.</div>')
    win, depth, lead = F["win_type"], F["by_depth"], F["by_lead"]
    order = [r["model"] for r in rep["leaderboard"]]
    lbwin = {r["model"]: r["win_rate"] for r in rep["leaderboard"]}
    MIN_N = 30   # below this a stack-depth cell is greyed out (too few decisions)

    def cell(d):
        """Heat cell for an {agg, n} bucket; greyed (not green) when n < MIN_N."""
        v, n = d.get("agg"), d.get("n", 0)
        if v is None:
            return "<td>—</td>"
        sub = f"<div class='small'>n={n}</div>"
        if n < MIN_N:
            return (f"<td style='background:var(--faint);color:var(--dim)' "
                    f"title='small sample — read as suggestive'>{v*100:.0f}%{sub}</td>")
        a = min(0.62, max(0.0, v * 0.62))
        return f"<td style='background:rgba(26,127,55,{a:.2f})'>{v*100:.0f}%{sub}</td>"

    rows2 = ""
    for m in order:
        d = depth[m]
        rows2 += (f"<tr><td class='model'>{model_cell(m)}</td>"
                  f"{cell(d['deep'])}{cell(d['mid'])}{cell(d['short'])}</tr>")
    def shift(a, b, good):
        """A 'ahead% → behind% (Δ)' cell; Δ green when it shifts toward fighting
        (good='up' = more aggressive when behind; good='down' = folds less)."""
        if a is None or b is None:
            return "<td>—</td>"
        d = (b - a) * 100
        fights = (d > 0) if good == "up" else (d < 0)
        col = ("#1a7f37" if fights and abs(d) >= 5
               else ("#c0392b" if (not fights) and abs(d) >= 5 else "var(--dim)"))
        return (f"<td>{a*100:.0f}% → {b*100:.0f}% "
                f"<b style='color:{col}'>({d:+.0f})</b></td>")
    rows3 = ""
    for m in order:
        a, b = lead[m]["ahead"], lead[m]["behind"]
        rows3 += (f"<tr><td class='model'>{model_cell(m)}</td>"
                  f"{shift(a.get('agg'), b.get('agg'), 'up')}"
                  f"{shift(a.get('fold'), b.get('fold'), 'down')}</tr>")

    return f"""
  <div class="strategy-intro">
    <b>A 30-hand match is usually won by <i>leading at the hand cap</i>, not by busting the
    opponent</b> — so the edge is chip management: fighting back when behind, gear-changing as stacks
    get short, and not blowing up a whole match in one hand. Note a real limitation of this fixed
    dataset: <b>the strong models rarely reach a short stack at all</b> (they protect their chips), so
    their short-stack columns are based on very few decisions — greyed cells (n&lt;{30}) are
    suggestive, not conclusive. We can't add more matches, so we read those honestly.
  </div>
  <h2>🪜 Aggression by stack depth <span class="note">(does it push/fold when short?)</span></h2>
  <div class="note"><b>Aggression</b> = (bet+raise+all-in) ÷ (bet+raise+all-in+call+check), folds
    excluded — bucketed by effective stack:
    <b>deep</b> ≥40bb · <b>mid</b> 15–40bb · <b>short</b> &lt;15bb (push-fold territory). Greener =
    more aggressive; <b>greyed cells have n&lt;30</b> (too few decisions — read as suggestive). The
    <b>n</b> on the short column doubles as an exposure signal: the top models barely appear there
    because they rarely get short-stacked.</div>
  <table>
    <tr><th class='model'>model</th><th>deep (≥40bb)</th><th>mid</th><th>short (&lt;15bb)</th></tr>
    {rows2}
  </table>
  <h2>⚖️ Gear-shift: ahead vs behind <span class="note">(does it change how it plays when losing?)</span></h2>
  <div class="note">Well sampled (every decision). Each cell is <b>when ahead → when behind (change)</b>.
    <b>Aggression</b> = (bet+raise+all-in) ÷ (bet+raise+all-in+call+check), folds excluded;
    <b>fold-to-bet</b> = how often it folds when facing a bet.
    Strong players <b>shift gears when behind</b> — they <b>raise more</b> (aggression ↑) and
    <b>fold less to bets</b> (fold-to-bet ↓), i.e. they fight for pots; the change is
    <span style="color:#1a7f37;font-weight:700">green</span> when it shifts toward fighting,
    <span style="color:#c0392b;font-weight:700">red</span> when it backs off. Weak players show
    <span style="color:var(--dim)">≈0</span> — they play the same whether winning or losing.</div>
  <table>
    <tr><th class='model'>model</th><th>aggression (ahead → behind)</th>
        <th>fold-to-bet (ahead → behind)</th></tr>
    {rows3}
  </table>
"""


def render_html(rep: dict, beh: dict) -> str:
    models = rep["models"]; lb = rep["leaderboard"]
    labels = [r["model"] for r in lb]          # slugs — key behavior stats / colors
    disp_labels = [display_name(m) for m in labels]   # official names for chart axis
    winpct = [round(r["win_rate"] * 100, 1) for r in lb]
    wincols = pb.colors_for(labels)
    beh_html = pb.profile_table(beh, labels) + pb.behavior_charts(beh, labels)
    factors_html = _match_factors_html(rep)
    dq_html = _match_decision_quality_html(rep)
    strategy_html, strategy_js = _match_strategy_html(rep)
    traj_html, traj_js = _lead_traj_html(rep)
    replay_btn = ('<a class="replaybtn" href="match_replay.html?cacheBust=19">'
                  '🎬 Watch featured replays →</a>')
    ep_range = rep.get("episode_count_range") or [rep["episodes_per_pair"], rep["episodes_per_pair"]]
    if ep_range[0] == ep_range[1]:
        match_count_label = f"{ep_range[1]} matches/pair"
    else:
        match_count_label = f"{ep_range[0]}-{ep_range[1]} matches/pair"

    trows = ""
    for i, r in enumerate(lb, 1):
        if r.get("elo") is None:
            elo_disp = "—"
        elif r.get("elo_sd") is not None:
            elo_disp = f"{r['elo']}<div class='small'>±{r['elo_sd']:.0f}</div>"
        else:
            elo_disp = str(r["elo"])
        trows += (f"<tr><td>{i}</td><td class='model'>{model_cell(r['model'])}</td>"
                  f"<td><b>{elo_disp}</b></td>"
                  f"<td>{r['win_rate']*100:.0f}%</td><td>{r['wins']}/{r['matches']}</td>"
                  f"<td>{r['draws']}</td><td>{r['busted_out_rate']*100:.0f}%</td>"
                  f"<td>{r['avg_hands_per_match']}</td><td>{r['avg_win_margin']}</td>"
                  f"<td>{r['matches']}</td>"
                  f"{token_cost_cells(r['model'], beh.get(r['model'], {}).get('avg_tokens'))}</tr>")
    hh_order = [r["model"] for r in rep["leaderboard"]]   # ranked (Elo) order
    head = "".join(f"<th>{display_name(m)}</th>" for m in hh_order)
    grid = ""
    for a in hh_order:
        cells = ""
        for b in hh_order:
            if a == b:
                cells += "<td class='diag'>—</td>"
                continue
            w = rep['h2h_wins'][a][b]; pl = rep['h2h_played'][a][b]
            if not pl:
                cells += "<td class='hh'>—</td>"
                continue
            pct = 100 * w / pl
            # Diverging red→green heatmap centred on 50% (an even split shows no
            # tint); alpha grows with the distance from even so lopsided cells pop.
            alpha = round(0.6 * abs(pct - 50) / 50, 3)
            rgb = "34,197,94" if pct >= 50 else "244,63,94"
            cells += (f"<td class='hh' style='background:rgba({rgb},{alpha})'>"
                      f"{pct:.0f}%<span class='rec'>{w}/{pl}</span></td>")
        grid += f"<tr><td class='model'>{model_cell(a)}</td>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Battle Arena — Hold'em Match Mode</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}<style>{EXTRA_CSS}{_STYLE}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/holdem/match<span class="cursor"></span></h1>
  <div class="sub">🃏 Hold'em Match · Heads-up · {match_count_label} · {rep['pair_count']} pair logs · {rep['total_matches']} total matches · up to {rep['max_hands']} hands/match · stacks carried, match-level winner · primary metric: match win rate</div>
  {replay_btn}
  <div class="rules">
    <h3>Setup — Hold'em Match</h3>
    Standard heads-up No-Limit
    <a href="https://en.wikipedia.org/wiki/Texas_hold_%27em" target="_blank" rel="noopener">Texas Hold'em</a>
    (full rules on Wikipedia); the difference from 1-Hand is that here a whole
    <b>sit-and-go match</b> is the unit, not a single hand:
    <ul>
      <li><b>Heads-up sit-and-go:</b> both start with <b>200 chips</b> (blinds
        <b>1 / 2</b>) and play until one is busted, or until a cap of <b>up to
        {rep['max_hands']} hands</b>.</li>
      <li><b>Stacks carry across hands</b> within a match — winning chips early creates a
        real lead, so position and stack pressure matter.</li>
      <li><b>{match_count_label}</b>, seats swapped where available; the
        <b>match winner</b> is whoever busts the other (or leads at the cap).</li>
    </ul>
    <div class="seq">Win-or-lose by design — chips don't count past the match outcome —
    so the <b>Elo rates match wins/losses</b>, opponent-adjusted. Match win rate is the
    headline metric.</div>
    <div class="seq"><b>What the model sees each turn:</b> the match score (which hand of the cap,
    and each side's chips), its own two hole cards, the community board, the pot and both stacks,
    its position, the bet it faces, the legal actions, and the action history — never the
    opponent's cards.</div>
  </div>
  <h2 class="section">1 · Results — who won</h2>
  <h3>Leaderboard <span class="note">(ranked by Elo; raw metrics kept for reference)</span></h3>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>Elo</th><th>win%</th><th>wins/matches</th>
        <th>draws</th><th>bust-out%</th><th>hands/match</th><th>avg win margin</th><th>matches</th>{TOKEN_HEADERS}</tr>
    {trows}
  </table>
  {_legend('match')}
  {TOKEN_NOTE}
  <div class="note"><b>Elo</b> = Bradley-Terry rating (field mean 1500) over match win/loss results.
    Match mode is win-or-lose — chips don't count past who took the match — so the rating uses match
    outcomes only, opponent-adjusted. ± is one bootstrap SD (resampling matches 300×); ratings within
    ±1 of each other are a statistical tie. win% and the rest are raw, unadjusted metrics.</div>
  <h3>Head-to-head <span class="note">(row's match win % vs column — green = winning, red = losing; raw record below)</span></h3>
  <table><tr><th class='model'></th>{head}</tr>{grid}</table>
  <h2 class="section">2 · Why — what makes a model win or lose matches</h2>
  {strategy_html}
  {traj_html}
  <h2 class="section">3 · Analysis</h2>
  {factors_html}
  {dq_html}
  {beh_html}
  <script>
  {strategy_js}
  {traj_js}
  </script>
</div></body></html>"""


def main():
    data = _load_all_match_data()
    # Drop GPT-OSS from the Hold'em reports: its schedule is incomplete (it never
    # played GPT 5.5 / 5.4), which inflates its raw win rate. Filtering at the data
    # layer means every other model's win rate / Elo / h2h / strategy / behaviour is
    # recomputed from the remaining, gpt-oss-free games. (runs/ data is untouched.)
    data["models"] = [m for m in data["models"] if m not in EXCLUDE_HOLDEM]
    data["pairs"] = [p for p in data["pairs"]
                     if not (EXCLUDE_HOLDEM & _pair_models(p))]
    rep = analyze(data)
    rep["strategy"] = _match_strategy(data, rep["models"])
    beh = pb.behavior(EP_GLOBS, "match_hand", rep["models"], exclude=EXCLUDE_HOLDEM)
    rep["behavior"] = beh
    html = render_html(rep, beh)
    os.makedirs(REPORT_DIR, exist_ok=True)
    for path in (OUT_HTML, os.path.join(REPORT_DIR, "match_tournament_report.html")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    # The per-model strategy "cases" (replay-linked per-hand evidence) are already
    # baked into the HTML above; drop them from the JSON dump so the committed file
    # stays small (they otherwise balloon it to ~1.5 MB).
    for s in rep.get("strategy", {}).values():
        s.pop("cases", None)
        s.pop("pairs", None)
    json.dump(rep, open(os.path.join(REPORT_DIR, "match_tournament_analysis.json"), "w"),
              indent=2)
    print(f"Wrote {OUT_HTML} and {REPORT_DIR}/match_tournament_report.html\n")
    ep_range = rep.get("episode_count_range") or [rep["episodes_per_pair"], rep["episodes_per_pair"]]
    ep_label = str(ep_range[1]) if ep_range[0] == ep_range[1] else f"{ep_range[0]}-{ep_range[1]}"
    print(f"=== Match Mode ({ep_label}/pair, {rep['pair_count']} pair logs, "
          f"{rep['total_matches']} matches, {rep['max_hands']} hands) ===")
    print(f"{'model':<18} win%   wins      bust%  hands/m  margin")
    for r in rep["leaderboard"]:
        print(f"{r['model']:<18} {r['win_rate']*100:>3.0f}%  {r['wins']:>3}/{r['matches']:<3}  "
              f"{r['busted_out_rate']*100:>4.0f}%  {r['avg_hands_per_match']:>6}  {r['avg_win_margin']:>6}")


if __name__ == "__main__":
    main()
