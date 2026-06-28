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

# The site navbar is a shared client-side component (reports/nav.css + nav.js);
# pages include those two files in <head> via NAV_HEAD and the bar is injected
# by JS, so the nav markup lives in one place.
NAV_HEAD = '<link rel="stylesheet" href="nav.css?v=5"><script defer src="nav.js?v=29"></script>'

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
    padding:14px 16px; margin:8px 0 18px; }
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
  @media (max-width:860px) { .strategy-grid, .strategy-glossary { grid-template-columns:1fr; } }
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


def _match_strategy_html(report: dict) -> tuple[str, str]:
    models = [r["model"] for r in report["leaderboard"]]
    stats = report["strategy"]
    dims = [
        ("pressure", "Stack pressure"),
        ("recovery", "Recovery"),
        ("volatility", "Volatility control"),
        ("position", "Position use"),
        ("pacing", "Match pacing"),
        ("adapt", "Adaptation"),
    ]
    dim_help = {
        "Stack pressure": {
            "meaning": "Does the agent convert a chip lead into match wins instead of giving the lead back?",
            "signals": "lead at checkpoint but lost, low bust-win share, small win margin",
        },
        "Recovery": {
            "meaning": "Does it stabilize after falling behind, or does an early deficit become deterministic loss?",
            "signals": "low comeback rate from <=160 chips, poor deficit survival, high bust-out share",
        },
        "Volatility control": {
            "meaning": "Does it avoid one-hand collapses that erase a whole match?",
            "signals": "large losing hands, high max drawdown, bust-outs, high big-loss rate",
        },
        "Position use": {
            "meaning": "Does it understand that BTN/SB should usually be the profitable, initiative seat heads-up?",
            "signals": "negative button EV, BB outperforming button, weak button-vs-BB gap",
        },
        "Match pacing": {
            "meaning": "Does the agent close when ahead and avoid late-match decay?",
            "signals": "late net worse than early net, many max-hand wins/losses, low bust conversion",
        },
        "Adaptation": {
            "meaning": "Does it change after the previous hand, or keep following a template through momentum shifts?",
            "signals": "negative next-hand response after big losses, frequent lead changes, repeated losses",
        },
    }

    def rate(num, den):
        return num / den if den else 0.0

    vals = {
        "big_loss": [rate(stats[m]["big_loss"], stats[m]["hands"]) for m in models],
        "bust_out": [rate(stats[m]["bust_outs"], stats[m]["matches"]) for m in models],
        "lead_fail": [rate(stats[m]["lead_fail"], stats[m]["lead_opp"]) for m in models],
        "def_fail": [rate(stats[m]["def_fail"], stats[m]["def_opp"]) for m in models],
        "drawdown": [rate(stats[m]["max_drawdown"], stats[m]["matches"]) for m in models],
    }
    t = {k: _percentile(v, .75) for k, v in vals.items()}
    t_low = {"bust_win": _percentile([rate(stats[m]["bust_wins"], stats[m]["wins"]) for m in models], .25)}

    cards = []
    chart_payload = []
    for rank, m in enumerate(models, 1):
        st = stats[m]
        signals = {k: [] for k, _ in dims}
        hands = st["hands"] or 1
        matches = st["matches"] or 1
        bb_hand = st["net"] / hands
        btn_ev = st["btn_net"] / (st["btn_hands"] or 1)
        bb_ev = st["bb_net"] / (st["bb_hands"] or 1)
        lead_fail = rate(st["lead_fail"], st["lead_opp"])
        def_fail = rate(st["def_fail"], st["def_opp"])
        big_loss_rate = rate(st["big_loss"], st["hands"])
        bust_out = rate(st["bust_outs"], st["matches"])
        bust_win = rate(st["bust_wins"], st["wins"])
        drawdown = rate(st["max_drawdown"], st["matches"])
        early = st["early_net"] / (st["early_hands"] or 1)
        late = st["late_net"] / (st["late_hands"] or 1)
        response = st["after_big_loss_net"] / (st["after_big_loss"] or 1)
        lead_changes = st["lead_changes"] / matches

        def add(key, cond, msg):
            if cond:
                signals[key].append(msg)

        add("pressure", st["lead_opp"] >= 2 and lead_fail >= t["lead_fail"],
            f"lead conversion failure: {lead_fail:.0%}")
        add("pressure", st["wins"] >= 3 and bust_win <= t_low["bust_win"],
            f"low bust-win share: {bust_win:.0%}")
        add("pressure", bb_hand > 0 and bust_win < .18, "wins tend to reach cap instead of finishing")

        add("recovery", st["def_opp"] >= 2 and def_fail >= t["def_fail"],
            f"deficit failure rate: {def_fail:.0%}")
        add("recovery", bust_out >= t["bust_out"], f"high bust-out share: {bust_out:.0%}")
        add("recovery", response < -3, f"poor next-hand response after big loss: {response:+.1f} chips")

        add("volatility", big_loss_rate >= t["big_loss"], f"large-loss hand rate: {big_loss_rate:.0%}")
        add("volatility", drawdown >= t["drawdown"], f"avg max drawdown: {drawdown:.0f} chips")
        add("volatility", st["big_loss"] > st["big_win"], "more large losses than large wins")

        add("position", btn_ev < 0, f"negative BTN/SB EV: {btn_ev:+.1f} chips/hand")
        add("position", btn_ev < bb_ev, f"BB outperforms button: BTN {btn_ev:+.1f}, BB {bb_ev:+.1f}")
        add("position", btn_ev - bb_ev < 1.0, "weak button advantage")

        add("pacing", late + 2 < early, f"late-match decay: early {early:+.1f}, late {late:+.1f}")
        add("pacing", st["wins"] >= 3 and bust_win < .20, f"low close-out rate: {bust_win:.0%} bust wins")
        add("pacing", st["lead_opp"] >= 2 and lead_fail > .45, "leads often drift to cap/loss")

        add("adapt", st["after_big_loss"] >= 2 and response < 0,
            f"negative response after big losses: {response:+.1f} chips")
        add("adapt", lead_changes > 2.0, f"frequent lead swings: {lead_changes:.1f}/match")
        add("adapt", st["big_loss"] >= 4 and response < 1, "big losses do not trigger stabilization")

        scores = [len(signals[k]) for k, _ in dims]
        chart_payload.append({
            "id": f"matchStrategyRadar{rank}",
            "label": display_name(m),
            "model": m,
            "color": pb.colors_for([m])[0],
            "scores": scores,
            "signals": {label: signals[k] for k, label in dims},
        })
        leak_dims = sorted(
            [(label, len(signals[k]), signals[k]) for k, label in dims if signals[k]],
            key=lambda x: x[1],
            reverse=True,
        )
        summary = ("No strong aggregate leak signal in the current scoring pass."
                   if not leak_dims else
                   f"Main review area: {leak_dims[0][0]} ({leak_dims[0][1]} signals).")
        cases = []
        seen = set()
        for c in sorted(st["cases"], key=lambda item: 0 if item.get("href") else 1):
            key = (c["dimension"], c["title"], c["opponent"], c["episode"], c.get("hand"))
            if key in seen:
                continue
            seen.add(key)
            cases.append(c)
            if len(cases) == 3:
                break
        while len(cases) < 3:
            label = leak_dims[len(cases) % len(leak_dims)][0] if leak_dims else "Review candidate"
            msg = leak_dims[len(cases) % len(leak_dims)][2][0] if leak_dims else "Audit a large pot in the replay viewer"
            cases.append({"dimension": label, "title": "aggregate signal", "signal": msg,
                          "opponent": "field", "episode": "—", "hand": None, "href": None})
        case_html = ""
        for c in cases[:3]:
            link = (f"""<a class="case-link" href="{html_lib.escape(c['href'], quote=True)}">watch replay</a>"""
                    if c.get("href") else "<div class='case-meta'>no replay file available for this pair</div>")
            hand = f" · hand {c.get('hand')}" if c.get("hand") else ""
            case_html += f"""
        <li>
          <div class="case-title">{html_lib.escape(c['dimension'])}: {html_lib.escape(c['title'])}</div>
          <div class="case-signal">{html_lib.escape(c['signal'])}</div>
          <div class="case-meta">vs {html_lib.escape(str(c['opponent']))} · match {html_lib.escape(str(c['episode']))}{hand}</div>
          {link}
        </li>"""
        cards.append(f"""
    <article class="strategy-card">
      <div class="strategy-head">
        <div><h3>{rank}. {model_cell(m)}</h3><div class="note">{html_lib.escape(summary)}</div></div>
        <div class="strategy-kpi {'pos' if bb_hand >= 0 else 'neg'}">{bb_hand:+.1f}<span>chips/hand</span></div>
      </div>
      <canvas id="matchStrategyRadar{rank}"></canvas>
      <h3>Case studies</h3>
      <ol class="strategy-cases">{case_html}</ol>
    </article>""")

    glossary = ""
    for _, label in dims:
        info = dim_help[label]
        glossary += f"""
      <div class="strategy-gloss">
        <b>{html_lib.escape(label)}</b>
        <span>{html_lib.escape(info['meaning'])}</span>
        <em>Signals: {html_lib.escape(info['signals'])}</em>
      </div>"""
    html = f"""
  <div class="strategy-intro">
    <b>Match strategy radar:</b> each dimension counts triggered diagnostic signals.
    Higher values mean more evidence to review, not better play. Hover a radar
    point to see the signals behind that score.
    <div class="strategy-glossary">{glossary}
    </div>
  </div>
  <div class="strategy-grid">
    {''.join(cards)}
  </div>"""
    js = f"""
const MATCH_STRATEGY = {json.dumps(chart_payload)};
const MATCH_STRATEGY_LABELS = {json.dumps([label for _, label in dims])};
MATCH_STRATEGY.forEach((card) => {{
  const el = document.getElementById(card.id);
  if (!el) return;
  new Chart(el, {{
    type:'radar',
    data:{{ labels:MATCH_STRATEGY_LABELS,
      datasets:[{{ label:card.label, data:card.scores,
        borderColor:card.color, backgroundColor:card.color + '33',
        pointBackgroundColor:card.color }}] }},
    options:{{ scales:{{ r:{{ beginAtZero:true, suggestedMax:4, ticks:{{ stepSize:1 }} }} }},
      plugins:{{ legend:{{ display:false }},
        tooltip:{{ callbacks:{{ afterLabel(ctx) {{
          const dim = ctx.label;
          const sigs = card.signals[dim] || [];
          return sigs.length ? sigs.map(s => '• ' + s) : ['no signal triggered'];
        }} }} }} }} }} }});
}});
"""
    return html, js


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


def render_html(rep: dict, beh: dict) -> str:
    models = rep["models"]; lb = rep["leaderboard"]
    labels = [r["model"] for r in lb]          # slugs — key behavior stats / colors
    disp_labels = [display_name(m) for m in labels]   # official names for chart axis
    winpct = [round(r["win_rate"] * 100, 1) for r in lb]
    wincols = pb.colors_for(labels)
    beh_html = pb.profile_table(beh, labels) + pb.behavior_charts(beh, labels)
    strategy_html, strategy_js = _match_strategy_html(rep)
    replay_btn = ('<a class="replaybtn" href="match_replay.html?v=18&cacheBust=18">'
                  '▶ watch match replays</a>')
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
    head = "".join(f"<th>{display_name(m)}</th>" for m in models)
    grid = ""
    for a in models:
        cells = ""
        for b in models:
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
  </div>
  <h2 class="section">1 · 🏆 Results — who won</h2>
  <h3>Match win rate</h3>
  <canvas id="wr"></canvas>
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
  <h2 class="section">2 · 🔍 Why — what decides win &amp; loss</h2>
  {beh_html}
  <h2 class="section">3 · 🔬 Additional analysis</h2>
  {strategy_html}
  <script>
  new Chart(document.getElementById('wr'), {{
    type:'bar',
    data:{{labels:{json.dumps(disp_labels)},datasets:[{{label:'win %',data:{json.dumps(winpct)},backgroundColor:{json.dumps(wincols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},
      scales:{{y:{{beginAtZero:true,max:100,grid:{{color:'#e7e2d8'}},ticks:{{color:'#1c1c1c'}}}},
               x:{{grid:{{color:'#e7e2d8'}},ticks:{{color:'#1c1c1c'}}}}}}}}
  }});
  {strategy_js}
  </script>
</div></body></html>"""


def main():
    data = _load_all_match_data()
    rep = analyze(data)
    rep["strategy"] = _match_strategy(data, rep["models"])
    beh = pb.behavior(EP_GLOBS, "match_hand", rep["models"])
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
