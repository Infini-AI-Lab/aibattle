"""Shared player-behavior stats for the Hold'em match & table modes.

The aggregate ``match_data.json`` / ``table_data.json`` drop the per-step actions,
so behavioral metrics are computed here from the per-episode ``ep*.json`` logs
(which carry every decision: action, amount, street, pot, to-call, reasoning
tokens). Both modes share the same step schema; only the per-hand key differs
(``match_hand`` vs ``table_hand``), so one accumulator serves both.

Also the single source of truth for the model→colour map, so every chart in
every report paints a given model the same colour.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

# One colour per model, used across ALL charts and replay viewers.
MODEL_COLORS = {
    "deepseek-v4-pro": "#60a5fa",  # blue
    "gpt-oss-120b":    "#f472b6",  # pink
    "kimi-k2p6":       "#4ade80",  # green
    "glm-5p1":         "#fbbf24",  # amber
    "minimax-m2p7":    "#a78bfa",  # purple
}
DEFAULT_COLOR = "#94a3b8"
STREETS = ["preflop", "flop", "turn", "river"]
AGGRO = {"bet", "raise", "all_in"}
VOLUNTARY = {"call", "bet", "raise", "all_in"}


def color_for(model: str) -> str:
    # Coached models (e.g. "deepseek-v4-pro-coached") inherit their base
    # model's colour so the two arenas stay visually consistent; without this
    # every coached model collapses to the same grey DEFAULT_COLOR.
    base = model[:-len("-coached")] if model.endswith("-coached") else model
    return MODEL_COLORS.get(base, DEFAULT_COLOR)


def colors_for(models) -> list:
    return [color_for(m) for m in models]


def _blank() -> dict:
    return {
        "decisions": 0,
        "acts": defaultdict(int),
        "pf_hands": 0, "vpip": 0, "pfr": 0,
        "facing_bet": 0, "fold_facing_bet": 0,
        "bet_raise": 0, "calls": 0, "allin": 0,
        "betsize_ratios": [],
        "by_street": {s: {"aggr": 0, "n": 0} for s in STREETS},
        "tokens": [], "latency": [],
        "hands": 0, "wtsd": 0, "wtsd_won": 0, "won": 0,
    }


def accumulate(stats: dict, ep: dict, hand_key: str) -> None:
    """Fold one episode's steps into ``stats`` (model -> counters)."""
    seat = ep.get("seat_assignment", {})
    hsmap = {h["hand"]: h for h in ep.get("hand_summaries", [])}

    hands = defaultdict(list)
    for s in ep.get("steps", []):
        hands[s["observation"]["public"].get(hand_key)].append(s)

    for hno, steps in hands.items():
        hs = hsmap.get(hno, {})
        reason = hs.get("reason")
        deltas = hs.get("deltas", {})
        pf_actions = defaultdict(set)   # player -> set of preflop actions
        folded, players_in = set(), set()

        for s in steps:
            p = s["player"]
            m = seat.get(p, p)
            st = stats[m]
            pub = s["observation"]["public"]
            act = (s.get("selected_action") or "").lower()
            street = pub.get("street", "preflop")
            to_call = pub.get("to_call") or 0
            pot = pub.get("pot") or 0
            amt = s.get("selected_amount")

            players_in.add(p)
            st["decisions"] += 1
            st["acts"][act] += 1
            if street in st["by_street"]:
                st["by_street"][street]["n"] += 1
                if act in AGGRO:
                    st["by_street"][street]["aggr"] += 1
            if street == "preflop":
                pf_actions[p].add(act)
            if to_call > 0:
                st["facing_bet"] += 1
                if act == "fold":
                    st["fold_facing_bet"] += 1
            if act in AGGRO:
                st["bet_raise"] += 1
            if act == "call":
                st["calls"] += 1
            if act == "all_in":
                st["allin"] += 1
            if act in ("bet", "raise") and amt and pot > 0:
                st["betsize_ratios"].append(amt / pot)
            if act == "fold":
                folded.add(p)

            meta = (s.get("response") or {}).get("metadata", {})
            ct = meta.get("completion_tokens")
            if isinstance(ct, (int, float)):
                st["tokens"].append(ct)
            lat = meta.get("latency_ms")
            if lat:
                st["latency"].append(lat)

        # per-hand, per-player roll-ups
        for p, acts in pf_actions.items():
            st = stats[seat.get(p, p)]
            st["pf_hands"] += 1
            if acts & VOLUNTARY:
                st["vpip"] += 1
            if acts & AGGRO:
                st["pfr"] += 1
        for p in players_in:
            st = stats[seat.get(p, p)]
            st["hands"] += 1
            won = deltas.get(p, 0) > 0
            if won:
                st["won"] += 1
            if reason == "showdown" and p not in folded:
                st["wtsd"] += 1
                if won:
                    st["wtsd_won"] += 1


def _finalize(st: dict) -> dict:
    def rate(a, b):
        return round(a / b, 3) if b else 0.0
    dec = st["decisions"] or 1
    mix = {a: rate(st["acts"].get(a, 0), dec)
           for a in ("fold", "check", "call", "bet", "raise", "all_in")}
    street_aggr = {s: rate(st["by_street"][s]["aggr"], st["by_street"][s]["n"])
                   for s in STREETS}
    return {
        "decisions": st["decisions"], "hands": st["hands"],
        "vpip": rate(st["vpip"], st["pf_hands"]),
        "pfr": rate(st["pfr"], st["pf_hands"]),
        "aggression": rate(st["bet_raise"], st["bet_raise"] + st["calls"]),
        "fold_to_bet": rate(st["fold_facing_bet"], st["facing_bet"]),
        "allin_rate": rate(st["allin"], dec),
        "avg_betsize": round(sum(st["betsize_ratios"]) / len(st["betsize_ratios"]), 2)
                       if st["betsize_ratios"] else 0.0,
        "wtsd": rate(st["wtsd"], st["hands"]),
        "wsd": rate(st["wtsd_won"], st["wtsd"]),          # won-at-showdown
        "hand_win": rate(st["won"], st["hands"]),
        "avg_tokens": round(sum(st["tokens"]) / len(st["tokens"]))
                      if st["tokens"] else 0,
        "avg_latency_s": round(sum(st["latency"]) / len(st["latency"]) / 1000, 1)
                         if st["latency"] else 0.0,
        "action_mix": mix,
        "street_aggr": street_aggr,
    }


def behavior(ep_glob: str, hand_key: str, models: list) -> dict:
    """Return {model: finalized behavior stats} over all ep files in ``ep_glob``."""
    stats = {m: _blank() for m in models}
    for path in sorted(glob.glob(ep_glob)):
        try:
            ep = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        accumulate(stats, ep, hand_key)
    return {m: _finalize(stats[m]) for m in models}


# ---------------------------------------------------------------------------
# Rendering helpers (shared by the match & table reports so the behavioural
# sections look identical and a model keeps its colour everywhere).
# ---------------------------------------------------------------------------

# Action segment colours (used only in the action-mix chart; distinct context
# from the model palette).
ACTION_COLORS = {
    "fold": "#fb7185", "check": "#64748b", "call": "#38bdf8",
    "bet": "#fbbf24", "raise": "#f97316", "all_in": "#a855f7",
}
_AXC = "{grid:{color:'#20242e'},ticks:{color:'#9aa3b5'}}"


def _swatch(model: str) -> str:
    return (f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{color_for(model)};margin-right:7px;vertical-align:middle'></span>")


def profile_table(beh: dict, models: list) -> str:
    """Behaviour leaderboard rows (caller passes models in ranked order)."""
    rows = ""
    for i, m in enumerate(models, 1):
        s = beh[m]
        rows += (
            f"<tr><td>{i}</td><td class='model'>{_swatch(m)}{m}</td>"
            f"<td>{s['vpip']*100:.0f}%</td><td>{s['pfr']*100:.0f}%</td>"
            f"<td>{s['aggression']*100:.0f}%</td><td>{s['fold_to_bet']*100:.0f}%</td>"
            f"<td>{s['avg_betsize']:.2f}x</td><td>{s['allin_rate']*100:.0f}%</td>"
            f"<td>{s['wtsd']*100:.0f}%</td><td>{s['wsd']*100:.0f}%</td>"
            f"<td>{s['hand_win']*100:.0f}%</td><td>{s['avg_tokens']:,}</td></tr>")
    return (
        "<h2>🎭 Player behaviour profiles</h2>\n"
        "<table>\n"
        "  <tr><th>#</th><th class='model'>model</th><th>VPIP</th><th>PFR</th>"
        "<th>aggr</th><th>fold→bet</th><th>bet size</th><th>all-in%</th>"
        "<th>WTSD</th><th>W@SD</th><th>hand-win</th><th>tokens/dec</th></tr>\n"
        f"  {rows}\n</table>\n"
        "<div class=\"note\">VPIP = voluntarily entered the pot preflop (looseness). "
        "PFR = preflop raise. aggr = bet+raise share of bet/raise/call. "
        "fold→bet = folds when facing a bet. bet size = avg bet/raise as a multiple of the pot. "
        "WTSD = went to showdown; W@SD = won at showdown. "
        "hand-win = share of hands that netted chips. tokens/dec = avg reasoning tokens per decision.</div>\n")


def behavior_charts(beh: dict, models: list) -> str:
    """Canvases + Chart.js init for the 4 shared behaviour charts. Returns a
    self-contained HTML block (model colours throughout)."""
    cols = [color_for(m) for m in models]
    short = [m.split("-")[0] for m in models]
    # style scatter: one single-point dataset per model so the legend is colour-keyed
    scatter_ds = [
        {"label": m, "data": [{"x": round(beh[m]["vpip"] * 100, 1),
                               "y": round(beh[m]["aggression"] * 100, 1)}],
         "backgroundColor": color_for(m), "pointRadius": 9, "pointHoverRadius": 11}
        for m in models]
    mix_ds = [
        {"label": a, "data": [round(beh[m]["action_mix"].get(a, 0) * 100, 1) for m in models],
         "backgroundColor": ACTION_COLORS[a]}
        for a in ("fold", "check", "call", "bet", "raise", "all_in")]
    street_ds = [
        {"label": m, "data": [round(beh[m]["street_aggr"][s] * 100, 1) for s in STREETS],
         "borderColor": color_for(m), "backgroundColor": color_for(m),
         "tension": 0.3, "fill": False}
        for m in models]
    tokens = [beh[m]["avg_tokens"] for m in models]

    j = json.dumps
    return f"""
  <h2>🎭 Style map <span class="note">(VPIP vs aggression — top-right = loose &amp; aggressive)</span></h2>
  <div class="grid2">
    <div><canvas id="behStyle"></canvas></div>
    <div><h2 style="margin-top:0">🧮 Thinking effort <span class="note">(avg reasoning tokens / decision)</span></h2><canvas id="behTokens"></canvas></div>
  </div>
  <div class="grid2">
    <div><h2>🃏 Action mix <span class="note">(% of all decisions)</span></h2><canvas id="behMix"></canvas></div>
    <div><h2>📈 Aggression by street</h2><canvas id="behStreet"></canvas></div>
  </div>
  <script>
  {{
  const AXC={_AXC};
  new Chart(document.getElementById('behStyle'),{{type:'scatter',
    data:{{datasets:{j(scatter_ds)}}},
    options:{{plugins:{{legend:{{position:'right',labels:{{color:'#cdd6f4'}}}}}},
      scales:{{x:{{title:{{display:true,text:'VPIP %',color:'#9aa3b5'}},min:0,max:100,...AXC}},
               y:{{title:{{display:true,text:'aggression %',color:'#9aa3b5'}},min:0,max:100,...AXC}}}}}}}});
  new Chart(document.getElementById('behTokens'),{{type:'bar',
    data:{{labels:{j(short)},datasets:[{{label:'tokens/decision',data:{j(tokens)},backgroundColor:{j(cols)}}}]}},
    options:{{plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true,...AXC}},x:AXC}}}}}});
  new Chart(document.getElementById('behMix'),{{type:'bar',
    data:{{labels:{j(short)},datasets:{j(mix_ds)}}},
    options:{{plugins:{{legend:{{labels:{{color:'#cdd6f4'}}}}}},
      scales:{{x:{{stacked:true,...AXC}},y:{{stacked:true,max:100,...AXC}}}}}}}});
  new Chart(document.getElementById('behStreet'),{{type:'line',
    data:{{labels:{j(STREETS)},datasets:{j(street_ds)}}},
    options:{{plugins:{{legend:{{labels:{{color:'#cdd6f4'}}}}}},
      scales:{{y:{{beginAtZero:true,max:100,title:{{display:true,text:'aggression %',color:'#9aa3b5'}},...AXC}},x:AXC}}}}}});
  }}
  </script>
"""


if __name__ == "__main__":  # quick self-test
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "match"
    models = ["deepseek-v4-pro", "gpt-oss-120b", "kimi-k2p6", "glm-5p1", "minimax-m2p7"]
    if mode == "match":
        b = behavior("runs/match_tournament/*__vs__*/ep*.json", "match_hand", models)
    else:
        b = behavior("runs/table_tournament/table/ep*.json", "table_hand", models)
    print(f"=== {mode} behavior ===")
    hdr = f"{'model':<17}{'VPIP':>6}{'PFR':>6}{'aggr':>6}{'f2b':>6}{'bet×':>6}{'AI%':>6}{'WTSD':>6}{'W@SD':>6}{'tok':>7}"
    print(hdr)
    for m in models:
        s = b[m]
        print(f"{m:<17}{s['vpip']*100:>5.0f}%{s['pfr']*100:>5.0f}%{s['aggression']*100:>5.0f}%"
              f"{s['fold_to_bet']*100:>5.0f}%{s['avg_betsize']:>6}{s['allin_rate']*100:>5.0f}%"
              f"{s['wtsd']*100:>5.0f}%{s['wsd']*100:>5.0f}%{s['avg_tokens']:>7}")
