"""Render a dedicated GPT-vs-Claude analysis mini-site from the coached
head-to-head run (runs/gpt_claude_coached_tournament).

That run pits two GPT models (gpt-5.5, gpt-5.4) against two Claude models
(claude-opus-4.8, claude-sonnet-4.6) across four games, all with one-line
coaching in the prompt. This script reuses the existing per-game analyzers and
adds a GPT-family-vs-Claude-family framing on top:

  reports/gpt_vs_claude/
    index.html                 family head-to-head + cross-game leaderboard + cards
    connect4_report.html       (board analyzer)
    gomoku_report.html         (board analyzer)
    holdem_1hand_report.html   (1-hand poker analyzer)
    holdem_match_report.html   (match results, custom render)

The per-game data files (<label>/<label>_data.json) carry the same step schema
the base analyzers consume; only light reshaping is needed (models is a list of
dicts here, and episodes live under `pairs`). Run from the repo root:

    python scripts/analyze_gpt_claude.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_board_tournament as bt          # noqa: E402
import analyze_tournament as ht                # noqa: E402  (Hold'em 1-hand)
import analyze_match_tournament as mt          # noqa: E402

RUN = "runs/gpt_claude_coached_tournament"
OUT = "reports/gpt_vs_claude"

# Family membership keyed by name prefix — the whole point of this mini-site.
GPT = ["gpt-5.5", "gpt-5.4"]
CLAUDE = ["claude-opus-4.8", "claude-sonnet-4.6"]


def family(name: str) -> str:
    return "GPT" if name.startswith("gpt") else "Claude"


def names(data: dict) -> list:
    return [m["name"] for m in data["models"]]


def load(label: str) -> dict:
    return json.load(open(os.path.join(RUN, label, f"{label}_data.json")))


NAV_HEAD = bt.NAV_HEAD


# ---------------------------------------------------------------------------
# Cross-family head-to-head (the headline).
# ---------------------------------------------------------------------------
def family_h2h(games: dict) -> dict:
    """Aggregate GPT-vs-Claude records over only the inter-family pairings.

    Intra-family games (gpt vs gpt, claude vs claude) are excluded — they say
    nothing about which family is stronger. Each episode contributes one
    win/loss/draw from GPT's perspective (by winner_name), and for the two
    poker games we also tally GPT's net chips so margin, not just frequency,
    is visible.
    """
    per_game = {}
    tot = {"w": 0, "l": 0, "d": 0, "chips": 0.0, "has_chips": False}
    for label, data in games.items():
        w = l = d = 0
        chips = 0.0
        is_poker = label.startswith("holdem")
        for pair in data["pairs"]:
            if family(pair["a"]) == family(pair["b"]):
                continue
            for e in pair["episodes"]:
                seat = e["seat_assignment"]
                gpt_seat = next(s for s, nm in seat.items() if family(nm) == "GPT")
                chips += e["returns"][gpt_seat]
                wn = e.get("winner_name")
                if wn is None:
                    d += 1
                elif family(wn) == "GPT":
                    w += 1
                else:
                    l += 1
        per_game[label] = {"w": w, "l": l, "d": d, "chips": round(chips, 1),
                           "poker": is_poker}
        tot["w"] += w; tot["l"] += l; tot["d"] += d
        if is_poker:
            tot["chips"] += chips; tot["has_chips"] = True
    tot["chips"] = round(tot["chips"], 1)
    return {"per_game": per_game, "total": tot}


# ---------------------------------------------------------------------------
# Per-model invalid-move rate, pooled across all four games (the Sonnet story).
# ---------------------------------------------------------------------------
def invalid_rates(games: dict, models: list) -> dict:
    moves = {m: 0 for m in models}
    inv = {m: 0 for m in models}
    for data in games.values():
        for pair in data["pairs"]:
            for e in pair["episodes"]:
                ic = e.get("invalid_count", {})
                for seat, nm in e["seat_assignment"].items():
                    inv[nm] += ic.get(seat, 0)
                for s in e["steps"]:
                    moves[s["agent_name"]] += 1
    return {m: {"invalid": inv[m], "moves": moves[m],
                "rate": round(inv[m] / max(moves[m], 1), 4)} for m in models}


# ---------------------------------------------------------------------------
# Hold'em Match page — custom render (the base match render needs per-episode
# behavior files we don't reproduce here; results + H2H tell the story).
# ---------------------------------------------------------------------------
def render_match(rep: dict, title_meta: str) -> str:
    models = rep["models"]
    lb = rep["leaderboard"]
    payload = json.dumps(rep)
    rows = ""
    for i, r in enumerate(lb, 1):
        fam = family(r["model"])
        rows += f"""<tr>
          <td>{i}</td>
          <td class='model'><span class='fam {fam.lower()}'>{fam}</span> {r['model']}</td>
          <td>{r['win_rate']*100:.0f}%</td><td>{r['wins']}</td><td>{r['draws']}</td>
          <td>{r['busted_out_rate']*100:.0f}%</td>
          <td>{r['avg_hands_per_match']:.1f}</td>
          <td>{r['avg_win_margin']:.0f}</td>
        </tr>"""

    hh = "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in models) + "</tr>"
    for a in models:
        hh += f"<tr><th class='model'>{a}</th>"
        for b in models:
            if a == b:
                hh += "<td class='diag'>—</td>"
            else:
                w = rep["h2h_wins"][a][b]
                pl = rep["h2h_played"][a][b]
                lo = rep["h2h_wins"][b][a]
                cls = "pos" if w > lo else ("neg" if lo > w else "")
                hh += f"<td class='{cls}'>{w}-{lo}<div class='small'>/{pl}</div></td>"
        hh += "</tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GPT vs Claude — 🃏 Hold'em Match</title>
{bt._favicon("🃏")}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:28px 28px 80px; }}
  h1 {{ font-size:25px; }} h2 {{ font-size:18px; margin-top:38px; border-bottom:1px solid #2a2f3a; padding-bottom:6px; }}
  h3 {{ font-size:14px; color:#9aa3b5; }}
  .sub {{ color:#8b93a7; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ padding:6px 8px; text-align:center; border-bottom:1px solid #20242e; }}
  th {{ color:#9aa3b5; }} td.model,th.model {{ text-align:left; font-weight:600; color:#cdd6f4; }}
  .pos {{ color:#4ade80; }} .neg {{ color:#f87171; }} .diag {{ color:#3a3f4b; }}
  .small {{ font-size:10px; color:#8b93a7; }}
  .note {{ color:#8b93a7; font-size:12px; margin:6px 0; }}
  canvas {{ max-height:300px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:10px; }}
  .fam {{ font-size:10px; padding:1px 6px; border-radius:999px; font-weight:600; }}
  .fam.gpt {{ background:#10322a; color:#4ade80; }}
  .fam.claude {{ background:#2a2138; color:#c4b5fd; }}
  .replaybtn {{ display:inline-block; margin-top:12px; background:#1b2030; color:#a5b4fc;
    border:1px solid #2a2f3a; border-radius:8px; padding:8px 14px; font-size:13px; text-decoration:none; }}
  .replaybtn:hover {{ border-color:#60a5fa; color:#fff; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🎲 GPT vs Claude — 🃏 Hold'em Match</h1>
  <div class="sub">{title_meta}</div>
  <a class="replaybtn" href="match_replay.html">▶ Watch match replays</a>

  <h2>Leaderboard — match win rate</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>match win%</th><th>wins</th>
        <th>draws</th><th>busted-out%</th><th>avg hands/match</th><th>avg win margin</th></tr>
    {rows}
  </table>
  <div class="note">A match is a heads-up sit-and-go (up to {rep['max_hands']} hands, stacks
    carried); the match winner is whoever holds more chips at the end. Win margin is the average
    final chip gap in matches won. Busted-out% = share of matches lost by going broke.</div>

  <div class="grid2">
    <div><h3>Match win rate</h3><canvas id="wr"></canvas></div>
    <div><h3>Avg hands per match</h3><canvas id="hp"></canvas></div>
  </div>

  <h2>Head-to-head (row match-wins vs column / matches played)</h2>
  <table class='h2h'>{hh}</table>

<script>
const R = {payload};
const lb = R.leaderboard, M = lb.map(r=>r.model);
Chart.defaults.color='#9aa3b5'; Chart.defaults.borderColor='#232838';
const col = m => m.startsWith('gpt') ? '#4ade80' : '#a78bfa';
new Chart(document.getElementById('wr'), {{ type:'bar',
  data:{{ labels:M, datasets:[{{label:'win %', data:lb.map(r=>r.win_rate*100),
    backgroundColor:M.map(col)}}]}},
  options:{{ indexAxis:'y', scales:{{x:{{min:0,max:100}}}}, plugins:{{legend:{{display:false}}}} }} }});
new Chart(document.getElementById('hp'), {{ type:'bar',
  data:{{ labels:M, datasets:[{{label:'hands', data:lb.map(r=>r.avg_hands_per_match),
    backgroundColor:M.map(col)}}]}},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}} }} }});
</script>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Index page.
# ---------------------------------------------------------------------------
GAME_META = {
    "connect4":     {"title": "🔴 Connect Four", "badges": ["Perfect info", "2P", "50 eps/pair"]},
    "gomoku":       {"title": "⚫ Gomoku-Lite", "badges": ["Perfect info", "2P", "50 eps/pair"]},
    "holdem_1hand": {"title": "🃏 Hold'em 1-Hand", "badges": ["Imperfect info", "Heads-up", "100 hands/pair"]},
    "holdem_match": {"title": "🃏 Hold'em Match", "badges": ["Imperfect info", "Heads-up", "20 matches/pair"]},
}
HREF = {g: f"{g}_report.html" for g in GAME_META}


def render_index(fh: dict, rates: dict, cards: list, arena_rows: list,
                 num_models: int) -> str:
    tot = fh["total"]
    n_inter = tot["w"] + tot["l"] + tot["d"]
    gpt_pct = tot["w"] / max(n_inter, 1) * 100
    cla_pct = tot["l"] / max(n_inter, 1) * 100
    lead = "GPT" if tot["w"] > tot["l"] else ("Claude" if tot["l"] > tot["w"] else "Even")

    # per-game family rows
    pg_rows = ""
    for g, meta in GAME_META.items():
        r = fh["per_game"][g]
        n = r["w"] + r["l"] + r["d"]
        wp = r["w"] / max(n, 1) * 100
        chips = (f"<span class='small'>GPT {r['chips']:+.0f} chips</span>"
                 if r["poker"] else "")
        bar = (f"<span class='hbar'><span class='hg' style='width:{wp:.1f}%'></span>"
               f"<span class='hc' style='width:{100-wp - (r['d']/max(n,1)*100):.1f}%'></span></span>")
        pg_rows += (f"<tr><td class='model'>{meta['title']}</td>"
                    f"<td>{r['w']}–{r['l']}–{r['d']}</td>"
                    f"<td>{wp:.0f}%</td><td class='barcell'>{bar}</td>"
                    f"<td class='best'>{chips}</td></tr>")

    # cross-game leaderboard rows
    lb_rows = ""
    for i, r in enumerate(arena_rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
        fam = family(r["model"])
        iv = rates[r["model"]]
        lb_rows += (
            f"<tr><td class='rk'>{medal}</td>"
            f"<td class='model'><span class='fam {fam.lower()}'>{fam}</span> {r['model']}</td>"
            f"<td class='scorecell'><span class='bar' style='width:{r['score']}%'></span>"
            f"<span class='sval'>{r['score']:.0f}</span></td>"
            f"<td class='cov'>{r['games']}/4</td>"
            f"<td>{iv['rate']*100:.2f}%</td>"
            f"<td class='best'>{r['best']}</td></tr>")

    card_html = "".join(cards)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GPT vs Claude — AI Battle Arena</title>
{bt._favicon("🥊")}
{NAV_HEAD}
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:36px 28px 70px; }}
  h1 {{ font-size:27px; margin-bottom:6px; }} .sub {{ color:#8b93a7; margin-bottom:26px; }}
  h2 {{ font-size:19px; color:#cdd6f4; margin:0; }}
  .note {{ color:#8b93a7; font-size:12px; margin:6px 0 14px; }}
  section {{ margin-top:30px; border:1px solid #232838; border-radius:16px;
    padding:22px; background:linear-gradient(180deg,#141823,#10131b); }}

  /* family scoreboard */
  .vs {{ display:flex; align-items:stretch; gap:0; margin:18px 0 6px; border-radius:14px; overflow:hidden;
    border:1px solid #232838; }}
  .vs .side {{ flex:1; padding:18px 22px; }}
  .vs .gptside {{ background:linear-gradient(180deg,#0f2a22,#0c1f1a); }}
  .vs .claside {{ background:linear-gradient(180deg,#241c33,#160f22); text-align:right; }}
  .vs .fname {{ font-size:13px; letter-spacing:.05em; text-transform:uppercase; color:#9aa3b5; }}
  .vs .fwins {{ font-size:40px; font-weight:800; line-height:1; margin-top:4px; }}
  .vs .gptside .fwins {{ color:#4ade80; }} .vs .claside .fwins {{ color:#c4b5fd; }}
  .vs .fpct {{ font-size:12px; color:#8b93a7; margin-top:4px; }}
  .vs .mid {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding:0 16px; background:#0d1017; color:#8b93a7; font-size:12px; }}
  .vs .mid b {{ font-size:18px; color:#e6e6e6; }}

  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ padding:7px 9px; text-align:center; border-bottom:1px solid #20242f; }}
  th {{ color:#9aa3b5; font-weight:600; }}
  td.model,th.model {{ text-align:left; font-weight:600; color:#cdd6f4; }}
  td.rk,th.rk {{ width:34px; color:#8b93a7; }} td.cov {{ color:#8b93a7; }} td.best {{ color:#aab2c5; text-align:left; }}
  .barcell {{ width:160px; }}
  .hbar {{ display:inline-flex; width:150px; height:14px; border-radius:4px; overflow:hidden; background:#3a2330; }}
  .hbar .hg {{ background:#4ade80; }} .hbar .hc {{ background:#a78bfa; }}
  .scorecell {{ position:relative; min-width:120px; }}
  .scorecell .bar {{ position:absolute; left:0; top:50%; transform:translateY(-50%);
    height:18px; border-radius:4px; background:linear-gradient(90deg,#3b82f6,#a78bfa); opacity:.5; }}
  .scorecell .sval {{ position:relative; font-weight:600; color:#f3f4f6; }}
  .fam {{ font-size:10px; padding:1px 6px; border-radius:999px; font-weight:600; }}
  .fam.gpt {{ background:#10322a; color:#4ade80; }}
  .fam.claude {{ background:#2a2138; color:#c4b5fd; }}

  .cards {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:6px; }}
  .card {{ display:block; text-decoration:none; color:inherit; background:#171a23;
    border:1px solid #232838; border-radius:14px; padding:18px; transition:.15s; }}
  .card:hover {{ border-color:#60a5fa; transform:translateY(-2px); }}
  .ctitle {{ font-size:18px; font-weight:700; color:#cdd6f4; }}
  .badges {{ margin:8px 0; display:flex; gap:6px; flex-wrap:wrap; }}
  .badge {{ font-size:10px; color:#aab2c5; background:#1c2130; border:1px solid #2a3142;
    padding:2px 7px; border-radius:999px; }}
  .champ {{ font-size:13px; color:#e6e6e6; }} .metric {{ color:#a78bfa; font-weight:600; }}
  .cgo {{ margin-top:12px; font-size:13px; color:#60a5fa; }}
  @media (max-width:640px) {{ .cards {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">
  <h1>🥊 GPT vs Claude — Coached Head-to-Head</h1>
  <div class="sub">Two GPT models ({", ".join(GPT)}) vs two Claude models ({", ".join(CLAUDE)})
    across four games, every move prompted with one-line coaching · reasoning effort medium.</div>

  <section>
    <div class="arena-head"><h2>🏆 Family scoreboard</h2></div>
    <div class="note">Aggregated over the inter-family games only (GPT-vs-Claude pairings;
      intra-family games excluded). Each episode is one win/loss/draw by winner.</div>
    <div class="vs">
      <div class="side gptside">
        <div class="fname">GPT family</div>
        <div class="fwins">{tot['w']}</div>
        <div class="fpct">{gpt_pct:.0f}% of decided · {tot['chips']:+.0f} net chips in poker</div>
      </div>
      <div class="mid"><div>wins</div><b>vs</b><div>{tot['d']} draws</div></div>
      <div class="side claside">
        <div class="fname">Claude family</div>
        <div class="fwins">{tot['l']}</div>
        <div class="fpct">{cla_pct:.0f}% of decided</div>
      </div>
    </div>
    <div class="note" style="text-align:center">{lead} leads across {n_inter} inter-family games.</div>

    <table style="margin-top:14px">
      <tr><th class='model'>game</th><th>GPT W–L–D</th><th>GPT win%</th>
          <th class='barcell'>GPT ◄ ► Claude</th><th class='best'>poker margin</th></tr>
      {pg_rows}
    </table>
  </section>

  <section>
    <div class="arena-head"><h2>🏅 Cross-game leaderboard</h2></div>
    <div class="note">Arena Score = mean within-game finishing position (best 100, worst 0)
      across all four games. Invalid% is illegal-move rate pooled over every decision.</div>
    <table>
      <tr><th class='rk'>#</th><th class='model'>model</th><th>Arena Score</th>
          <th>coverage</th><th>invalid%</th><th class='best'>best game</th></tr>
      {lb_rows}
    </table>
  </section>

  <section>
    <div class="arena-head"><h2>🎮 Per-game analysis</h2></div>
    <div class="note">Full per-game breakdowns — tactical accuracy for the board games,
      poker behavior and results for Hold'em.</div>
    <div class="cards">{card_html}</div>
  </section>
</div></body></html>"""


# ---------------------------------------------------------------------------
def write(path: str, html: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


NAV_JS = r"""// Navbar for the GPT-vs-Claude mini-site. Links resolve within
// reports/gpt_vs_claude/; a back-link returns to the base arena.
(function () {
  var ACTIVE = {
    "index.html": "index.html",
    "connect4_report.html": "connect4_report.html",
    "connect4_replay.html": "connect4_report.html",
    "gomoku_report.html": "gomoku_report.html",
    "gomoku_replay.html": "gomoku_report.html",
    "holdem_1hand_report.html": "holdem_1hand_report.html",
    "holdem_replay.html": "holdem_1hand_report.html",
    "holdem_match_report.html": "holdem_match_report.html",
    "match_replay.html": "holdem_match_report.html"
  };
  var file = location.pathname.split("/").pop() || "index.html";
  var active = ACTIVE[file] || "";
  function a(href, label, cls) {
    var on = href === active ? " active" : "";
    return '<a class="' + cls + on + '" href="' + href + '">' + label + "</a>";
  }
  var html =
    '<a class="brand" href="index.html">🥊 GPT vs Claude</a>' +
    a("index.html", "Overview", "nav") +
    a("connect4_report.html", "🔴 Connect Four", "nav") +
    a("gomoku_report.html", "⚫ Gomoku", "nav") +
    "<span class=\"navclust\">🃏 Hold'em</span>" +
    a("holdem_1hand_report.html", "1-Hand", "nav") +
    a("holdem_match_report.html", "Match", "nav") +
    '<a class="navgrp" href="../index.html">← Base Arena</a>';
  function mount() {
    var nav = document.querySelector("nav.navbar");
    if (!nav) {
      nav = document.createElement("nav");
      nav.className = "navbar";
      document.body.insertBefore(nav, document.body.firstChild);
    }
    nav.innerHTML = html;
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else { mount(); }
})();
"""


# Replay viewers are shared client-side apps; we copy each base viewer in and
# only repoint its data BASE at this run's prebuilt replays (under the reports/
# `runs` symlink, one level up from this subdir). The replay JSON is produced by
# scripts/build_gpt_claude_replays.py.
VIEWERS = {
    # viewer file          (old BASE,                                  replay label)
    "connect4_replay.html": ("runs/board_tournament/replays/connect4/", "connect4"),
    "gomoku_replay.html":   ("runs/board_tournament/replays/gomoku/",   "gomoku"),
    "holdem_replay.html":   ("runs/tournament/replays/holdem/",         "holdem_1hand"),
    "match_replay.html":    ("runs/match_tournament/replays/match/",    "holdem_match"),
}


def copy_viewers():
    for fname, (old_base, label) in VIEWERS.items():
        src = os.path.join("reports", fname)
        if not os.path.exists(src):
            print(f"  WARN: missing viewer {src}; skipping")
            continue
        html = open(src, encoding="utf-8").read()
        new_base = f"../runs/{os.path.basename(RUN)}/replays/{label}/"
        needle = f'const BASE = "{old_base}";'
        repl = f'const BASE = "{new_base}";'
        if needle not in html:
            print(f"  WARN: BASE not found in {fname}; viewer left unpatched")
        html = html.replace(needle, repl)
        write(os.path.join(OUT, fname), html)


def main():
    os.makedirs(OUT, exist_ok=True)
    # Shared nav assets (css from base, custom js for this subdir).
    shutil.copy("reports/nav.css", os.path.join(OUT, "nav.css"))
    write(os.path.join(OUT, "nav.js"), NAV_JS)
    copy_viewers()

    games = {g: load(g) for g in GAME_META}
    model_names = names(games["connect4"])

    cards = []
    arena_entries = []  # for cross-game leaderboard (bt._arena_scores)

    # --- board games: reuse the board analyzer + renderer ---
    # Leave bt._VARIANT empty so render_game emits the "Watch replays" button
    # (-> {game}_replay.html); we ship those viewers into this subdir below.
    bt._VARIANT = ""
    for g in ("connect4", "gomoku"):
        data = games[g]
        reshaped = {"models": model_names, "games": data["pairs"]}
        rep = bt.analyze_game(g, reshaped)
        write(os.path.join(OUT, HREF[g]), bt.render_game(g, rep))
        ordered = sorted(rep["models"],
                         key=lambda m: bt._elo_key(rep["elo"], m), reverse=True)
        champ = ordered[0]
        arena_entries.append({"title": GAME_META[g]["title"], "ranking": ordered})
        cards.append(_card(g, f"🏆 {champ} "
                           f"<span class='metric'>Elo {bt._elo_txt(rep['elo'][champ])}</span>"))

    # --- Hold'em 1-hand: reuse the poker analyzer + renderer ---
    # Empty _VARIANT keeps render_html's "Watch hand replays" button
    # (-> holdem_replay.html), shipped into this subdir below.
    ht._VARIANT = ""
    d1 = games["holdem_1hand"]
    reshaped = {"models": model_names, "games": d1["pairs"],
                "hands": d1["episodes_per_pair"]}
    hrep = ht.analyze(reshaped)
    write(os.path.join(OUT, HREF["holdem_1hand"]), ht.render_html(hrep))
    pm = hrep["per_model"]
    ordered = sorted(hrep["models"], key=lambda m: pm[m]["bb_per_100"], reverse=True)
    champ = ordered[0]
    arena_entries.append({"title": GAME_META["holdem_1hand"]["title"], "ranking": ordered})
    cards.append(_card("holdem_1hand", f"🏆 {champ} "
                       f"<span class='metric'>{pm[champ]['bb_per_100']:+.1f} bb/100</span>"))

    # --- Hold'em match: reuse analyze, custom render ---
    dm = games["holdem_match"]
    max_hands = dm["pairs"][0]["episodes"][0].get("max_hands")
    reshaped = {"models": model_names, "pairs": dm["pairs"],
                "max_hands": max_hands, "episodes_per_pair": dm["episodes_per_pair"]}
    mrep = mt.analyze(reshaped)
    meta = (f"Heads-up · {mrep['episodes_per_pair']} matches/pair · up to "
            f"{mrep['max_hands']} hands/match · stacks carried, match-level winner")
    write(os.path.join(OUT, HREF["holdem_match"]), render_match(mrep, meta))
    lb = mrep["leaderboard"]
    arena_entries.append({"title": GAME_META["holdem_match"]["title"],
                          "ranking": [r["model"] for r in lb]})
    champ = lb[0]
    cards.append(_card("holdem_match", f"🏆 {champ['model']} "
                       f"<span class='metric'>{champ['win_rate']*100:.0f}% match wins</span>"))

    # --- headline aggregates + index ---
    fh = family_h2h(games)
    rates = invalid_rates(games, model_names)
    arena_rows = bt._arena_scores(arena_entries)
    write(os.path.join(OUT, "index.html"),
          render_index(fh, rates, cards, arena_rows, len(model_names)))

    print(f"Wrote GPT-vs-Claude mini-site to {OUT}/")
    t = fh["total"]
    print(f"  Family H2H (inter-family): GPT {t['w']} – {t['l']} Claude  "
          f"({t['d']} draws, GPT poker chips {t['chips']:+.0f})")
    for r in arena_rows:
        print(f"  {r['model']:<20} arena={r['score']:>5}  invalid={rates[r['model']]['rate']*100:.2f}%")


def _card(game: str, champ_line: str) -> str:
    meta = GAME_META[game]
    badges = "".join(f"<span class='badge'>{b}</span>" for b in meta["badges"])
    return (f"<a class='card' href='{HREF[game]}'>"
            f"<div class='ctitle'>{meta['title']}</div>"
            f"<div class='badges'>{badges}</div>"
            f"<div class='champ'>{champ_line}</div>"
            f"<div class='cgo'>View analysis →</div></a>")


if __name__ == "__main__":
    main()
