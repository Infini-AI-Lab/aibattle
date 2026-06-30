"""Analyze the new-games tournament (independent_blackjack, leduc_poker,
repeated_colonel_blotto, othello_lite_6x6) and emit one interactive HTML report
PER GAME — the new-games counterpart of analyze_board_tournament.py.

Two report shapes, picked by game structure:

  * versus  (leduc / blotto / othello, round-robin seat-swapped): per-model
    results (win/draw/loss, net/game, invalid%, plies, latency), a head-to-head
    win matrix, and Bradley-Terry / Elo ratings fit from all head-to-head
    results. Mirrors the board-game report minus the board-specific tactical
    signals.

  * dealer  (independent_blackjack, model vs the built-in dealer): per-model
    profit, mean/hand, win/push/loss split, plus blackjack-specific play signals
    (bust rate, double rate, natural rate) read from the per-hand episode files.

Reads each game's run folder under runs/new_games_experiment/ (the 6-model rerun;
see GAMES[*]["dir"]); writes <name>_report.html
per game to reports/, plus a
small new_games_index.json that analyze_board_tournament.render_index() reads to
add the four games to the landing page and the cross-game Arena Score.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

from elo_util import bootstrap_elo, wld_from_records, gross_from_records
from model_names import display_name, model_cell
from report_tokens import tokens_from_episodes, token_cost_cells, TOKEN_HEADERS
from report_theme import BASE_CSS, CHART_SETUP
from report_legends import legend as _legend

REPORT_DIR = os.environ.get("AIBATTLE_REPORT_DIR", "reports")

# Per-game presentation + taxonomy. "dir" is the per-game run folder under runs/
# (coached is canonical, one flat folder per game). "kind" selects the report
# shape; "group" (perfect/imperfect) and "badges" drive the landing-page card,
# matching the vocabulary in analyze_board_tournament.GAME_TAXONOMY.
GAMES = {
    "independent_blackjack": {
        "dir": "new_games_experiment/independent_blackjack", "kind": "dealer", "title": "🂡 Blackjack", "emoji": "🂡",
        "href": "blackjack_report.html", "area": "blackjack", "replay": "blackjack_replay.html", "replay_verb": "hand", "group": "imperfect",
        "badges": ["Imperfect info", "vs Dealer", "Stochastic"],
        "blurb": "Model vs the built-in dealer · hit/stand/double · scored by chip profit",
    },
    "leduc_poker": {
        "dir": "new_games_experiment/leduc_poker", "kind": "versus", "title": "🎴 Leduc Holdem", "emoji": "🎴",
        "href": "leduc_report.html", "area": "leduc", "replay": "leduc_replay.html", "replay_verb": "hand", "group": "imperfect",
        "badges": ["Imperfect info", "Heads-up", "Stochastic"],
        "blurb": "Imperfect-information poker · 6-card deck · round-robin, seat-swapped",
        # Poker — score is chips, so rate by a chip-weighted Elo and rank by it
        # (like Hold'em 1-Hand), not by win counts.
        "elo_basis": "chips",
    },
    "repeated_colonel_blotto": {
        "dir": "new_games_experiment/repeated_colonel_blotto", "kind": "versus", "title": "⚔️ Colonel Blotto", "emoji": "⚔️",
        "href": "blotto_report.html", "area": "blotto", "replay": "blotto_replay.html", "replay_verb": "game", "group": "imperfect",
        "badges": ["Imperfect info", "Heads-up", "Simultaneous"],
        "blurb": "Simultaneous resource allocation · repeated rounds · round-robin, seat-swapped",
    },
    "othello_lite_6x6": {
        "dir": "new_games_experiment/othello_lite_6x6", "kind": "versus", "title": "⚫ Othello 6×6", "emoji": "⚫",
        "href": "othello_report.html", "area": "othello", "replay": "othello_replay.html", "replay_verb": "game", "group": "perfect",
        "badges": ["Perfect info", "2P", "Deterministic"],
        "blurb": "Perfect-information board game · 6×6 board · round-robin, seat-swapped",
    },
}

# Short intro per game (shown as a callout under the header, like the Kuhn page).
INTRO = {
    "independent_blackjack": (
        '<input type="checkbox" class="rules-toggle" id="rules-toggle" hidden>'
        '<label class="rules-summary" for="rules-toggle">Setup &amp; rules<span class="rules-hint"> · expand</span></label>'
        '<div class="rules">'
        "<h3>Setup — Blackjack</h3>"
        "Standard "
        '<a href="https://en.wikipedia.org/wiki/Blackjack" target="_blank" rel="noopener">blackjack</a>'
        " against a fixed house dealer (full rules on Wikipedia); this is a "
        "solitaire-style benchmark, <b>not</b> model-vs-model:"
        "<ul>"
        "<li><b>Model vs the built-in dealer</b>, which plays the fixed house line "
        "(hits until 17, then stands). Player actions are <b>hit / stand / double</b> — "
        "no split or insurance.</li>"
        "<li><b>500 hands per model</b>, each from a fresh shuffle. Deals are "
        "<b>independent</b> — not shared across models — so the luck of the cards "
        "differs from model to model.</li>"
        "<li><b>Flat 1-unit bet</b> per hand (2 units on a double); a natural pays "
        "<b>3:2</b>.</li>"
        "</ul>"
        '<div class="seq">Because deals aren\'t shared and the dealer keeps a fixed house '
        "edge, a <b>negative field net is expected</b> and short runs are luck-heavy — so "
        "models are ranked by <b>mean profit per hand</b>, not total. Bust / double / "
        "natural rates show how soundly each one plays.</div>"
        '<div class="seq"><b>What the model sees each turn:</b> its own cards and running total, '
        "the dealer's up-card, and the legal actions (hit / stand / double) — not the dealer's "
        "hole card or the rest of the shoe.</div>"
        "</div>"),
    "leduc_poker": (
        '<input type="checkbox" class="rules-toggle" id="rules-toggle" hidden>'
        '<label class="rules-summary" for="rules-toggle">Setup &amp; rules<span class="rules-hint"> · expand</span></label>'
        '<div class="rules">'
        "<h3>How Leduc Holdem works</h3>"
        "A tiny <b>imperfect-information</b> poker on a 6-card deck — two each of "
        '<span class="card">J</span><span class="card">Q</span><span class="card">K</span> '
        "(J &lt; Q &lt; K). Heads-up, two betting rounds:"
        "<ul>"
        "<li>Both players <b>ante 1 chip</b>, then each is dealt <b>one private card</b>.</li>"
        "<li><b>Round 1 (pre-flop):</b> players bet on their private card alone. "
        "Actions are <code>check</code>/<code>bet</code> and <code>call</code>/"
        "<code>raise</code>/<code>fold</code>; bets this round are <b>2 chips</b>.</li>"
        "<li>One <b>public card</b> is then revealed (the “flop”), shared by both.</li>"
        "<li><b>Round 2:</b> another betting round, now with bets of <b>4 chips</b>.</li>"
        "<li>At <b>showdown</b>, pairing your private card with the public card wins; "
        "otherwise the <b>higher card</b> wins. A fold concedes the pot.</li>"
        "</ul>"
        '<div class="seq">Small enough to reason about precisely, yet it has genuine '
        "bluffing, value-betting and pot odds — pairing the board is the nuts, and a "
        "lone King bluffs well. Rated by a <b>chip-weighted Elo</b> (like Hold'em), so "
        "the size of pots won/lost matters, not just who won the hand.</div>"
        '<div class="seq"><b>What the model sees each turn:</b> its own private card, the public '
        "card once it is revealed, the pot and the bets so far, and its legal actions — never the "
        "opponent's card.</div>"
        "</div>"),
    "repeated_colonel_blotto": (
        '<input type="checkbox" class="rules-toggle" id="rules-toggle" hidden>'
        '<label class="rules-summary" for="rules-toggle">Setup &amp; rules<span class="rules-hint"> · expand</span></label>'
        '<div class="rules">'
        "<h3>How Colonel Blotto works</h3>"
        "A two-player game of <b>simultaneous</b> resource allocation, played over "
        "<b>20 rounds</b>. Each round:"
        "<ul>"
        "<li>Both players <b>secretly</b> split <b>100 troops</b> across <b>5 "
        "battlefields</b> worth <b>1, 2, 3, 4 and 5 points</b> (15 points up for grabs "
        "per round).</li>"
        "<li>Allocations are revealed <b>at the same time</b> — neither sees the other's "
        "before committing.</li>"
        "<li>On each battlefield the <b>larger force wins</b> that field's points; an "
        "exact tie awards the field to neither.</li>"
        "<li>Whoever takes the <b>most points</b> wins the round; only resolved past "
        "rounds are shown to the opponent.</li>"
        "</ul>"
        '<div class="seq">After 20 rounds the <b>higher cumulative score wins</b> '
        "(equal totals draw). No hidden cards or chance — the difficulty is purely "
        "strategic: spread thin to grab the cheap fields, or stack up to guarantee the "
        "expensive ones, while reading and misdirecting the opponent across rounds.</div>"
        '<div class="seq"><b>What the model sees each round:</b> the 5 battlefields and their point '
        "values, its 100-troop budget, and the results of past rounds — not the opponent's current "
        "allocation (both commit at the same time).</div>"
        "</div>"),
    "othello_lite_6x6": (
        "Othello 6×6 is a <b>perfect-information</b> board game — flank to flip discs, "
        "most discs at the end wins. Late-game swings make lookahead and stable-disc "
        "control decisive. Each turn the model sees the full current board and its legal "
        "moves (perfect information)."),
}

NAV_HEAD = '<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="nav.css?v=7"><script defer src="nav.js?v=32"></script>'


def _intro_html(game: str) -> str:
    """Structured .rules block as-is, else wrap the plain blurb in a .callout."""
    body = INTRO[game]
    return body if body.lstrip().startswith("<") else f'<div class="callout">{body}</div>'


def _favicon(emoji: str) -> str:
    return (f"<link rel=\"icon\" href=\"data:image/svg+xml,"
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
            f"<text y='.9em' font-size='90'>{emoji}</text></svg>\">")


# ---------------------------------------------------------------------------
def bradley_terry(models, h2h, iters=300):
    """Fit Bradley-Terry strengths from pairwise results; return (strength, elo).

    Draws count as half a win to each side. Elo is the BT log-strength on the
    400/decade scale, recentred so the rated field averages 1500. A model with
    no wins/draws (or no losses/draws) has no finite rating and is reported with
    elo=None (rendered as "—"), excluded from the fit so a degenerate record does
    not drag the whole field to nonsense. Same fit as the board-game report.
    """
    wins = {m: 0.0 for m in models}
    losses = {m: 0.0 for m in models}
    draws = {m: 0.0 for m in models}
    for a in models:
        for b in models:
            if a == b:
                continue
            w, l, d = h2h[a][b]
            wins[a] += w
            losses[a] += l
            draws[a] += d

    rated = [m for m in models
             if (wins[m] + draws[m]) > 0 and (losses[m] + draws[m]) > 0]

    W = {m: 0.0 for m in rated}
    N = defaultdict(lambda: defaultdict(float))
    for a in rated:
        for b in rated:
            if a == b:
                continue
            w, l, d = h2h[a][b]
            W[a] += w + 0.5 * d
            N[a][b] += w + l + d

    p = {m: 1.0 for m in rated}
    for _ in range(iters):
        newp = {}
        for i in rated:
            denom = sum(N[i][j] / (p[i] + p[j])
                        for j in rated if j != i and N[i][j])
            newp[i] = (W[i] / denom) if denom > 0 else p[i]
        gm = math.exp(sum(math.log(max(v, 1e-9)) for v in newp.values())
                      / len(newp)) if newp else 1.0
        p = {i: newp[i] / gm for i in rated}

    elo = {m: None for m in models}
    if rated:
        raw = {m: 400 * math.log10(max(p[m], 1e-9)) for m in rated}
        mean = sum(raw.values()) / len(raw)
        for m in rated:
            elo[m] = int(round(1500 + raw[m] - mean))
    return p, elo


def _histogram(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            hi_ok = v < edges[i + 1] or (i == len(edges) - 2 and v <= edges[i + 1])
            if edges[i] <= v and hi_ok:
                counts[i] += 1
                break
    return counts


def _len_bins(all_lengths):
    lo, hi = min(all_lengths), max(all_lengths)
    nbins = min(12, max(hi - lo + 1, 1))
    step = max((hi - lo) / nbins, 1)
    edges = [lo + i * step for i in range(nbins + 1)]
    edges[-1] = hi
    labels = [f"{int(round(edges[i]))}–{int(round(edges[i + 1]))}"
              for i in range(len(edges) - 1)]
    return edges, labels


def _elo_txt(e):
    return "—" if e is None else str(e)


def _elo_key(elo, m):
    return elo[m] if elo[m] is not None else float("-inf")


def _latency(step):
    return (step.get("response") or {}).get("metadata", {}).get("latency_ms")


# ---------------------------------------------------------------------------
def analyze_versus(game: str, data: dict) -> dict:
    models = data["models"]
    stats = {m: {"games": 0, "wins": 0, "losses": 0, "draws": 0, "net": 0.0,
                 "decisions": 0, "invalid": 0, "lengths": [], "latencies": [],
                 "first_moves": 0, "first_move_wins": 0} for m in models}
    h2h = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [w,l,d] a vs b
    elo_records = []  # per-game (a, b, result) for the win/loss Elo bootstrap
    # chip-basis accumulators (used when the game scores in chips, e.g. Leduc)
    gross = defaultdict(lambda: defaultdict(float))
    chip_records = []  # per-game (a, b, chips_a, chips_b) for the chip Elo
    fp_games = fp_wins = 0

    for pair in data["pairs"]:
        a, b = pair["a"], pair["b"]
        for e in pair["episodes"]:
            seat = e["seat_assignment"]
            winner = e.get("winner_name")
            length = e["length"]
            name_seat = {v: k for k, v in seat.items()}
            ca, cb = e["returns"][name_seat[a]], e["returns"][name_seat[b]]
            gross[a][b] += max(ca, 0.0); gross[b][a] += max(cb, 0.0)
            chip_records.append((a, b, ca, cb))
            for s_, nm in seat.items():
                st = stats[nm]
                st["games"] += 1
                st["lengths"].append(length)
                pay = e["returns"][s_]
                st["net"] += pay
                if pay > 0:
                    st["wins"] += 1
                elif pay < 0:
                    st["losses"] += 1
                else:
                    st["draws"] += 1

            if winner == a:
                h2h[a][b][0] += 1; h2h[b][a][1] += 1
                elo_records.append((a, b, 1))
            elif winner == b:
                h2h[b][a][0] += 1; h2h[a][b][1] += 1
                elo_records.append((a, b, -1))
            else:
                h2h[a][b][2] += 1; h2h[b][a][2] += 1
                elo_records.append((a, b, 0))

            steps = e["steps"]
            if steps:
                fm = steps[0]["agent_name"]
                stats[fm]["first_moves"] += 1
                fp_games += 1
                if winner == fm:
                    stats[fm]["first_move_wins"] += 1
                    fp_wins += 1
            for s in steps:
                nm = s["agent_name"]
                st = stats[nm]
                st["decisions"] += 1
                if s.get("invalid"):
                    st["invalid"] += 1
                lat = _latency(s)
                if lat:
                    st["latencies"].append(lat)

    all_lengths = [ln for m in models for ln in stats[m]["lengths"]]
    edges, bin_labels = _len_bins(all_lengths)

    out = {}
    for m in models:
        st = stats[m]
        n = max(st["games"], 1)
        out[m] = {
            "games": st["games"], "wins": st["wins"], "losses": st["losses"],
            "draws": st["draws"],
            "win_rate": round(st["wins"] / n, 4),
            "draw_rate": round(st["draws"] / n, 4),
            "net_per_game": round(st["net"] / n, 4),
            "first_move_win_rate": round(
                st["first_move_wins"] / max(st["first_moves"], 1), 4),
            "invalid_rate": round(st["invalid"] / max(st["decisions"], 1), 4),
            "decisions": st["decisions"],
            "avg_len": round(sum(st["lengths"]) / len(st["lengths"]), 2)
                       if st["lengths"] else 0,
            "avg_latency_s": round(sum(st["latencies"]) / len(st["latencies"]) / 1000, 1)
                             if st["latencies"] else 0.0,
            "len_hist": _histogram(st["lengths"], edges),
        }

    basis = GAMES[game].get("elo_basis", "wins")
    if basis == "chips":
        # Poker: rate by chips won (magnitude matters), like Hold'em 1-Hand.
        chip_h2h = {a: {b: (gross[a][b], gross[b][a], 0.0)
                        for b in models if b != a} for a in models}
        _, elo = bradley_terry(models, chip_h2h)
        elo_ci = bootstrap_elo(models, chip_records,
                               lambda s: gross_from_records(models, s))
    else:
        _, elo = bradley_terry(models, h2h)
        elo_ci = bootstrap_elo(models, elo_records,
                               lambda s: wld_from_records(models, s))
    h2h_out = {a: {b: h2h[a][b] for b in models if b != a} for a in models}
    return {
        "game": game, "kind": "versus", "models": models, "per_model": out,
        "h2h": h2h_out, "elo": elo, "elo_ci": elo_ci, "elo_basis": basis,
        "len_bins": bin_labels,
        "first_player_win_rate": round(fp_wins / max(fp_games, 1), 4),
        "num_games": sum(len(p["episodes"]) for p in data["pairs"]),
        "episodes_per_pair": data.get("episodes_per_pair"),
        "avg_tokens": tokens_from_episodes(
            ep for p in data["pairs"] for ep in p["episodes"]),
    }


def analyze_dealer(game: str, data: dict) -> dict:
    run_dir = os.path.join("runs", GAMES[game]["dir"])
    # The per-game data.json roster can lag the episode dirs (e.g. a model added
    # after the aggregate was last written). Trust the dirs on disk as the source
    # of truth: keep data["models"] order, then append any extra dealer opponents.
    discovered = [os.path.basename(d)[: -len("__vs__dealer")]
                  for d in sorted(glob.glob(os.path.join(run_dir, "*__vs__dealer")))
                  if os.path.isdir(d)]
    models = list(data["models"]) + [m for m in discovered
                                     if m not in data["models"]]
    out = {}
    for m in models:
        eps = sorted(glob.glob(os.path.join(run_dir, f"{m}__vs__dealer", "ep*.json")))
        hands = wins = losses = pushes = busts = doubles = naturals = 0
        invalid = decisions = 0
        profit = 0.0
        lats = []
        tok_sum = tok_n = 0
        for path in eps:
            e = json.load(open(path))
            # The model always occupies player_0 against the dealer.
            pay = e["returns"]["player_0"]
            hands += 1
            profit += pay
            if pay > 0:
                wins += 1
            elif pay < 0:
                losses += 1
            else:
                pushes += 1
            if e.get("player_bust"):
                busts += 1
            if e.get("doubled"):
                doubles += 1
            if e.get("player_natural"):
                naturals += 1
            for s in e.get("steps", []):
                if s.get("agent_name") != m:
                    continue
                decisions += 1
                if s.get("invalid"):
                    invalid += 1
                lat = _latency(s)
                if lat:
                    lats.append(lat)
                ct = ((s.get("response") or {}).get("metadata") or {}).get("completion_tokens")
                if isinstance(ct, (int, float)):
                    tok_sum += ct; tok_n += 1
        h = max(hands, 1)
        out[m] = {
            "hands": hands, "profit": round(profit, 2),
            "mean_per_hand": round(profit / h, 4),
            "win_rate": round(wins / h, 4), "loss_rate": round(losses / h, 4),
            "push_rate": round(pushes / h, 4),
            "bust_rate": round(busts / h, 4), "double_rate": round(doubles / h, 4),
            "natural_rate": round(naturals / h, 4),
            "invalid_rate": round(invalid / max(decisions, 1), 4),
            "avg_latency_s": round(sum(lats) / len(lats) / 1000, 1) if lats else 0.0,
        }
        out[m]["avg_tokens"] = round(tok_sum / tok_n) if tok_n else 0
    total_hands = sum(out[m]["hands"] for m in models)
    field_profit = sum(out[m]["profit"] for m in models)
    return {
        "game": game, "kind": "dealer", "models": models, "per_model": out,
        "total_hands": total_hands, "field_profit": round(field_profit, 2),
        "avg_tokens": {m: out[m]["avg_tokens"] for m in models},
    }


# ---------------------------------------------------------------------------
_HEAD_CSS = BASE_CSS


def render_versus(rep: dict) -> str:
    cfg = GAMES[rep["game"]]
    emoji, name = cfg["title"].split(" ", 1)  # single leading emoji, Hold'em-style header
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    elo = rep["elo"]
    elo_ci = rep.get("elo_ci", {})
    ranked = sorted(models, key=lambda m: _elo_key(elo, m), reverse=True)

    def _elo_cell(m):
        if elo[m] is None:
            return "—"
        sd = (elo_ci.get(m) or {}).get("sd")
        return f"{elo[m]}<div class='small'>±{sd:.0f}</div>" if sd is not None else str(elo[m])

    # Rank by Elo (opponent-adjusted), with net/game as the tiebreaker.
    order = sorted(models, key=lambda m: (_elo_key(elo, m), pm[m]["net_per_game"]),
                   reverse=True)
    rows = ""
    for i, m in enumerate(order, 1):
        s = pm[m]
        net_cls = "pos" if s["net_per_game"] > 0 else ("neg" if s["net_per_game"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{model_cell(m)}</td>
          <td>{_elo_cell(m)}</td>
          <td class='{net_cls}'>{s['net_per_game']:+.2f}</td>
          <td>{s['win_rate']*100:.0f}%</td><td>{s['draw_rate']*100:.0f}%</td>
          <td>{s['first_move_win_rate']*100:.0f}%</td>
          <td>{s['invalid_rate']*100:.1f}%</td>
          <td>{s['avg_len']:.1f}</td><td>{s['avg_latency_s']:.1f}s</td>
          <td>{s['games']}</td>
          {token_cost_cells(m, rep.get('avg_tokens', {}).get(m))}
        </tr>"""

    hh = "<tr><th></th>" + "".join(f"<th>{display_name(m)}</th>" for m in models) + "</tr>"
    for a in models:
        hh += f"<tr><th class='model'>{model_cell(a)}</th>"
        for b in models:
            if a == b:
                hh += "<td class='diag'>—</td>"
            else:
                w, l, d = rep["h2h"][a][b]
                cls = "pos" if w > l else ("neg" if l > w else "")
                hh += f"<td class='{cls}'>{w}-{l}<div class='small'>{d}d</div></td>"
        hh += "</tr>"

    fpw = rep["first_player_win_rate"] * 100
    deals = (rep["episodes_per_pair"] or 0) // 2
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {cfg['title']}</title>
{_favicon(cfg['emoji'])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>{_HEAD_CSS}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/{cfg['area']}<span class="cursor"></span></h1>
  <div class="sub">{emoji} {name} · {cfg['blurb']} · {rep['num_games']} games</div>
  <a class="replaybtn" href="{cfg['replay']}?cacheBust=19">🎬 Watch featured replays →</a>
  {_intro_html(rep['game'])}


  <h2>🏆 Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>Elo</th><th>net/game</th><th>win%</th>
        <th>draw%</th><th>1st-move win%</th><th>invalid%</th><th>plies</th><th>think</th><th>games</th>{TOKEN_HEADERS}</tr>
    {rows}
  </table>
  {_legend('versus')}

  <div class="grid2">
    <div><h3>Elo rating</h3><canvas id="elo"></canvas></div>
    <div><h3>Win / draw / loss</h3><canvas id="wdl"></canvas></div>
  </div>
  <div class="grid2">
    <div><h3>Net result per game</h3><canvas id="net"></canvas></div>
    <div><h3>Game-length distribution (plies)</h3><canvas id="len"></canvas></div>
  </div>

  <h2>⚔️ Head-to-head (row wins–losses vs column)</h2>
  <table class='h2h'>{hh}</table>

<script>
const R = {payload};
const pm=R.per_model, M=R.models;
const DN = {json.dumps({m: display_name(m) for m in models})};
const dn = m => DN[m] || m;
{CHART_SETUP}
const COLORS=PALETTE;
const eloRanked=[...M].filter(m=>R.elo[m]!=null).sort((a,b)=>R.elo[b]-R.elo[a]);
const ELO_CI = R.elo_ci || {{}};
// Horizontal ±1-SD whisker over each Elo bar (bootstrap uncertainty).
const eloWhiskers = {{ id:'eloWhiskers', afterDatasetsDraw(c) {{
  const {{ctx, scales:{{x, y}}}} = c;
  ctx.save(); ctx.strokeStyle='#1c1c1c'; ctx.lineWidth=1.5;
  eloRanked.forEach((m, i) => {{
    const ci = ELO_CI[m]; if (!ci || ci.sd == null) return;
    const yc = y.getPixelForValue(i), cap = 5;
    const x1 = x.getPixelForValue(R.elo[m]-ci.sd), x2 = x.getPixelForValue(R.elo[m]+ci.sd);
    ctx.beginPath();
    ctx.moveTo(x1, yc); ctx.lineTo(x2, yc);
    ctx.moveTo(x1, yc-cap); ctx.lineTo(x1, yc+cap);
    ctx.moveTo(x2, yc-cap); ctx.lineTo(x2, yc+cap);
    ctx.stroke();
  }});
  ctx.restore();
}} }};
new Chart(document.getElementById('elo'), {{ type:'bar',
  data:{{ labels:eloRanked.map(dn), datasets:[{{label:'Elo', backgroundColor:ACCENT,
    data:eloRanked.map(m=>R.elo[m])}}]}},
  options:{{ indexAxis:'y',
    scales:{{x:{{min:eloRanked.length?Math.min(...eloRanked.map(m=>R.elo[m]-(ELO_CI[m]?.sd||0)))-20:0}}}},
    plugins:{{legend:{{display:false}}}} }}, plugins:[eloWhiskers] }});

new Chart(document.getElementById('wdl'), {{ type:'bar',
  data:{{ labels:M.map(dn), datasets:[
    {{label:'win %', backgroundColor:'#4ade80', data:M.map(m=>100*pm[m].wins/(pm[m].games||1))}},
    {{label:'draw %', backgroundColor:'#94a3b8', data:M.map(m=>100*pm[m].draws/(pm[m].games||1))}},
    {{label:'loss %', backgroundColor:'#f87171', data:M.map(m=>100*pm[m].losses/(pm[m].games||1))}},
  ]}},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100,title:{{display:true,text:'% of games'}}}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});

new Chart(document.getElementById('net'), {{ type:'bar',
  data:{{ labels:M.map(dn), datasets:[{{label:'net/game', backgroundColor:M.map(m=>pm[m].net_per_game>=0?'#4ade80':'#f87171'),
    data:M.map(m=>pm[m].net_per_game)}}]}},
  options:{{ plugins:{{legend:{{display:false}}}} }} }});

new Chart(document.getElementById('len'), {{ type:'bar',
  data:{{ labels:R.len_bins, datasets:M.map((m,i)=>({{label:m,
    backgroundColor:COLORS[i%COLORS.length], data:pm[m].len_hist}}))}},
  options:{{ scales:{{y:{{title:{{display:true,text:'games'}}}}}},
    plugins:{{legend:{{position:'bottom'}}}} }} }});
</script>
</div></body></html>"""


def render_dealer(rep: dict) -> str:
    cfg = GAMES[rep["game"]]
    emoji, name = cfg["title"].split(" ", 1)  # single leading emoji, Hold'em-style header
    payload = json.dumps(rep)
    pm = rep["per_model"]
    models = rep["models"]
    ranked = sorted(models, key=lambda m: pm[m]["mean_per_hand"], reverse=True)

    rows = ""
    for i, m in enumerate(ranked, 1):
        s = pm[m]
        pcls = "pos" if s["mean_per_hand"] > 0 else ("neg" if s["mean_per_hand"] < 0 else "")
        rows += f"""<tr>
          <td>{i}</td><td class='model'>{model_cell(m)}</td>
          <td class='{pcls}'>{s['mean_per_hand']:+.3f}</td>
          <td>{s['win_rate']*100:.0f}%</td><td>{s['push_rate']*100:.0f}%</td>
          <td>{s['loss_rate']*100:.0f}%</td>
          <td>{s['bust_rate']*100:.0f}%</td><td>{s['double_rate']*100:.0f}%</td>
          <td>{s['natural_rate']*100:.0f}%</td>
          <td>{s['invalid_rate']*100:.1f}%</td>
          <td>{s['hands']}</td><td>{s['avg_latency_s']:.1f}s</td>
          {token_cost_cells(m, rep.get('avg_tokens', {}).get(m))}
        </tr>"""

    champ = ranked[0]
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Battle Arena — {cfg['title']}</title>
{_favicon(cfg['emoji'])}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{NAV_HEAD}
<style>{_HEAD_CSS}</style></head>
<body><div class="wrap">
  <h1>$ ~/aibattle/{cfg['area']}<span class="cursor"></span></h1>
  <div class="sub">{emoji} {name} · {cfg['blurb']} · {rep['total_hands']} hands total</div>
  <a class="replaybtn" href="{cfg['replay']}?cacheBust=19">🎬 Watch featured replays →</a>
  {_intro_html(rep['game'])}


  <h2>🏆 Leaderboard</h2>
  <table>
    <tr><th>#</th><th class='model'>model</th><th>mean/hand</th>
        <th>win%</th><th>push%</th><th>loss%</th><th>bust%</th><th>double%</th>
        <th>natural%</th><th>invalid%</th><th>hands</th><th>think</th>{TOKEN_HEADERS}</tr>
    {rows}
  </table>
  {_legend('blackjack')}
  <div class="note">Each model plays independent hands against the built-in dealer; the dealer holds an
    inherent house edge, so a negative field average is expected. mean/hand = average chips per hand
    (a doubled hand pays ±2), the fair comparison since hand counts can differ. bust = player busted;
    double = chose to double down; natural = dealt a blackjack. Read directly from per-hand outcomes.</div>

  <div class="grid2">
    <div><h3>Mean chips per hand</h3><canvas id="profit"></canvas></div>
    <div><h3>Win / push / loss</h3><canvas id="wpl"></canvas></div>
  </div>
  <div class="grid2">
    <div><h3>Play style (bust / double / natural %)</h3><canvas id="style"></canvas></div>
    <div></div>
  </div>

<script>
const R = {payload};
const pm=R.per_model, M=R.models;
const DN = {json.dumps({m: display_name(m) for m in models})};
const dn = m => DN[m] || m;
{CHART_SETUP}
const ranked=[...M].sort((a,b)=>pm[b].mean_per_hand-pm[a].mean_per_hand);

new Chart(document.getElementById('profit'), {{ type:'bar',
  data:{{ labels:ranked, datasets:[{{label:'mean/hand', backgroundColor:ranked.map(m=>pm[m].mean_per_hand>=0?'#1a7f37':'#b91c1c'),
    data:ranked.map(m=>pm[m].mean_per_hand)}}]}},
  options:{{ indexAxis:'y', plugins:{{legend:{{display:false}}}} }} }});

new Chart(document.getElementById('wpl'), {{ type:'bar',
  data:{{ labels:M.map(dn), datasets:[
    {{label:'win', backgroundColor:'#4ade80', data:M.map(m=>pm[m].win_rate*100)}},
    {{label:'push', backgroundColor:'#94a3b8', data:M.map(m=>pm[m].push_rate*100)}},
    {{label:'loss', backgroundColor:'#f87171', data:M.map(m=>pm[m].loss_rate*100)}},
  ]}},
  options:{{ scales:{{x:{{stacked:true}},y:{{stacked:true,max:100}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});

new Chart(document.getElementById('style'), {{ type:'bar',
  data:{{ labels:M.map(dn), datasets:[
    {{label:'bust %', backgroundColor:'#f87171', data:M.map(m=>pm[m].bust_rate*100)}},
    {{label:'double %', backgroundColor:'#fbbf24', data:M.map(m=>pm[m].double_rate*100)}},
    {{label:'natural %', backgroundColor:'#60a5fa', data:M.map(m=>pm[m].natural_rate*100)}},
  ]}},
  options:{{ scales:{{y:{{min:0}}}}, plugins:{{legend:{{position:'bottom'}}}} }} }});
</script>
</div></body></html>"""


# ---------------------------------------------------------------------------
def _index_entry(rep: dict) -> dict:
    """The compact record analyze_board_tournament.render_index() consumes to add
    this game as a landing-page card and a row in the cross-game Arena Score."""
    cfg = GAMES[rep["game"]]
    pm = rep["per_model"]
    base = {"key": rep["game"], "title": cfg["title"], "href": cfg["href"],
            "group": cfg["group"], "badges": cfg["badges"]}
    if rep["kind"] == "versus":
        ranked = sorted(rep["models"], key=lambda m: _elo_key(rep["elo"], m), reverse=True)
        champ = ranked[0]
        deals = (rep["episodes_per_pair"] or 0) // 2
        base.update({
            "ranking": ranked,
            # Per-model rating for the cross-game Elo composite (chip/Bradley-Terry
            # Elo; None for unrated models is dropped when standardizing).
            "ratings": {m: rep["elo"].get(m) for m in rep["models"]},
            "meta": (f"{rep['num_games']} games · {deals} deals/pair · "
                     f"first-mover {rep['first_player_win_rate']*100:.0f}%"),
            "champ_line": f"🏆 {champ} <span class='metric'>Elo {_elo_txt(rep['elo'][champ])}</span>",
        })
    else:  # dealer
        ranked = sorted(rep["models"], key=lambda m: pm[m]["mean_per_hand"], reverse=True)
        champ = ranked[0]
        per_model_hands = rep["per_model"][champ]["hands"]
        base.update({
            "ranking": ranked,
            # Blackjack has no head-to-head Elo (it's vs the dealer), so the
            # composite standardizes mean chips/hand instead.
            "ratings": {m: pm[m]["mean_per_hand"] for m in rep["models"]},
            "meta": f"vs dealer · {per_model_hands} hands/model · mean chips/hand",
            "champ_line": f"🏆 {champ} <span class='metric'>{pm[champ]['mean_per_hand']:+.3f}/hand</span>",
        })
    return base


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    index_entries = []
    for game, cfg in GAMES.items():
        path = os.path.join("runs", cfg["dir"], "data.json")
        if not os.path.exists(path):
            print(f"skip {game}: no data at {path}")
            continue
        data = json.load(open(path))
        if cfg["kind"] == "versus":
            if not data.get("pairs"):
                print(f"skip {game}: no completed pairs yet")
                continue
            rep = analyze_versus(game, data)
            html = render_versus(rep)
        else:
            rep = analyze_dealer(game, data)
            if rep["total_hands"] == 0:
                print(f"skip {game}: no hands found under runs/{cfg['dir']}")
                continue
            html = render_dealer(rep)

        out = os.path.join(REPORT_DIR, cfg["href"])
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        json.dump(rep, open(os.path.join(REPORT_DIR, f"{game}_analysis.json"), "w"),
                  indent=2)
        index_entries.append(_index_entry(rep))
        print(f"Wrote {out}")

        pm = rep["per_model"]
        if rep["kind"] == "versus":
            order = sorted(pm, key=lambda x: _elo_key(rep["elo"], x), reverse=True)
            print(f"=== {game} ({rep['num_games']} games) ===")
            for m in order:
                s = pm[m]
                print(f"  {m:<16} elo={_elo_txt(rep['elo'][m]):>4} "
                      f"net/g={s['net_per_game']:+.2f} win%={s['win_rate']*100:3.0f} "
                      f"draw%={s['draw_rate']*100:3.0f} invalid={s['invalid_rate']*100:.1f}%")
        else:
            order = sorted(pm, key=lambda x: pm[x]["mean_per_hand"], reverse=True)
            print(f"=== {game} ({rep['total_hands']} hands) ===")
            for m in order:
                s = pm[m]
                print(f"  {m:<16} profit={s['profit']:+.1f} mean/hand={s['mean_per_hand']:+.3f} "
                      f"win%={s['win_rate']*100:3.0f} bust%={s['bust_rate']*100:3.0f} "
                      f"invalid={s['invalid_rate']*100:.1f}%")

    if index_entries:
        json.dump(index_entries,
                  open(os.path.join(REPORT_DIR, "new_games_index.json"), "w"), indent=2)
        print(f"\nWrote {REPORT_DIR}/new_games_index.json ({len(index_entries)} games)")
        print("Now run analyze_board_tournament.py to refresh index.html with these entries.")


if __name__ == "__main__":
    main()
