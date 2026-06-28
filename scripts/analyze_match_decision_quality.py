"""Decision-quality factors for the Hold'em Match report (uses hole-card equity).

Every decision in the match logs carries the acting player's hole cards
(observation.private.hole) and the public board, so we can compute that
player's equity vs a random hand (Monte-Carlo). The opponent's cards are never
revealed, so this is equity-vs-random — a range-free *proxy*: good for gross
signals (is the aggression backed by cards? are there clear blunders?), not for
subtle EV. Writes reports/match_decision_quality.json with three blocks:

  fire_equity : per model, average equity when it bets/raises vs when it
                calls/checks, plus the equity mix of its aggressive actions —
                "is its aggression honest, or does it fire with air?"
  blunders    : per model, in all-in confrontations, how often it commits with
                near-dead equity (<20%) or folds the best of it (>55%) — clear,
                range-insensitive mistakes.
  adaptation  : per model, does it vary aggression by opponent and does it open
                up against passive opponents (exploitation)?

Postflop equity is sampled (P_POST) and preflop is cached, to stay tractable.
GPT-OSS is excluded (incomplete schedule), matching the rest of the report.
"""
from __future__ import annotations
import json, glob, random, collections

from aibattle.games.poker_eval import evaluate7, full_deck

GLOB = "runs/holdem_match/*__vs__*/ep*.json"
OUT = "reports/match_decision_quality.json"
EXCLUDE = {"gpt-oss-120b"}
AGG = ("bet", "raise", "all_in")
PASSIVE = ("call", "check")
N_MC = 160          # Monte-Carlo samples per equity estimate
P_POST = 0.18       # fraction of postflop decisions sampled for fire-equity
SEED = 7

_DECK = full_deck()
_cache: dict = {}


def norm(m): return m.replace("-coached", "")


def equity(hole, board, n=N_MC):
    """Actor equity vs one random opponent hand (cached by hole+board)."""
    key = (tuple(sorted(hole)), tuple(sorted(board)))
    if key in _cache:
        return _cache[key]
    used = set(hole) | set(board)
    avail = [c for c in _DECK if c not in used]
    need = 2 + (5 - len(board))
    w = t = 0
    for _ in range(n):
        s = random.sample(avail, need)
        opp, rb = s[:2], s[2:]
        full = board + rb
        a, b = evaluate7(hole + full), evaluate7(opp + full)
        if a > b:
            w += 1
        elif a == b:
            t += 1
    v = (w + t / 2) / n
    _cache[key] = v
    return v


def main():
    rng = random.Random(SEED)
    random.seed(SEED)
    files = glob.glob(GLOB)
    rng.shuffle(files)

    # fire_equity: model -> {agg:[sum,n], pas:[sum,n], bucket:Counter}  (buckets of agg equity)
    fire = collections.defaultdict(lambda: {"agg": [0.0, 0], "pas": [0.0, 0],
                                            "bucket": collections.Counter(),
                                            "bluff": [0, 0]})   # [opp folded, total bluffs]
    # blunders: model -> Counter(spots, dead_calls, bad_folds)
    blun = collections.defaultdict(collections.Counter)
    # adaptation: (model, opp) -> [agg, total];  and model -> [agg,total] overall
    pair_agg = collections.defaultdict(lambda: [0, 0])
    overall_agg = collections.defaultdict(lambda: [0, 0])

    for fi, f in enumerate(files):
        try:
            e = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        sa = {k: norm(v) for k, v in e["seat_assignment"].items()}
        if EXCLUDE & set(sa.values()):
            continue
        steps = e.get("steps", [])
        for i, s in enumerate(steps):
            m = norm(s.get("agent_name", "")); act = s.get("selected_action")
            if not act:
                continue
            obs = s.get("observation", {})
            pub = obs.get("public", {})
            hole = (obs.get("private") or {}).get("hole")
            board = pub.get("board") or []
            # --- adaptation (no equity): aggression overall + per opponent ---
            if act in AGG + PASSIVE:
                opp = next((v for k, v in sa.items() if v != m), None)
                overall_agg[m][0] += int(act in AGG); overall_agg[m][1] += 1
                if opp:
                    pa = pair_agg[(m, opp)]; pa[0] += int(act in AGG); pa[1] += 1
            if not hole:
                continue
            # --- blunders: clear mistakes in all-in confrontations ---
            facing_allin = pub.get("opp_all_in") or pub.get("to_call", 0) >= pub.get("your_stack", 1e9) * 0.8
            if facing_allin and pub.get("to_call", 0) > 0:
                eq = equity(hole, board, 200)
                b = blun[m]; b["spots"] += 1
                if act in ("call", "all_in") and eq < 0.20:
                    b["dead_calls"] += 1
                if act == "fold" and eq > 0.55:
                    b["bad_folds"] += 1
            # --- fire equity: sample (preflop always — cache makes it cheap) ---
            if act in AGG + PASSIVE:
                if board and random.random() > P_POST:
                    continue
                eq = equity(hole, board)
                fr = fire[m]
                if act in AGG:
                    fr["agg"][0] += eq; fr["agg"][1] += 1
                    bk = ("lt40" if eq < 0.40 else "mid" if eq < 0.60
                          else "gt60" if eq < 0.80 else "gt80")
                    fr["bucket"][bk] += 1
                    if eq < 0.40:   # a bluff — did the opponent fold to it?
                        resp = next((steps[j].get("selected_action")
                                     for j in range(i + 1, len(steps))
                                     if norm(steps[j].get("agent_name", "")) != m), None)
                        if resp is not None:
                            fr["bluff"][1] += 1
                            if resp == "fold":
                                fr["bluff"][0] += 1
                else:
                    fr["pas"][0] += eq; fr["pas"][1] += 1
        if fi % 500 == 0:
            print(f"  {fi}/{len(files)} (cache {len(_cache)})", flush=True)

    # finalize fire_equity
    out_fire = {}
    for m, fr in fire.items():
        a, p = fr["agg"], fr["pas"]
        tot = sum(fr["bucket"].values()) or 1
        bf = fr["bluff"]
        out_fire[m] = {
            "fire_eq": round(a[0] / a[1], 3) if a[1] else None,
            "passive_eq": round(p[0] / p[1], 3) if p[1] else None,
            "gap": round(a[0] / a[1] - p[0] / p[1], 3) if a[1] and p[1] else None,
            "n_agg": a[1],
            "mix": {k: round(fr["bucket"][k] / tot, 3) for k in ("lt40", "mid", "gt60", "gt80")},
            "bluff_success": round(bf[0] / bf[1], 3) if bf[1] else None,
            "bluff_n": bf[1],
        }
    # finalize blunders
    out_blun = {m: {"spots": c["spots"],
                    "dead_call_rate": round(c["dead_calls"] / c["spots"], 3) if c["spots"] else None,
                    "bad_fold_rate": round(c["bad_folds"] / c["spots"], 3) if c["spots"] else None,
                    "dead_calls": c["dead_calls"], "bad_folds": c["bad_folds"]}
               for m, c in blun.items()}
    # finalize adaptation — intuitive form: a model's aggression vs the *passive*
    # opponents vs vs the *aggressive* opponents. Bullying the soft ones (higher
    # aggression vs passive foes) is the smart, exploitative move.
    ov = {m: (v[0] / v[1] if v[1] else 0.0) for m, v in overall_agg.items()}
    field_med = sorted(ov.values())[len(ov) // 2] if ov else 0.5   # split opponents here
    out_adapt = {}
    for m in ov:
        passive_aggs, aggro_aggs = [], []   # my aggression vs passive / aggressive opponents
        for (mm, opp), v in pair_agg.items():
            if mm != m or v[1] < 50 or opp not in ov:
                continue
            (passive_aggs if ov[opp] <= field_med else aggro_aggs).append(v[0] / v[1])
        if not passive_aggs or not aggro_aggs:
            continue
        vp = sum(passive_aggs) / len(passive_aggs)
        va = sum(aggro_aggs) / len(aggro_aggs)
        out_adapt[m] = {
            "vs_passive": round(vp, 3), "vs_aggressive": round(va, 3),
            "bully_gap": round(vp - va, 3),     # >0 = more aggressive vs soft opponents (smart)
            "n_opps": len(passive_aggs) + len(aggro_aggs),
        }

    json.dump({"fire_equity": out_fire, "blunders": out_blun, "adaptation": out_adapt},
              open(OUT, "w"), indent=2)
    print(f"WROTE {OUT}: fire={len(out_fire)} blun={len(out_blun)} adapt={len(out_adapt)} models")


if __name__ == "__main__":
    main()
