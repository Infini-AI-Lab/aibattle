"""Shared Bradley-Terry → Elo fit over head-to-head records.

Several reports (board games, leduc, blotto, othello) already rate models this
way; this module is the single source of truth so the holdem and kuhn reports
rate models on the *same* scale. Elo is the right comparison when models face
different opponents (e.g. an incomplete round-robin): it weights a win by how
strong the beaten opponent is, which a raw win% or chip total cannot.

h2h format: h2h[a][b] = (wins, losses, draws) of a against b. Draws count as
half a win to each side. Elo is the BT log-strength on the 400/decade scale,
recentred so the rated field averages 1500. A model with no wins+draws (or no
losses+draws) has no finite rating and is reported elo=None (render as "—"),
excluded from the fit so a degenerate record cannot drag the field to nonsense.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict


def bradley_terry(models, h2h, iters=300):
    """Fit Bradley-Terry strengths from pairwise (w, l, d) records.

    Returns (strength, elo): strength is the raw BT parameter per rated model,
    elo maps every model to an int rating (or None when unrated).
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


def elo_key(elo, m):
    """Sort key that pushes unrated models (elo=None) to the bottom."""
    return elo[m] if elo.get(m) is not None else float("-inf")


def wld_from_records(models, records):
    """Build {a: {b: (w, l, d)}} from (a, b, result) games (+1 a wins, -1 b, 0 draw)."""
    h = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for a, b, r in records:
        if r > 0:
            h[a][b][0] += 1; h[b][a][1] += 1
        elif r < 0:
            h[a][b][1] += 1; h[b][a][0] += 1
        else:
            h[a][b][2] += 1; h[b][a][2] += 1
    return {a: {b: tuple(h[a][b]) for b in models if b != a} for a in models}


def gross_from_records(models, records):
    """Build chip-weighted {a: {b: (a_gross, b_gross, 0)}} from (a, b, ca, cb) games."""
    g = defaultdict(lambda: defaultdict(float))
    for a, b, ca, cb in records:
        g[a][b] += max(ca, 0.0)
        g[b][a] += max(cb, 0.0)
    return {a: {b: (g[a][b], g[b][a], 0.0) for b in models if b != a} for a in models}


def bootstrap_elo(models, records, build_h2h, n_boot=300, ci=0.95, seed=1234):
    """Bootstrap Elo uncertainty by resampling games with replacement.

    `records` is a flat list of independent per-game outcome items; `build_h2h`
    turns a resampled list into the {a: {b: (w, l, d)}} map bradley_terry wants
    (the same construction the caller uses for its point estimate, so this works
    for both win/loss and chip-weighted fits). Each of `n_boot` resamples is
    refit; the spread of a model's ratings across fits is its uncertainty.

    Returns {model: {"sd": float|None, "lo": int|None, "hi": int|None}} — the
    bootstrap standard deviation (the error bar) and the central `ci` percentile
    interval. None when a model is unrated in too few resamples to summarise.
    """
    rng = random.Random(seed)
    n = len(records)
    draws = {m: [] for m in models}
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        _, elo = bradley_terry(models, build_h2h(sample))
        for m in models:
            if elo[m] is not None:
                draws[m].append(elo[m])
    lo_q, hi_q = (1 - ci) / 2, (1 + ci) / 2
    out = {}
    for m in models:
        xs = sorted(draws[m])
        if len(xs) < 2:
            out[m] = {"sd": None, "lo": None, "hi": None}
            continue
        mean = sum(xs) / len(xs)
        sd = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        out[m] = {"sd": round(sd, 1),
                  "lo": xs[max(0, int(lo_q * len(xs)))],
                  "hi": xs[min(len(xs) - 1, int(hi_q * len(xs)))]}
    return out


def elo_from_wld(models, wld):
    """Convenience: return just the elo dict for a {a: {b: (w, l, d)}} map."""
    return bradley_terry(models, wld)[1]


def _solve(A, b):
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        if abs(pv) < 1e-12:
            continue
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / pv
            if f:
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def margin_rating(models, margins):
    """Opponent-adjusted rating from signed margins (a Massey least-squares fit).

    margins[a][b] = average signed margin of a vs b (omit unplayed pairs). Finds
    ratings r minimising sum over played ordered pairs of (r_a - r_b - margin)^2,
    mean-centred to 0. Unlike a win/loss Elo this keeps *magnitude*: a model that
    wins big pots rates above one that wins many small ones. The least-squares
    fit also discounts an easy schedule, so it is fair when the round-robin is
    incomplete (a model that faced only part of the field).
    """
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    A = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for a in models:
        for b in models:
            if a == b:
                continue
            mv = margins.get(a, {}).get(b)
            if mv is None:
                continue
            i, j = idx[a], idx[b]
            A[i][i] += 1.0
            A[i][j] -= 1.0
            rhs[i] += mv
    # Replace the last normal equation with the mean-zero gauge constraint.
    A[-1] = [1.0] * n
    rhs[-1] = 0.0
    r = _solve(A, rhs)
    return {m: r[idx[m]] for m in models}
