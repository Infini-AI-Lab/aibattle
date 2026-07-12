"""Replay extracted must-block threat probes under modified prompt renderings.

The experiment: the board reports show miss-rate ordered horizontal < vertical
< diag_dr < diag_dl, and blame the row-major TEXT rendering. Two mechanisms
could explain the ↘/↙ diagonal asymmetry:

  E1 (perceptual): models read the 2-D layout of the text itself; ↙ is hard
      because it runs against the left-to-right reading gradient.
  E2 (coordinate arithmetic): models track (row, col) label indices; ↙ is hard
      because the two counters move in opposite directions.

Arm ``flip_rows`` separates them with a labels-only vertical flip: the board
lines stay in the same textual order, only the printed row labels change from
1..9 (top->bottom) to 9..1. Physically nothing moves, but in LABEL space every
physical ↘ line becomes "row decreases as column increases" (anti-diagonal) and
vice versa. So:

  E1 predicts miss-rates stay with the PHYSICAL layout (↘ still easier);
  E2 predicts they follow the LABELS (physical ↙, now label-↘, becomes easier).

Each probe replays the VERBATIM prompt stored in the run logs (arm ``base``)
and the transformed prompt (arm ``flip_rows``) — a paired design on identical
positions. Scoring maps the model's answer back through the arm's label frame;
answers that would only block under the WRONG frame are recorded as
``frame_slip`` so coordinate-frame confusion is separable from true misses.

Usage (repo root; needs PYTHONPATH=src and the .fireworks key file):
  python3 scripts/threat_probe_replay.py run \
      --models deepseek-v4-pro,kimi-k2p6 --axes diag_dr,diag_dl \
      --arms base,flip_rows [--limit 40] [--concurrency 16]
  python3 scripts/threat_probe_replay.py analyze

Results append to runs/threat_probes/replays_gomoku.jsonl (resumable: finished
(model, arm, probe) triples are skipped on re-run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if os.path.exists(".fireworks"):
    os.environ.setdefault("FIREWORKS_API_KEY", open(".fireworks").read().strip())

from aibattle.models.registry import make_client  # noqa: E402

PROBES = "runs/threat_probes/probes_gomoku.jsonl"
RESULTS = "runs/threat_probes/replays_gomoku.jsonl"
# The image arm keeps its own home: results + (with --save-images) the PNGs sent.
IMG_DIR = "runs/threat_probes/images"
IMG_RESULTS = os.path.join(IMG_DIR, "replays_gomoku_image.jsonl")


def results_path(arm: str) -> str:
    return IMG_RESULTS if arm == "image" else RESULTS
_COLS = "ABCDEFGHI"
_COORD = re.compile(r"\b([A-I])-?([1-9])\b", re.IGNORECASE)
_BOARD_LINE = re.compile(r"^ ?([1-9]) ((?:[XO.] ){8}[XO.])\s*$")
_REPAIR = ("Your previous reply was not a valid empty cell. Reply with one "
           "coordinate like E5 that is currently empty.")


# --- label frames -----------------------------------------------------------
# label_to_r / r_to_label for each arm (gomoku rows; columns never change).
# The image arm keeps normal labels; its board travels as a PNG, not text.
FRAMES = {
    "base": (lambda l: l - 1, lambda r: r + 1),
    "flip_rows": (lambda l: 9 - l, lambda r: 9 - r),
    "image": (lambda l: l - 1, lambda r: r + 1),
}

_IMG_POINTER = ("The current board is shown in the attached image: a 9x9 grid, "
                "columns A-I left to right, rows 1-9 top to bottom, with X and O "
                "marking the stones.")
_HEADER_LINE = re.compile(r"^   A B C D E F G H I\s*$")


def transform_prompt(prompt: str, arm: str) -> str:
    if arm == "base":
        return prompt
    out, labels = [], []
    for line in prompt.splitlines():
        m = _BOARD_LINE.match(line)
        if m:
            l = int(m.group(1))
            labels.append(l)
            if arm == "flip_rows":
                out.append(f"{10 - l:>2} {m.group(2)}")
            elif l == 1:  # image: replace the whole block with one pointer line
                out.append(_IMG_POINTER)
        elif not (arm == "image" and _HEADER_LINE.match(line)):
            out.append(line)
    if labels != list(range(1, 10)):
        raise ValueError(f"board block not found/complete in prompt (labels={labels})")
    return "\n".join(out)


# --- board image rendering (arm: image) --------------------------------------
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
CELL, MARGIN = 48, 44


def render_board_png(board) -> bytes:
    """9x9 grid with X/O glyphs, column letters on top, row numbers on the left —
    the same symbols as the text rendering, just laid out in true 2-D."""
    from PIL import Image, ImageDraw, ImageFont
    glyph = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSansMono-Bold.ttf", 30)
    label = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSans.ttf", 20)
    size = MARGIN + 9 * CELL + 12
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    for i in range(10):
        v = MARGIN + i * CELL
        d.line([(MARGIN, v), (MARGIN + 9 * CELL, v)], fill=(180, 180, 180))
        d.line([(v, MARGIN), (v, MARGIN + 9 * CELL)], fill=(180, 180, 180))
    for i in range(9):
        c = MARGIN + i * CELL + CELL // 2
        d.text((c, MARGIN // 2), _COLS[i], font=label, fill="black", anchor="mm")
        d.text((MARGIN // 2, c), str(i + 1), font=label, fill="black", anchor="mm")
        for j in range(9):
            cell = board[i][j]
            if cell:
                x = MARGIN + j * CELL + CELL // 2
                d.text((x, c), cell, font=glyph, fill="black", anchor="mm")
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def image_messages(text: str, png: bytes) -> list:
    import base64
    return [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{base64.b64encode(png).decode()}"}},
    ]}]


def parse_answer(raw: str, legal: set) -> str | None:
    """GomokuTemplate.parse logic: bottom-up, first line naming exactly ONE
    distinct legal coordinate."""
    if not raw:
        return None
    for ln in reversed([l for l in raw.splitlines() if l.strip()]):
        coords = {f"{m.group(1).upper()}{m.group(2)}" for m in _COORD.finditer(ln)} & legal
        if len(coords) == 1:
            return next(iter(coords))
    return None


# --- run --------------------------------------------------------------------
def load_probes(axes, limit, seed):
    probes = [json.loads(l) for l in open(PROBES)]
    probes = [p for p in probes if p["axis"] in axes]
    if limit:
        rng = random.Random(seed)
        by_axis = defaultdict(list)
        for p in probes:
            by_axis[p["axis"]].append(p)
        probes = []
        for ax in sorted(by_axis):
            pool = sorted(by_axis[ax], key=lambda p: p["id"])
            probes += pool if len(pool) <= limit else rng.sample(pool, limit)
    return probes


def done_keys():
    keys = set()
    for path in (RESULTS, IMG_RESULTS):
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l)
                keys.add((r["model"], r["arm"], r["id"]))
    return keys


async def replay_one(client, probe, arm, max_tokens, save_images=False):
    label_to_r, r_to_label = FRAMES[arm]
    prompt = transform_prompt(probe["prompt"], arm)
    legal = {f"{_COLS[c]}{r_to_label(r)}"
             for r in range(9) for c in range(9) if probe["board"][r][c] is None}

    png = render_board_png(probe["board"]) if arm == "image" else None
    if png is not None and save_images:
        fn = os.path.join(IMG_DIR, probe["id"].replace(":", "__") + ".png")
        if not os.path.exists(fn):
            with open(fn, "wb") as fh:
                fh.write(png)

    attempts, out, coord = 0, None, None
    p = prompt
    while attempts < 3:
        attempts += 1
        req = image_messages(p, png) if png else p
        out = await client.generate(req, max_tokens=max_tokens)
        coord = parse_answer(out.content, legal)
        if coord:
            break
        p = f"{prompt}\n\n{_REPAIR}"

    rec = {
        "model": None, "arm": arm, "id": probe["id"], "axis": probe["axis"],
        "coord": coord, "attempts": attempts,
        "truncated": bool(out.truncated), "completion_tokens": out.completion_tokens,
        "content_tail": (out.content or "")[-300:],
    }
    if coord is None:
        rec.update(parsed=False, blocked=False, won_instead=False,
                   missed=True, frame_slip=False)
        return rec
    r, c = label_to_r(int(coord[1:])), _COLS.index(coord[0])
    cell = tuple(probe["threat_cell"])
    blocked = (r, c) == cell
    won_instead = [r, c] in probe["own_win_cells"]
    # Would the answer have blocked under the OTHER frame? (frame confusion,
    # not detection failure)
    other = [a for a in FRAMES if a != arm][0]
    ro = FRAMES[other][0](int(coord[1:]))
    rec.update(parsed=True, blocked=blocked, won_instead=won_instead,
               missed=not blocked and not won_instead,
               frame_slip=not blocked and (ro, c) == cell)
    return rec


async def run(args):
    axes = args.axes.split(",")
    arms = args.arms.split(",")
    models = args.models.split(",")
    probes = load_probes(axes, args.limit, args.seed)
    done = done_keys()
    jobs = [(m, a, p) for m in models for a in arms for p in probes
            if (m, a, p["id"]) not in done]
    print(f"{len(probes)} probes x {len(models)} models x {len(arms)} arms "
          f"= {len(probes) * len(models) * len(arms)} runs; {len(done)} already done, "
          f"{len(jobs)} to go")
    if not jobs:
        return

    clients = {m: make_client({
        "provider": "fireworks",
        "model_id": f"accounts/fireworks/models/{m}",
        "api_key_env": "FIREWORKS_API_KEY",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout,
    }) for m in models}

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    t0, n_done = time.time(), 0

    async def one(model, arm, probe):
        nonlocal n_done
        async with sem:
            try:
                rec = await replay_one(clients[model], probe, arm, args.max_tokens,
                                       save_images=args.save_images)
            except Exception as e:  # log-and-continue: a dead call is retryable later
                print(f"  ERROR {model} {arm} {probe['id']}: {e!r}")
                return
        rec["model"] = model
        async with lock:
            with open(results_path(arm), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 25 == 0 or n_done == len(jobs):
                print(f"  {n_done}/{len(jobs)} ({time.time() - t0:.0f}s)")

    await asyncio.gather(*[one(m, a, p) for m, a, p in jobs])


# --- analyze ----------------------------------------------------------------
def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial) on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def analyze(args):
    probes = {p["id"]: p for p in (json.loads(l) for l in open(PROBES))}
    rows = [json.loads(l) for path in (RESULTS, IMG_RESULTS)
            if os.path.exists(path) for l in open(path)]
    # keep the last record per key (re-runs append)
    latest = {}
    for r in rows:
        latest[(r["model"], r["arm"], r["id"])] = r
    rows = list(latest.values())

    models = sorted({r["model"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    axes = ["horizontal", "vertical", "diag_dr", "diag_dl"]

    print("miss = neither blocked nor took own immediate win; "
          "frame-slips shown separately\n")
    for m in models:
        print(f"== {m}")
        print(f"   {'arm':10s} " + "".join(f"{ax:>24s}" for ax in axes))
        for arm in arms:
            cells = []
            for ax in axes:
                rs = [r for r in rows if r["model"] == m and r["arm"] == arm
                      and r["axis"] == ax]
                if not rs:
                    cells.append("—")
                    continue
                miss = sum(r["missed"] for r in rs)
                slip = sum(r["frame_slip"] for r in rs)
                extra = f" +{slip}slip" if slip else ""
                cells.append(f"{miss}/{len(rs)}={miss / len(rs):.1%}{extra}")
            print(f"   {arm:10s} " + "".join(f"{c:>24s}" for c in cells))
        # paired McNemar per axis between each pair of arms
        from itertools import combinations
        for a0, a1 in combinations(arms, 2):
            for ax in axes:
                pair = defaultdict(dict)
                for r in rows:
                    if r["model"] == m and r["axis"] == ax and r["arm"] in (a0, a1):
                        pair[r["id"]][r["arm"]] = not r["missed"]
                both = {k: v for k, v in pair.items() if len(v) == 2}
                if not both:
                    continue
                b = sum(1 for v in both.values() if v[a0] and not v[a1])
                c = sum(1 for v in both.values() if not v[a0] and v[a1])
                print(f"   McNemar {ax:10s}: n={len(both)}  "
                      f"{a0}-ok/{a1}-miss={b}  {a0}-miss/{a1}-ok={c}  "
                      f"p={mcnemar_p(b, c):.3f}")
        print()

    # headline: the ↘ vs ↙ gap per arm, pooled over models
    print("== pooled ↘ vs ↙ (physical axes)")
    for arm in arms:
        line = [f"{arm:10s}"]
        for ax in ["diag_dr", "diag_dl"]:
            rs = [r for r in rows if r["arm"] == arm and r["axis"] == ax]
            if rs:
                miss = sum(r["missed"] for r in rs)
                line.append(f"{ax}: {miss}/{len(rs)} = {miss / len(rs):5.1%}")
        print("   " + "   ".join(line))
    print("\nE1 (perceptual layout) predicts flip_rows keeps physical diag_dl harder;")
    print("E2 (label arithmetic) predicts flip_rows makes physical diag_dr the harder one.")


# --- transcribe: vision-fidelity control for the image arm -------------------
async def transcribe(args):
    """Can the model even READ the board image? Sample boards, ask for a full
    stone listing, score cell-level accuracy against ground truth. Separates
    'vision can't parse the board' from 'vision parses fine, axis effect
    remains/disappears'."""
    probes = load_probes(["diag_dr", "diag_dl"], 0, 0)
    rng = random.Random(args.seed)
    sample = rng.sample(sorted(probes, key=lambda p: p["id"]), args.n)
    client = make_client({
        "provider": "fireworks",
        "model_id": f"accounts/fireworks/models/{args.models}",
        "api_key_env": "FIREWORKS_API_KEY",
        "temperature": 0.0, "max_tokens": args.max_tokens, "timeout_s": args.timeout,
    })
    ask = (f"{_IMG_POINTER}\n\nList the coordinates of every X stone and every O "
           "stone on the board. Answer with exactly two lines:\nX: <coords>\nO: <coords>\n"
           "using coordinates like E5, comma-separated, or 'none'.")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(p):
        async with sem:
            out = await client.generate(image_messages(ask, render_board_png(p["board"])))
        got = {"X": set(), "O": set()}
        for ln in (out.content or "").splitlines():
            s = ln.strip().upper()
            if s.startswith(("X:", "O:")):
                got[s[0]] = {m.group(1) + m.group(2) for m in _COORD.finditer(s[2:])}
        truth = {"X": set(), "O": set()}
        for r in range(9):
            for c in range(9):
                if p["board"][r][c]:
                    truth[p["board"][r][c]].add(f"{_COLS[c]}{r + 1}")
        stones = truth["X"] | truth["O"]
        correct = (got["X"] & truth["X"]) | (got["O"] & truth["O"])
        extra = (got["X"] | got["O"]) - stones
        return len(correct), len(stones), len(extra)

    res = await asyncio.gather(*[one(p) for p in sample])
    tc, ts, te = map(sum, zip(*res))
    perfect = sum(1 for c, s, e in res if c == s and e == 0)
    print(f"{args.models} board transcription over {args.n} boards "
          f"({ts} stones total):")
    print(f"  stone recall {tc}/{ts} = {tc / ts:.1%}   hallucinated cells: {te}")
    print(f"  boards transcribed perfectly: {perfect}/{args.n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--models", required=True)
    rp.add_argument("--arms", default="base,flip_rows")
    rp.add_argument("--axes", default="diag_dr,diag_dl")
    rp.add_argument("--limit", type=int, default=0, help="max probes per axis (0 = all)")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--concurrency", type=int, default=16)
    rp.add_argument("--temperature", type=float, default=0.6)   # tournament setting
    rp.add_argument("--max-tokens", type=int, default=131072)   # tournament setting
    rp.add_argument("--timeout", type=float, default=300)
    rp.add_argument("--save-images", action="store_true",
                    help="persist the PNG sent for each image-arm probe to "
                         f"{IMG_DIR}/ (default: off, images are built in-memory)")
    tp = sub.add_parser("transcribe")
    tp.add_argument("--models", required=True, help="one vision model name")
    tp.add_argument("--n", type=int, default=30)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--concurrency", type=int, default=8)
    tp.add_argument("--max-tokens", type=int, default=16384)
    tp.add_argument("--timeout", type=float, default=300)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "transcribe":
        asyncio.run(transcribe(args))
    else:
        analyze(args)


if __name__ == "__main__":
    main()
