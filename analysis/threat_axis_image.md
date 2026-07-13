# Experiment: board-as-image input vs the ↘/↙ diagonal miss-rate gap (Gomoku)

**Question.** The board reports blame diagonal misses on flattening the 2-D board into
1-D row-major text. If so, presenting the same position as a true 2-D **image** should
weaken the axis dependence, in particular the ↙ vs ↘ gap.

**Design.** Same probe set as the flip-rows experiment: the 515 Gomoku decisions where
the acting model faced exactly one blockable immediate-loss diagonal threat (284 ↘,
231 ↙), replayed as single-move queries. Two arms, same positions, same sampling
settings as the tournament (temperature 0.6, max_tokens 131072, Fireworks):

- `base` — the verbatim text prompt stored in the run logs;
- `image` — identical prompt text, except the ASCII board block is replaced by one
  pointer sentence and the board is attached as a PNG: a 9×9 grid with `X`/`O` glyphs
  drawn in the cells, column letters A–I on top, row numbers 1–9 on the left (same
  symbols as the text rendering, laid out in true 2-D). Rendered by
  `render_board_png()` in `scripts/threat_probe_replay.py`; samples under
  `runs/threat_probes/images/`.

Models: **kimi-k2p6** and **qwen3p7-plus** — the only models on this Fireworks account
that genuinely accept image input (deepseek-v4-pro, minimax-m2p7, gpt-oss-120b,
glm-5p1/5p2 reject multimodal requests; minimax-m3 accepts the request but silently
drops the image, so it is unusable). Answers are parsed and scored exactly as in the text arms; misses = neither
blocked nor took an available own win.

**Transcription control.** Before interpreting miss rates, a pure perception check:
on 30 sampled boards (462 stones), each model was asked to list every X and O
coordinate from the image alone.

- kimi-k2p6: stone recall 445/462 = **96.3%**, hallucinated cells **0**, perfect boards **29/30**
- qwen3p7-plus: stone recall 461/462 = **99.8%**, hallucinated cells **1**, perfect boards **29/30**

So the image channel reads these boards essentially losslessly for both models;
image-arm misses reflect reasoning over the 2-D input, not vision noise.

**Results** (miss rate, n=284 ↘ / 231 ↙ per arm; image arms: 0 unparsed, 0 truncated):

kimi-k2p6:

| arm             | ↘ diag_dr   | ↙ diag_dl   |
|-----------------|-------------|-------------|
| `base` (text)   | 1/284 = 0.4% | 8/231 = 3.5% |
| `image` (PNG)   | 2/284 = 0.7% | 2/231 = 0.9% |

qwen3p7-plus:

| arm             | ↘ diag_dr    | ↙ diag_dl     |
|-----------------|--------------|---------------|
| `base` (text)   | 22/284 = 7.7% | 26/231 = 11.3% |
| `image` (PNG)   | 22/284 = 7.7% | 35/231 = 15.2% |

Factual notes, no more: for kimi, the text arm's misses concentrate on ↙ (8 vs 1) and
the concentration is gone in the image arm (2 vs 2) on the same positions — but counts
are small (near-floor model). For qwen, the text-arm ↙ concentration (26 vs 22 at
unequal n) persists in the image arm (35 vs 22); image input did not remove it. The
two models therefore do not agree. Cross-experiment comparison (vs flip_rows etc.) is
deferred until all independent results are collected.

## Reproduce

```bash
# probes (shared with the flip-rows experiment)
python3 scripts/threat_probes_extract.py gomoku

# vision-fidelity control (per model)
python3 scripts/threat_probe_replay.py transcribe --models kimi-k2p6 --n 30
python3 scripts/threat_probe_replay.py transcribe --models qwen3p7-plus --n 30

# image arm (resumable; add --save-images to persist the PNGs sent).
# kimi already has text-base results; qwen needs both arms:
python3 scripts/threat_probe_replay.py run \
    --models kimi-k2p6 --arms image --axes diag_dr,diag_dl --concurrency 32
python3 scripts/threat_probe_replay.py run \
    --models qwen3p7-plus --arms base,image --axes diag_dr,diag_dl --concurrency 16
#   -> runs/threat_probes/images/replays_gomoku_image.jsonl

# result tables
python3 scripts/threat_probe_replay.py analyze
```
