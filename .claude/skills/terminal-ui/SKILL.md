---
name: terminal-ui
description: Terminal-style design system for AI Battle Arena report/replay pages. Use when creating or editing any HTML page, report, replay viewer, or UI under reports/ in this project — covers design tokens, type scale, spacing rhythm, component patterns, nav integration, and cache-busting conventions.
---

# Terminal-style UI design system

The whole site reads like a shell session printed on paper: **white paper background, dark-red
accent, monospace everywhere, square corners, no shadows**. Headings are shell prompts/comments,
metadata sits in `[brackets]`, links are lowercase commands.

Canonical sources (always match these, never invent new values):
- `reports/index.html` — canonical tokens, type scale, section/card/table patterns
- `reports/nav.css` + `reports/nav.js` — fixed left sidebar, injected into every page
- `reports/replay.css` — shared stylesheet for the five replay viewers (extends the tokens)

## Design tokens

Define on `:root`, use the same names everywhere:

```css
:root { --bg:#fbfbf8; --fg:#1c1c1c; --red:#8f1d1d; --dim:#6b6b6b;
        --line:#ddd8cf; --panel:#ffffff; --faint:#f4f1ea; }
```

| token | value | role |
|---|---|---|
| `--bg` | `#fbfbf8` | page background (paper) |
| `--fg` | `#1c1c1c` | body text |
| `--red` | `#8f1d1d` | THE accent: headings, prefixes, active states, highlight borders |
| `--dim` | `#6b6b6b` | secondary text, labels, disabled |
| `--line` | `#ddd8cf` | all 1px borders |
| `--panel` | `#ffffff` | surface of section boxes / cards / panels |
| `--faint` | `#f4f1ea` | inset accents only (code/`<pre>` blocks, chips) — NOT panels |

Extended tokens (replay.css) when semantics are needed beyond the accent:
`--green:#1a7f37` (win/positive), `--amber:#b45309` (warning/dealer/gold), `--link:#2563eb`
(last-move markers), `--neg:#b91c1c` (negative numbers — distinct from `--red`, which is the
accent, so losses don't look like UI chrome).

Font stack (body, everything): `ui-monospace,SFMono-Regular,Menlo,Consolas,monospace`.

## Type scale

| element | spec |
|---|---|
| `h1` | 22px bold `--red`, `margin:0 0 4px`; text is a fake prompt `$ ~/aibattle/<area>/<page>` ending in `<span class="cursor"></span>` (blinking 11px red block) |
| `.sub` | 13px `--dim`, `margin-bottom:30px` — one-line page intro directly under h1 |
| `h2` | 15px bold, `margin:0`, with `h2::before{content:"## "; color:var(--red);}` |
| body text | 13px |
| `.note` | 12px `--dim`, `margin:6px 0 14px` — explanatory paragraph under a section head |
| `.arena-tag` | 11px `--dim`, wrapped in red `[` `]` via `::before/::after` |
| `.badge` | 10px `--dim`, wrapped in `[` `]` |

## Spacing rhythm

- `.wrap { max-width:1100px; margin:0 auto; padding:36px 24px; }` on every page.
- **box-sizing gotcha:** pages are content-box (1100px content + 24px padding = 1148px column).
  If a stylesheet adds `*{box-sizing:border-box}` for its widgets, it MUST also set
  `.wrap{box-sizing:content-box}` or the whole text column shifts 24px right vs. other pages.
- 14px gaps: card grids, selector rows, and vertical gaps between adjacent components.
- 34px before each major section (`.arena { margin-top:34px; }`).
- Section boxes: `border:1px solid var(--line); background:var(--panel); padding:20px 20px 24px;`.
- No `border-radius` anywhere except circles (avatars, discs, stones) and the poker felt.

## Component patterns

Section head — every titled block uses this exact markup, including side panels:
```html
<div class="arena-head"><h2>section title</h2><span class="arena-tag">meta · hint</span></div>
```

Group label inside a section: 13px bold, `padding-left:10px; border-left:3px solid var(--red);`
with an 11px `--dim` sub-label span.

Cards (`.cards` 2-col grid, 1-col under 640px): white panel, `--line` border,
`:hover{border-color:var(--red)}`, title 15px bold red with `::before{content:"> "; color:var(--dim)}`,
footer link 12px red ("view analysis →").

Tables (`table.lb`): 13px, `border-collapse:collapse`, 7px 10px cells, bottom borders only
(`--line`), `th` 600-weight `--dim`, centered except left-aligned bold model column, medal emoji
(🥇🥈🥉) then plain numbers in the rank column. Inline bars: absolutely-positioned `--red` div at
`opacity:.16` behind a bold red value (see `.scorecell`).

Buttons (replay.css `.btn`): white, 1px `--line` border, 13px, square; hover → red border + red
text; `.primary` → filled `--red` white text. Active/selected states use a red border +
`box-shadow:0 0 0 1px var(--red)` ring, never a fill change.

Replay button (`.replaybtn`) — every report page that has a replay viewer links to it with this
ONE canonical control, placed directly under the `.sub` line (not in a section, not at the bottom):
```html
<a class='replaybtn' href='<game>_replay.html'>▶ watch <game|hand|table> replays</a>
```
```css
.replaybtn { display:inline-block; margin-top:12px; background:var(--faint); color:#4338ca;
  border:1px solid var(--line); padding:8px 14px; font-size:13px; text-decoration:none; }
.replaybtn:hover { border-color:var(--red); color:var(--fg); }
```
Note the label is **indigo `#4338ca`** (the one place a non-token link color is used) on a `--faint`
inset, square, with the `▶` glyph. Verb matches the unit replayed: board games → "game", poker →
"hand", ring poker → "table". Keep it short — no trailing description.

Status colors: winner/positive `--green`, negative numbers `--neg`, attention/dealer `--amber`,
acting/current `--red`. Emoji serve as icons (🏆 winner, 🎲 brand, per-game emoji in titles).

## Page skeleton

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Battle Arena — <emoji> <Page></title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'><emoji></text></svg>">
<link rel="stylesheet" href="nav.css?v=4"><script defer src="nav.js?v=9"></script>
<style> /* :root tokens + page styles, or <link rel="stylesheet" href="replay.css?v=N"> */ </style>
</head>
<body>
<div class="wrap">
  <h1>$ ~/aibattle/<area>/<page><span class="cursor"></span></h1>
  <div class="sub"><emoji> <Name> · one-line description with <b>key terms</b> bolded.</div>
  <!-- sections -->
</div>
</body></html>
```

## Nav + caching logistics

- `nav.css` offsets content with `body{padding-left:190px}` — page styles must set `margin:0`,
  never `margin-left`, or they'd clobber nothing / fight the sidebar offset.
- New pages must be registered in `nav.js`: add a link line AND an `ACTIVE` map entry. Replay
  pages are not listed as links; map them to their parent report so that nav item stays lit
  (e.g. `"connect4_replay.html": "connect4_report.html"`).
- Cache-busting (the dev server sends no Cache-Control): every stylesheet/script href carries
  `?v=N` — bump it whenever that file changes. After restyling pages, also bump `var V` in
  `nav.js` so sidebar links fetch fresh HTML. Tell the user to hard-refresh (Ctrl+Shift+R) the
  page they're currently on.

## New/edited page checklist

1. Tokens, type scale, and spacing taken from this spec — no new colors, sizes, or fonts.
2. `.wrap` column is 1148px total (watch the box-sizing gotcha).
3. Section heads use `arena-head` + `## `-prefixed h2 + bracketed tag.
4. nav.css/nav.js linked; nav.js links + ACTIVE map updated for new pages.
5. `?v=` bumped on every file you changed; `V` bumped in nav.js if pages were restyled.
6. Verify with the local server (`python3 -m http.server 8000` from `reports/`), pages return 200.
