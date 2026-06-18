"""Shared terminal-theme CSS for the generated report pages.

The site's design system (see .claude/skills/terminal-ui) is a light "shell
session printed on paper" look: paper-white background, dark-red accent,
monospace everywhere. It was originally hand-applied to the committed report
HTML but never to the generator scripts, so every regeneration reverted the
pages to the old dark theme. This module is the single source of truth: every
analyze_*.py embeds BASE_CSS (and CHART_SETUP for Chart.js pages) so regenerated
reports keep the theme. Tokens mirror reports/index.html (the canonical page).

BASE_CSS is a plain string (not an f-string), so its literal `{`/`}` are safe to
drop into a generator's f-string template via `{BASE_CSS}`.
"""

# Canonical design tokens + the components shared across every report page.
BASE_CSS = """
  /* Terminal theme: paper-white background, dark-red accent, monospace. */
  :root { --bg:#fbfbf8; --fg:#1c1c1c; --red:#8f1d1d; --dim:#6b6b6b;
    --line:#ddd8cf; --panel:#ffffff; --faint:#f4f1ea;
    --pos:#1a7f37; --neg:#b91c1c; --diag:#c9c2b6; }
  body { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    margin:0; background:var(--bg); color:var(--fg); }
  .wrap { max-width:1100px; margin:0 auto; padding:36px 24px 80px; }
  a { color:var(--red); }
  h1 { font-size:22px; margin:0 0 4px; color:var(--red); font-weight:700; }
  h1 .cursor { display:inline-block; width:11px; height:1em; background:var(--red);
    vertical-align:text-bottom; margin-left:5px; animation:blink 1.1s steps(1) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  h1 + .sub { margin-bottom:30px; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:30px; }
  h2 { font-size:15px; margin:36px 0 10px; font-weight:700; }
  h2::before { content:"## "; color:var(--red); }
  h3 { font-size:14px; color:var(--fg); margin:0 0 6px; }
  h3::before { content:"### "; color:var(--red); }
  .note { color:var(--dim); font-size:12px; margin:6px 0 14px; }
  .small { font-size:10px; color:var(--dim); font-weight:400; }

  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { padding:7px 10px; text-align:center; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:600; }
  td.model, th.model { text-align:left; font-weight:700; color:var(--fg); }
  .pos { color:var(--pos); } .neg { color:var(--neg); } .diag { color:var(--diag); }
  .tag { background:var(--faint); color:var(--red); border:1px solid var(--line);
    border-radius:0; padding:1px 8px; font-size:11px; margin-left:3px;
    white-space:nowrap; }

  .kpis { display:flex; gap:14px; flex-wrap:wrap; margin:16px 0; }
  .kpi { background:var(--faint); border:1px solid var(--line); padding:12px 16px; }
  .kpi .v { font-size:22px; font-weight:700; color:var(--fg); }
  .kpi .l { font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.04em; }

  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:10px; }
  .cards { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:0; padding:16px; }
  canvas { max-height:340px; }
  .callout { background:var(--faint); border:1px solid var(--line); border-left:3px solid var(--red);
    padding:12px 14px; margin:14px 0; font-size:13px; }
  /* Replay button: the one place a non-token link color is used (indigo),
     per the terminal-ui SKILL. Sits directly under the subtitle. */
  .replaybtn { display:inline-block; margin-top:12px; background:var(--faint); color:#4338ca;
    border:1px solid var(--line); border-radius:0; padding:8px 14px; font-size:13px; text-decoration:none; }
  .replaybtn:hover { border-color:var(--red); color:var(--fg); }
  @media (max-width:760px) { .grid2, .cards { grid-template-columns:1fr; } }
"""

# Drop-in <script> prelude: makes Chart.js readable on the light background
# (dark labels, faint gridlines) and exposes a red-accent palette for datasets.
# Use instead of the old dark `Chart.defaults.color='#9aa3b5'` lines.
CHART_SETUP = """
Chart.defaults.color='#1c1c1c'; Chart.defaults.borderColor='#e7e2d8';
Chart.defaults.font.family='ui-monospace,SFMono-Regular,Menlo,Consolas,monospace';
const ACCENT='#8f1d1d';
const PALETTE=['#8f1d1d','#b45309','#1a7f37','#1d4ed8','#6d28d9','#be185d','#0f766e'];
"""
