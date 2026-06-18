"""Fold the Claude (opus-4.8 / sonnet-4.6) matchups from the separate
gpt_claude_fireworks_coached_tournament log into the main runs/ tree, so the
existing aggregators/analyzers rate Claude as two new models against the field.

Claude played the 4 shared games (connect4, gomoku, holdem_1hand, holdem_match)
against 5 models that are also in runs/ (deepseek-v4-pro, gpt-oss-120b,
kimi-k2p6, glm-5p1, minimax-m2p7) plus each other. We symlink each Claude
matchup dir into runs/<game>/ under that game's naming convention:

  - claude models -> bare name (new identity; strip_coached leaves it alone)
  - fireworks opponents -> "<name>-coached" so they merge with the existing
    entry (runs/ uses the -coached raw naming; analysis strips it back)
  - connect4/gomoku: "<game>__A__vs__B"; holdem_1hand: "A__vs__B__r0";
    holdem_match: "A__vs__B"

Symlinks (not copies) keep it space-cheap and trivially reversible:
    python3 scripts/merge_claude_into_runs.py          # add links
    python3 scripts/merge_claude_into_runs.py --remove # remove them
"""

from __future__ import annotations

import glob
import os
import sys

# The Claude-vs-field tournament log, cloned into the repo (gitignored). Default
# is repo-relative; override with AIBATTLE_CLAUDE_SRC. abspath() so the symlinks
# we create under runs/<game>/ resolve regardless of where they live.
SRC = os.path.abspath(os.environ.get(
    "AIBATTLE_CLAUDE_SRC",
    "aibattle-logs/gpt_claude_fireworks_coached_tournament"))
CLAUDE = {"claude-opus-4.8", "claude-sonnet-4.6"}
# game -> (dest subdir, name builder given ordered a,b raw model names)
GAMES = {
    "connect4":     lambda a, b: f"connect4__{a}__vs__{b}",
    "gomoku":       lambda a, b: f"gomoku__{a}__vs__{b}",
    "holdem_1hand": lambda a, b: f"{a}__vs__{b}__r0",
    "holdem_match": lambda a, b: f"{a}__vs__{b}",
}


def _coached(name: str) -> str:
    """Fireworks opponents carry the -coached suffix in runs/; Claude stays bare."""
    return name if name in CLAUDE else f"{name}-coached"


def _links():
    """Yield (src_dir, dest_path) for every Claude matchup in every shared game."""
    for game, namer in GAMES.items():
        for src in sorted(glob.glob(os.path.join(SRC, game, "*__vs__*"))):
            base = os.path.basename(src)
            a, b = base.split("__vs__")
            if not (a in CLAUDE or b in CLAUDE):
                continue
            dest = os.path.join("runs", game, namer(_coached(a), _coached(b)))
            yield src, dest


def main(remove=False):
    n = 0
    for src, dest in _links():
        if remove:
            if os.path.islink(dest):
                os.unlink(dest); n += 1
            continue
        if os.path.lexists(dest):
            continue
        os.symlink(src, dest)
        n += 1
        print(f"link {os.path.basename(dest)} -> {src.split('/')[-2]}/{os.path.basename(src)}")
    print(f"\n{'removed' if remove else 'linked'} {n} matchup dirs")


if __name__ == "__main__":
    main(remove="--remove" in sys.argv)
