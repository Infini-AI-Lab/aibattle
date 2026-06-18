"""Convert existing tournament output into the unified arena raw-data format.

This is both a working adapter for the historical *coached* pool data and a
reference example of the interface documented in ``aibattle.eval.arena``. You
hand the onboarding script a directory of unified per-game JSON files; this
script produces them from a ``runs/<exp>/`` tree.

  * versus games (round_robin_seat_swap): each stored episode -> one unified
    versus episode {"scores": {modelA: ret, modelB: ret}}.
  * environment games (independent_vs_dealer): each per-episode ``ep*.json`` ->
    one unified environment episode {"model": m, "score": player_0 return}.
    The dealer/opponent seat is dropped (it is never ranked).

Usage:
  PYTHONPATH=src python scripts/convert_to_unified.py \
      runs/new_games_5models_coached  pool_unified
"""

from __future__ import annotations

import glob
import json
import os
import sys

ENV_STRUCTURES = {"independent_vs_dealer", "model_vs_baseline"}
ENV_GAMES = {"independent_blackjack"}
# Games rated by chip-weighted Elo (magnitude matters), matching the analyzers.
CHIP_BASIS_GAMES = {"leduc_poker", "kuhn_poker", "holdem_1hand", "holdem_match"}


def convert_game(game_dir: str) -> dict | None:
    data_path = os.path.join(game_dir, "data.json")
    if not os.path.exists(data_path):
        return None
    data = json.load(open(data_path, encoding="utf-8"))
    game = data.get("game") or os.path.basename(game_dir.rstrip("/"))
    structure = data.get("structure", "")
    models = data.get("models", [])

    is_env = (structure in ENV_STRUCTURES or game in ENV_GAMES
              or structure.startswith("model_vs_") or structure.startswith("independent"))

    episodes: list = []
    if is_env:
        # Read per-episode files; player_0 is the ranked model, player_1 the env.
        for ep_path in sorted(glob.glob(os.path.join(game_dir, "*", "ep*.json"))):
            ep = json.load(open(ep_path, encoding="utf-8"))
            seat = ep.get("seat_assignment", {})
            ret = ep.get("returns", {})
            model = seat.get("player_0")
            if model is None or "player_0" not in ret:
                continue
            episodes.append({"model": model, "score": float(ret["player_0"]),
                             "seed": ep.get("seed")})
        kind = "environment"
    else:
        for pair in data.get("pairs", []):
            for ep in pair.get("episodes", []):
                seat = ep.get("seat_assignment", {})
                ret = ep.get("returns", {})
                scores = {seat[p]: float(ret[p]) for p in ("player_0", "player_1")
                          if p in seat and p in ret}
                if len(scores) >= 2:
                    episodes.append({"scores": scores, "seed": ep.get("seed")})
        kind = "versus"

    out = {"game": game, "kind": kind, "models": models, "episodes": episodes}
    if kind == "versus" and game in CHIP_BASIS_GAMES:
        out["elo_basis"] = "chips"
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for game_dir in sorted(glob.glob(os.path.join(src, "*"))):
        if not os.path.isdir(game_dir):
            continue
        unified = convert_game(game_dir)
        if unified is None:
            continue
        out = os.path.join(dst, f"{unified['game']}.json")
        json.dump(unified, open(out, "w", encoding="utf-8"))
        print(f"{unified['game']}: kind={unified['kind']} "
              f"models={len(unified['models'])} episodes={len(unified['episodes'])} "
              f"-> {out}")


if __name__ == "__main__":
    main()
