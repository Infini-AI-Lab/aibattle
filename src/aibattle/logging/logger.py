"""JSONL match logger.

One file per match. Records share a ``record_type`` discriminator:
  - "match"   : header (first line) — game, agents, resolved config, seed
  - "step"    : one decision point
  - "episode" : per-episode summary

JSONL is chosen so logs stream as produced, are greppable, and convert to
trajectory formats later without a schema migration.
"""

from __future__ import annotations

import json
import os
from typing import Optional, TextIO

from ..types import StepRecord


def serialize_step(rec: StepRecord) -> dict:
    """Serialize a StepRecord to a plain dict (shared by log + trajectory files)."""
    return {
        "step": rec.step_index,
        "player": rec.player,
        "observation": rec.observation.to_dict(),
        "response": rec.response.to_dict(),
        "selected_action": rec.selected_action,
        "selected_amount": rec.selected_amount,
        "invalid": rec.invalid_info.invalid,
        "invalid_info": rec.invalid_info.to_dict(),
    }


class MatchLogger:
    def __init__(self, path: Optional[str]):
        """If ``path`` is None, logging is a no-op (summary-only runs)."""
        self.path = path
        self._fh: Optional[TextIO] = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def _write(self, record: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def match_header(self, header: dict) -> None:
        self._write({"record_type": "match", **header})

    def step(self, episode: int, pair_id: int, rec: StepRecord) -> None:
        self._write({
            "record_type": "step",
            "episode": episode,
            "pair_id": pair_id,
            **serialize_step(rec),
        })

    def episode_end(self, record: dict) -> None:
        self._write({"record_type": "episode", **record})

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
