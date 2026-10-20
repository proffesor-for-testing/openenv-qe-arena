"""Strict, bounded readers for model output. Output is data: parsed, never evaluated."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

MAX_DEPTH = 32


def _depth(value: Any, level: int = 0) -> int:
    if level > MAX_DEPTH:
        raise ValueError("output nests too deeply")
    if isinstance(value, dict):
        return max([_depth(v, level + 1) for v in value.values()], default=level)
    if isinstance(value, list):
        return max([_depth(v, level + 1) for v in value], default=level)
    return level


def load_object(text: str) -> dict:
    """The output file as one JSON object (no NaN or Infinity), or ValueError."""
    def reject(token: str) -> None:
        raise ValueError(f"non-finite number {token}")

    value = json.loads(text, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("output must be a JSON object")
    _depth(value)
    return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(value: Any, reference: float, tolerance: float) -> bool:
    """A finite number within ``tolerance`` (absolute) of the reference."""
    return is_number(value) and abs(value - reference) <= tolerance


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def write_csv(header: list[str], records: list[list[Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(records)
    return out.getvalue()
