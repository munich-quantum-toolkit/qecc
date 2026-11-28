# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""In-process memo for pure, repeated computations of the search: one bounded dictionary per kind and process."""

from __future__ import annotations

import hashlib
import pickle  # ruff: ignore[suspicious-pickle-import]  (only hashes objects, never loads)

MAX_ENTRIES = 100_000  # per kind, unless ``put`` gives a smaller limit; the oldest entries go first
_MEMO: dict[str, dict[str, object]] = {}


def digest(*items: object) -> str:
    """Hex key of ``items`` (bytes as they are, everything else pickled)."""
    h = hashlib.sha1(usedforsecurity=False)
    for p in items:
        h.update(p if isinstance(p, bytes) else pickle.dumps(p, protocol=4))
    return h.hexdigest()


def get(kind: str, key: str) -> object | None:
    """The stored value, or None."""
    return _MEMO.get(kind, {}).get(key)


def put(kind: str, key: str, value: object, limit: int = MAX_ENTRIES) -> None:
    """Store ``value``; drops the oldest entry of ``kind`` beyond ``limit`` entries."""
    table = _MEMO.setdefault(kind, {})
    table[key] = value
    if len(table) > limit:
        del table[next(iter(table))]


def clear() -> None:
    """Forget everything."""
    _MEMO.clear()
