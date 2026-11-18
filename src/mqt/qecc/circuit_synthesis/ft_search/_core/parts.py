# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parts: subtrees of designs (ancilla states with their checks) that are fault tolerant for their own target.

Every exact evaluation adds the parts it finds; block moves and completions install the cheapest known part of an
ancilla state instead of protecting it again. Each process keeps its own parts; the main process collects those a job
found (:func:`take_new`) and sends its collection with every new job (:func:`snapshot`, :func:`merge`).
"""

from __future__ import annotations

import copy
import hashlib
from typing import TYPE_CHECKING, Any

from .circuit import metrics
from .design import Design, Node, compile_node, scope_name
from .front import pareto
from .memo import digest
from .solvers import part_cost, scope_slack
from .verify import target_key

if TYPE_CHECKING:
    from .config import Context

_PARTS: dict[str, dict[str, dict[str, Any]]] = {}  # state key -> design key -> {"key", "node", "metrics"}
_NEW: list[tuple[str, dict[str, Any]]] = []  # parts this process found since the last take_new


def state_key(node: Node, t: int) -> str:
    """Key of what a part prepares: its state, its local target and t."""
    return digest("state", *target_key(node.state), *target_key(node.target), t)[:24]


def design_key(node: Node) -> str:
    """Key of a subtree: its compiled circuit."""
    return hashlib.sha1(str(compile_node(node).to_stim()).encode(), usedforsecurity=False).hexdigest()


def add(node: Node, t: int) -> None:
    """Record ``node`` (fault tolerant for its own target) as a part of its state."""
    skey, key = state_key(node, t), design_key(node)
    entries = _PARTS.setdefault(skey, {})
    if key not in entries:
        entry = {"key": key, "node": copy.deepcopy(node), "metrics": metrics(compile_node(node))}
        entries[key] = entry
        _NEW.append((skey, entry))


def known(node: Node, t: int) -> list[dict[str, Any]]:
    """The known parts of the state of ``node``, without ``node`` itself."""
    cur = design_key(node)
    return [e for k, e in _PARTS.get(state_key(node, t), {}).items() if k != cur]


def cheapest(parts: list[dict[str, Any]], ctx: Context, lam: float, slack: int | None) -> dict[str, Any]:
    """The part of the lowest cost under depth weight ``lam``; depth counts only beyond ``slack``."""
    return min(parts, key=lambda e: part_cost(ctx, e["metrics"], lam, slack))


def install(design: Design, path: tuple[int, ...], entry: dict[str, Any]) -> Design:
    """A copy of ``design`` with the ancilla scope at ``path`` replaced by the part ``entry``."""
    m = entry["metrics"]
    return design.with_node(path, copy.deepcopy(entry["node"]), f"{scope_name(path)}:parts[{m['ops']}/{m['depth']}]")


def with_known_parts(parent: Design, props: list, ctx: Context) -> list[tuple[str, Design]]:
    """Variants of the proposals with each new ancilla scope replaced by a known part.

    Per scope the cheapest part under the run's smallest, main and largest depth weight, at most two distinct.
    """
    old = {p for p, _ in parent.nodes()}
    out = []
    for label, d in props:
        for p, node in d.nodes():
            if not p or p in old or node.opaque or p[:-1] not in old:
                continue
            parts = known(node, d.t)
            if not parts:
                continue
            chosen: dict[str, dict[str, Any]] = {}
            slack = scope_slack(d, p)
            for lam in ctx.extreme_lams():
                if len(chosen) < 2:
                    e = cheapest(parts, ctx, lam, slack)
                    chosen.setdefault(e["key"], e)
            out += [(f"{label}+parts", install(d, p, e)) for e in chosen.values()]
    return out


def take_new() -> list[tuple[str, dict[str, Any]]]:
    """The parts this process found since the last call (returned with a job's result)."""
    new = list(_NEW)
    _NEW.clear()
    return new


def merge(entries: list[tuple[str, dict[str, Any]]]) -> None:
    """Add parts found elsewhere (another worker's result, or the main process's snapshot)."""
    for skey, e in entries:
        _PARTS.setdefault(skey, {}).setdefault(e["key"], e)


def snapshot() -> list[tuple[str, dict[str, Any]]]:
    """All known parts not dominated in operations and depth within their state, to send with a job."""
    return [
        (skey, e)
        for skey, entries in _PARTS.items()
        for e in pareto(list(entries.values()), lambda e: (e["metrics"]["ops"], e["metrics"]["depth"]), ties=True)
    ]


def clear() -> None:
    """Forget all parts (a new search)."""
    _PARTS.clear()
    _NEW.clear()
