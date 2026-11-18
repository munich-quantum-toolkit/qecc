# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Ancilla merges as moves, on rectangle partitions of a set of same-type checks (Huang & Brown, arXiv:2012.15403).

An *atom* is (check row c, data qubit q in c). Each ancilla qubit covers a *rectangle* C x D: check rows C, data qubits
D (in coupling order), every check in C containing every qubit in D. A gadget is a partition of all atoms into
rectangles. Cats are single atoms, a bare check is {c} x supp(c), a block is C_q x {q}, Steane is the block on all
stabilizers of a type. Two merges and their reverses reach every partition from the cats: (i) share, two rectangles
with the same D become (C1 u C2) x D; (ii) group, two rectangles with the same C become C x (D1 + D2). The new ancillas
are left unprotected for the completion to repair.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import numpy as np

from ..blocks import BlockSpec
from ..design import ancilla_states, make_node
from ..gauge import Gadget
from .common import CHOICE_CTX_DEPS, block_slack, circuit_choices

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import Context
    from ..design import Design, Node


Rect = tuple[frozenset[int], tuple[int, ...]]  # (check rows, data qubits in coupling order)


def rects_of(spec: BlockSpec) -> list[Rect]:
    """The rectangles of a block (rows index ``spec.checks``)."""
    checks = np.asarray(spec.checks, dtype=np.uint8) % 2
    if not spec.split:
        return [(frozenset(int(r) for r in np.nonzero(checks[:, q])[0]), (q,)) for q in spec.support]
    ht = np.asarray(spec.htilde, dtype=np.uint8) % 2
    return [(frozenset(int(r) for r in np.nonzero(ht[:, a])[0]), tuple(g)) for a, g in enumerate(spec.anc_groups())]


def valid(checks: np.ndarray, rects: list[Rect]) -> bool:
    """Every atom in exactly one rectangle, every rectangle a rectangle (each check of C contains each qubit of D)."""
    seen = set()
    for rows, qs in rects:
        if not rows or not qs:
            return False
        for r in rows:
            for q in qs:
                if not checks[r, q] or (r, q) in seen:
                    return False
                seen.add((r, q))
    return len(seen) == int(np.asarray(checks, dtype=np.int64).sum())


def canonical(rects: list[Rect]) -> tuple:
    """Key of a partition (order of the rectangles does not matter, the coupling order inside one does)."""
    return tuple(sorted((tuple(sorted(r)), d) for r, d in rects))


def components(rects: list[Rect]) -> list[list[Rect]]:
    """Rectangles grouped into independent gadgets (rows connected through shared rectangles)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rows, _ in rects:
        r0 = min(rows)
        for r in rows:
            parent[find(r)] = find(r0)
    out: dict[int, list[Rect]] = {}
    for rc in rects:
        out.setdefault(find(min(rc[0])), []).append(rc)
    return [out[k] for k in sorted(out)]


def realize(kind: str, checks: np.ndarray, rects: list[Rect], pos: int | None) -> list[tuple[str, object]]:
    """The gadgets of a partition: ("bare", (row, support)) for one check on one ancilla qubit, else ("block", spec)."""
    checks = np.asarray(checks, dtype=np.uint8) % 2
    out: list[tuple[str, object]] = []
    for comp in components(rects):
        rows = sorted({r for rs, _ in comp for r in rs})
        if len(rows) == 1 and len(comp) == 1:
            out.append(("bare", (rows[0], list(comp[0][1]))))
            continue
        sub = checks[rows]
        idx = {r: i for i, r in enumerate(rows)}
        plain = all(len(d) == 1 for _, d in comp) and len({d[0] for _, d in comp}) == len(comp)
        if plain and all(frozenset(idx[r] for r in rs) == frozenset(np.nonzero(sub[:, d[0]])[0]) for rs, d in comp):
            out.append(("block", BlockSpec(kind, sub, pos=pos)))
            continue
        ht = np.zeros((len(rows), len(comp)), dtype=np.uint8)
        for a, (rs, _) in enumerate(comp):
            for r in rs:
                ht[idx[r], a] = 1
        out.append(("block", BlockSpec(kind, sub, pos=pos, groups=tuple(d for _, d in comp), htilde=ht)))
    return out


def groups(rects: list[Rect], gmax: int) -> Iterator[tuple[str, list[Rect]]]:
    """(ii): every pair of rectangles for the same checks merged (at most ``gmax`` data qubits on one ancilla qubit)."""
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            (ri, di), (rj, dj) = rects[i], rects[j]
            if ri == rj and len(di) + len(dj) <= gmax:
                rest = [x for k, x in enumerate(rects) if k not in {i, j}]
                yield f"group{i}+{j}", [*rest, (ri, di + dj)]


def share_max(rects: list[Rect]) -> list[Rect]:
    """(i) as far as it goes: one rectangle per set of data qubits."""
    by: dict[frozenset, list[Rect]] = {}
    for rc in rects:
        by.setdefault(frozenset(rc[1]), []).append(rc)
    out = []
    for grp in by.values():
        rows: frozenset[int] = frozenset()
        for rs, _ in grp:
            rows |= rs
        out.append((rows, grp[0][1]))
    return out


def unshare_max(rects: list[Rect]) -> list[Rect]:
    """Reverse of (i) all the way: one rectangle per (check, data set)."""
    return [(frozenset([r]), d) for rows, d in rects for r in sorted(rows)]


def group_to(rects: list[Rect], g: int | None) -> list[Rect]:
    """(ii) per set of checks: its data qubits regrouped into ancilla qubits of ``g`` each (None: all on one)."""
    by: dict[frozenset, list[int]] = {}
    for rows, d in rects:
        by.setdefault(rows, []).extend(d)
    out = []
    for rows, qs in by.items():
        k = len(qs) if g is None else g
        out += [(rows, tuple(qs[i : i + k])) for i in range(0, len(qs), k)]
    return out


def ungroup_max(rects: list[Rect]) -> list[Rect]:
    """Reverse of (ii) all the way: one data qubit per ancilla qubit."""
    return [(rows, (q,)) for rows, d in rects for q in d]


MACROS = {
    "sharemax": share_max,
    "unsharemax": unshare_max,
    "group2": lambda r: group_to(r, 2),
    "group3": lambda r: group_to(r, 3),
    "groupmax": lambda r: group_to(r, None),
    "ungroupmax": ungroup_max,
}


COMBOS = {  # macros as fixed combinations of maximal steps
    "cats": lambda r: unshare_max(ungroup_max(r)),
    "block": lambda r: share_max(ungroup_max(r)),
    "bare": lambda r: group_to(unshare_max(r), None),
    "groupedblock": lambda r: group_to(share_max(ungroup_max(r)), None),
    "groupedcats2": lambda r: group_to(unshare_max(ungroup_max(r)), 2),
}


def _join(us: list[dict]) -> dict:
    """Gadgets of one type at one time point as one unit (its rectangles over the stacked checks)."""
    rects, k = [], 0
    for u in us:
        rects += [(frozenset(r + k for r in rs), d) for rs, d in u["rects"]]
        k += u["checks"].shape[0]
    return {
        "lab": "+".join(u["lab"] for u in us),
        "kind": us[0]["kind"],
        "checks": np.vstack([u["checks"] for u in us]),
        "rects": rects,
        "pos": us[0]["pos"],
        "blocks": [b for u in us for b in u["blocks"]],
        "gadgets": [g for u in us for g in u["gadgets"]],
    }


GMAX = 4  # "group" merges: data qubits per ancilla qubit at most
MAX_PROPS = 16  # proposals per scope at most


class MergeSolver:
    """One kind of merge step on each gadget of the scope (single, all of a type at one time point joined, or pairs).

    ``op`` "group": single steps, at most ``GMAX`` data qubits per ancilla qubit; "macro": maximal steps and
    ``COMBOS``. At most ``MAX_PROPS`` proposals per scope, one per gadget in turn.
    """

    ctx_deps = CHOICE_CTX_DEPS

    def __init__(self, op: str) -> None:
        """Set the merge step ``op`` ("group" or "macro")."""
        assert op in {"group", "macro"}, op
        self.op = op
        self.name = f"blockmerge:{op}"

    @staticmethod
    def _units(node: Node) -> list[dict]:
        end = len(node.base.ops)
        units: list[dict] = []
        for bi, (spec, _) in enumerate(node.blocks):
            units.append({
                "lab": f"b{bi}",
                "kind": spec.kind,
                "checks": np.asarray(spec.checks, dtype=np.uint8) % 2,
                "rects": rects_of(spec),
                "pos": spec.pos,
                "blocks": [bi],
                "gadgets": [],
            })
        n = len(node.base.data)
        for gi, g in enumerate(node.gadgets):
            if g.block is not None:
                continue  # an ancilla qubit of a block: it belongs to that block's unit
            slots = set(g.coupling_slots())
            if len(slots) != 1 or len(set(g.support)) != len(g.support):
                continue  # couplings at several times or a repeated qubit: not a rectangle gadget
            s = next(iter(slots))
            checks = np.zeros((1, n), dtype=np.uint8)
            checks[0, list(g.support)] = 1
            units.append({
                "lab": f"g{gi}",
                "kind": g.kind,
                "checks": checks,
                "rects": [(frozenset([0]), tuple(int(q) for q in g.support))],
                "pos": None if g.late or s >= end else s,
                "blocks": [],
                "gadgets": [gi],
            })
        by_time: dict = {}
        for u in units:
            by_time.setdefault((u["kind"], u["pos"]), []).append(u)
        # all gadgets of one type at one time point joined: the macros reach Steane-like blocks in one move
        joined_all = [_join(us) for us in by_time.values() if len(us) > 2]
        pairs = [_join([a, b]) for us in by_time.values() for i, a in enumerate(us) for b in us[i + 1 :]]
        return units + joined_all + pairs

    def _steps(self, unit: dict) -> Iterator[tuple[str, list]]:
        r = unit["rects"]
        joined = "+" in unit["lab"]
        if self.op == "group" and not joined:
            yield from groups(r, GMAX)
        elif self.op == "macro":
            fs = {**COMBOS} if joined else {**MACROS, **COMBOS}
            for lab, f in fs.items():
                yield lab, f(r)

    @staticmethod
    def _apply(design: Design, node: Node, unit: dict, rects: list, ctx: Context) -> Node | None:
        assert valid(unit["checks"], rects), "merge step broke the partition"
        nd = copy.deepcopy(node)
        nd.gadgets = [g for i, g in enumerate(nd.gadgets) if i not in unit["gadgets"]]
        for b in sorted(unit["blocks"], reverse=True):
            nd.remove_block(b)
        pos = unit["pos"]
        for what, x in realize(unit["kind"], unit["checks"], rects, pos):
            if what == "bare":
                _, sup = cast("tuple[int, list[int]]", x)
                if pos is None:
                    nd.gadgets.append(Gadget(unit["kind"], 0, sup, late=True))
                else:
                    nd.gadgets.append(Gadget(unit["kind"], pos, sup))
                continue
            spec = cast("BlockSpec", x)
            full, target = ancilla_states(spec)
            choices = circuit_choices(spec.ancilla_generator(), target, design.t, ctx, slack=block_slack(nd, spec))
            if not choices:
                return None
            tag, c = choices[0]
            nd.add_block(spec, make_node(f"{nd.name}.m{len(nd.blocks)}", c, full, target, origin=tag))
        return nd

    def propose(self, design: Design, path: tuple[int, ...], ctx: Context) -> Iterator[tuple[str, Node]]:
        """Yield the scope after one merge step on one gadget."""
        node = design.node(path)
        if node.opaque:
            return
        # round-robin over the gadgets (pairs last), so the proposal cap leaves every gadget some proposals
        units = self._units(node)
        tiers = [[u for u in units if u["lab"].count("+") != 1], [u for u in units if u["lab"].count("+") == 1]]
        seen, n = set(), 0
        for tier in tiers:
            its = [(u, canonical(u["rects"]), self._steps(u)) for u in tier]
            while its:
                nxt = []
                for unit, key0, it in its:
                    for lab, rects in it:
                        key = (unit["lab"], canonical(rects))
                        if key[1] == key0 or key in seen:
                            continue
                        seen.add(key)
                        nd = self._apply(design, node, unit, rects, ctx)
                        if nd is None:
                            continue
                        yield f"{self.name}[{unit['lab']}:{lab}]", nd
                        n += 1
                        if n >= MAX_PROPS:
                            return
                        nxt.append((unit, key0, it))
                        break
                its = nxt
