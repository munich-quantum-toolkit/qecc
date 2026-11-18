# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Design tree of a composed fault-tolerant state preparation.

A *node* prepares a CSS state on its own qubits: a layered preparation circuit, gadgets inserted into it (bare checks
with their flags, and the ancilla qubits of blocks) and blocks, each with a child node that prepares its ancilla state.
The root node prepares the data state.

Each node has two states: ``state``, what it prepares (noiseless correctness), and ``target``, what its local FT check
judges. A block ancilla's target holds only the error type that propagates back onto its parent; its other errors only
flip its own outcomes, which the global check judges. FT is always decided by the global check of the compiled root
circuit; local checks only say which scopes are still open.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .blocks import BlockSpec, dual
from .gauge import Gadget, build, layered_base
from .verify import CSSState, FTReport, WeightedTarget, check_ft

if TYPE_CHECKING:
    from collections.abc import Callable

    from .circuit import Circuit


@dataclass
class Node:
    """One scope of a design: a preparation circuit, its gadgets and its blocks, each block with its ancilla's node."""

    name: str
    base: Circuit  # layered preparation circuit (data = this node's qubits); for ``opaque`` nodes a complete circuit
    state: CSSState
    target: CSSState
    gadgets: list[Gadget] = field(default_factory=list)  # in the preparation circuit, or (``late``) after all blocks
    blocks: list[tuple[BlockSpec, Node]] = field(default_factory=list)
    opaque: bool = False  # base already contains its own checks (external tool output): no gadgets
    origin: str = ""
    layered: bool = True  # base is the ASAP-layered preparation circuit (else its own gate order)

    def children(self) -> list[Node]:
        """The nodes of the block ancillas."""
        return [c for _, c in self.blocks]

    @property
    def prep_gadgets(self) -> list[Gadget]:
        """Every gadget that is not a late check (bare checks in the preparation circuit, blocks' ancilla qubits)."""
        return [g for g in self.gadgets if not g.late]

    def add_block(self, spec: BlockSpec, child: Node) -> int:
        """Append a block and one gadget per ancilla qubit; return the block's index."""
        b = len(self.blocks)
        self.blocks.append((spec, child))
        pos = len(self.base.ops) if spec.pos is None else spec.pos
        for a, grp in enumerate(spec.anc_groups()):
            self.gadgets.append(Gadget(spec.kind, pos, list(grp), block=b, anc=a))
        return b

    def truncated(self, k: int) -> Node:
        """A shallow copy with only the first ``k`` blocks (and their gadgets)."""
        out = copy.copy(self)
        out.blocks = self.blocks[:k]
        out.gadgets = [g for g in self.gadgets if g.block is None or g.block < k]
        return out

    def remove_block(self, b: int) -> None:
        """Remove block ``b`` with its gadgets (indices of later blocks shift down)."""
        self.blocks = [x for i, x in enumerate(self.blocks) if i != b]
        out = []
        for g in self.gadgets:
            if g.block == b:
                continue
            out.append(dataclasses.replace(g, block=g.block - 1) if g.block is not None and g.block > b else g)
        self.gadgets = out

    @property
    def late_gadgets(self) -> list[Gadget]:
        """The late checks (after all blocks)."""
        return [g for g in self.gadgets if g.late]


def make_node(
    name: str, prep: Circuit, state: CSSState, target: CSSState | None = None, origin: str = "", *, layer: bool = True
) -> Node:
    """Build a node from a preparation circuit; with ``layer`` the circuit is ASAP-layered first."""
    if not layer:
        return Node(name, prep, state, state if target is None else target, origin=origin, layered=False)
    base, _ = layered_base(prep)
    again, _ = layered_base(base)
    assert again.ops == base.ops, "layered_base must be idempotent"
    return Node(name, base, state, state if target is None else target, origin=origin)


def ancilla_states(spec: BlockSpec) -> tuple[CSSState, CSSState]:
    """(full state, local target) of a block's ancilla."""
    h = spec.ancilla_generator()
    full = CSSState(h, dual(h))
    m = h.shape[1]
    empty = np.zeros((0, m), dtype=np.uint8)
    # a Z block's ancilla sends Z errors back: judge Z errors (X-type stabilizers), and the other way round
    if not spec.split:
        target = CSSState(h, empty) if spec.kind == "Z" else CSSState(empty, dual(h))
    elif spec.grouped:  # back errors weigh their weight on the data (one ancilla qubit reaches its whole group)
        judge = dual(spec.harmless_back())
        sup = spec.support
        gamma, hd = spec.gamma_matrix()[sup], np.asarray(spec.checks, dtype=np.uint8)[:, sup] % 2
        if spec.kind == "Z":
            target = WeightedTarget(judge, empty, back="Z", gamma=gamma, hd=hd)
        else:
            target = WeightedTarget(empty, judge, back="X", gamma=gamma, hd=hd)
    else:  # split block: back errors modulo rowspace(H̃) and pairs on one data qubit (they cancel there)
        judge = dual(spec.harmless_back())
        target = CSSState(judge, empty) if spec.kind == "Z" else CSSState(empty, judge)
    return full, target


def compile_node(node: Node) -> Circuit:
    """Compiled circuit of the subtree; ``.base_index[p]`` = op index of base op p (or of the end, p = len)."""
    if node.opaque and not node.gadgets and not node.blocks:
        out = copy.copy(node.base)
        out.base_index = {p: p for p in range(len(node.base.ops) + 1)}
        return out
    out, where = build(node.base, node.gadgets, [(spec, compile_node(ch)) for spec, ch in node.blocks])
    out.base_index = dict(where)
    return out


def scope_name(path: tuple[int, ...]) -> str:
    """Label of a scope in histories: "root" or the block indices from the root, e.g. "0/1"."""
    return "/".join(map(str, path)) or "root"


@dataclass
class Design:
    """A design: the tree of nodes from the data preparation circuit down, for fault distance ``t``."""

    root: Node
    t: int
    history: list[str] = field(default_factory=list)  # moves since this design became a tree node (``MAX_MOVES``)
    origin: list[str] = field(default_factory=list)  # moves before that (front designs that start a subtree)

    def clone(self) -> Design:
        """Deep copy of the design."""
        return copy.deepcopy(self)

    def full_history(self) -> list[str]:
        """Provenance and actions of this run, in order."""
        return [*self.origin, *self.history]

    def with_node(self, path: tuple[int, ...], node: Node, label: str | None = None) -> Design:
        """A copy with the scope at ``path`` replaced by ``node`` (not copied) and ``label`` appended to the history."""
        if path:
            new = self.clone()
            parent = new.node(path[:-1])
            spec, _ = parent.blocks[path[-1]]
            parent.blocks[path[-1]] = (spec, node)
        else:
            new = Design(node, self.t, list(self.history), list(self.origin))
        if label is not None:
            new.history.append(label)
        return new

    def nodes(self) -> list[tuple[tuple[int, ...], Node]]:
        """(path, node) for every node; path = block indices from the root."""
        out = []

        def walk(n: Node, path: tuple[int, ...]) -> None:
            out.append((path, n))
            for i, ch in enumerate(n.children()):
                walk(ch, (*path, i))

        walk(self.root, ())
        return out

    def node(self, path: tuple[int, ...]) -> Node:
        """The node at ``path`` (block indices from the root)."""
        n = self.root
        for i in path:
            n = n.blocks[i][1]
        return n

    def circuit(self) -> Circuit:
        """The compiled circuit."""
        return compile_node(self.root)

    def structure(self) -> str:
        """Structure class of the design.

        "shared" if some block measures several checks, "grouped" if some block has an ancilla qubit with several data
        qubits or a check uses flag takeover, both, or "separate"; "opaque" for a circuit without structure.
        """
        if self.root.opaque:
            return "opaque"
        specs = [spec for _, n in self.nodes() for spec, _ in n.blocks]
        shared = any(np.asarray(sp.checks).shape[0] > 1 for sp in specs)
        grouped = any(sp.grouped for sp in specs) or any(g.drive for _, n in self.nodes() for g in n.gadgets)
        return "+".join(x for x, on in (("shared", shared), ("grouped", grouped)) if on) or "separate"

    def key(self) -> str:
        """Hash of the compiled circuit (identifies the design)."""
        return hashlib.sha1(str(self.circuit().to_stim()).encode(), usedforsecurity=False).hexdigest()


def local_check(node: Node, t: int, *, collect: bool = True) -> FTReport:
    """Exact FT check of one node against its own target."""
    return check_ft(compile_node(node), node.target, t, collect=collect)


@dataclass
class Evaluation:
    """The result of ``evaluate``: FT or not, the raw failure counts and the failing scopes."""

    ok: bool
    raw: dict  # s -> failing combinations of s elementary faults (``FTReport.raw_failures``)
    open_scopes: list[tuple[int, ...]]  # paths whose local check fails (root: the global one)


def evaluate(design: Design, check: Callable[[Circuit, CSSState, int], tuple]) -> Evaluation:
    """Global check of the design and, if it fails, the local check of every ancilla scope (``check`` -> ok, _, raw)."""
    ok, _, raw = check(design.circuit(), design.root.target, design.t)
    open_scopes: list[tuple[int, ...]] = [] if ok else [()]
    if not ok:
        for path, n in design.nodes():
            if path and not n.opaque and not check(compile_node(n), n.target, design.t)[0]:
                open_scopes.append(path)
    return Evaluation(ok, raw, open_scopes)
