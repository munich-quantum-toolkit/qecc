# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Criger-style verification with bare-ancilla checks, built by promotion rounds.

A *gadget* is one ancilla qubit with its couplings to data qubits, its flags and its measurement: a stabilizer check of
the state pulled back to a base position, measured through its own |0> / |+> ancilla, or (``block`` set) one ancilla
qubit of a block's prepared state (``blocks.BlockSpec``), whose checks are parities over the block's measurements. A
flag is a second ancilla coupled to the gadget ancilla at two couplings; it catches hook faults between them. The
circuit is always rebuilt from the base prep plus the list of gadgets (``build``).

Promotion (``promote``): exact check -> failing combinations -> ILP over candidates (checks at every position, flags on
every gadget) with cost CNOTs + measurements -> add -> repeat. From round ``SAFE_FROM`` on only hook-safe candidates
are allowed (flags, and checks of weight <= 3 that are not at the end), so the hooks of the last round cannot create
new uncatchable faults (Criger et al.'s final-round restriction).
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import operator
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from . import memo
from .blocks import dual
from .circuit import MEASUREMENTS, Circuit, Op, layering
from .combos import Combos
from .cover import EXACT_LIMIT, group_elements
from .faults import fault_table
from .setcover import solve_cover
from .verify import CSSState, check_ft, fault_classes, gf2_row_basis

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .blocks import BlockSpec
    from .verify import FTReport

logger = logging.getLogger(__name__)


def _debug(entry: object) -> None:
    """Default ``log`` of the check engines: a debug message."""
    logger.debug("%s", entry)


@dataclass
class Gadget:
    """A check measured through one ancilla qubit: its couplings, flags and position in the prep."""

    kind: str  # "Z" (Z-type check) or "X"
    pos: int  # base position
    support: list[int]  # data qubit indices (order = coupling order)
    flags: list[tuple[int, int, int]] = field(default_factory=list)  # (j, k, fid): flag spans couplings j .. k-1
    slots: list[int] | None = None  # per coupling: base op index it goes before (non-decreasing); None: all at pos
    ticks: list[float] | None = None  # per coupling: order among all couplings at the same slot (default: gadget order)
    late: bool = False  # a check after all the node's blocks (``pos`` / ``slots`` unused)
    # flag takeover (Forlivesi & Amaro): (j, fid) = flag fid drives coupling j instead of the check ancilla. Between its
    # two CNOTs a flag holds a copy of the ancilla, so the state is unchanged. Valid only for an unshared flag of this
    # gadget with fj <= j < fk; ``build`` ignores other entries.
    drive: tuple[tuple[int, int], ...] = ()
    block: int | None = None  # index of the node's block whose prepared state this ancilla qubit belongs to
    anc: int = 0  # the ancilla qubit of that block

    def copy(self) -> Gadget:
        """Copy with its own support and flag lists."""
        return dataclasses.replace(self, support=list(self.support), flags=list(self.flags))

    def coupling_slots(self) -> list[int]:
        """Base op index before which each coupling is inserted."""
        return list(self.slots) if self.slots is not None else [self.pos] * len(self.support)

    def driven(self, shared: set[int] | frozenset[int] = frozenset()) -> dict[int, int]:
        """Valid takeover entries: coupling index -> flag id (see ``drive``)."""
        spans = {fid: (fj, fk) for fj, fk, fid in self.flags}
        out = {}
        for j, fid in self.drive:
            if (
                fid in spans
                and fid not in shared
                and spans[fid][0] <= j < min(spans[fid][1], len(self.support))
                and j not in out
            ):
                out[j] = fid
        return out

    @property
    def cost(self) -> int:
        """Operation count: one per coupling, one for the ancilla and three per flag."""
        return len(self.support) + 1 + 3 * len(self.flags)


def layered_base(circ: Circuit) -> tuple[Circuit, list[int]]:
    """Same prep with ops in ASAP-layer order; returns (circuit, base positions of layer boundaries).

    Raises:
        ValueError: A reset or H acts on a qubit after a CNOT on it (layering moves these ops before all CNOTs).
    """
    lay = layering(circ)
    for op, end in zip(circ.ops, lay.slice_of_op, strict=True):
        if op.name != "CX" and end > 0:
            msg = f"{op.name} after a CNOT on qubit {op.qubits[0]} is not supported"
            raise ValueError(msg)
    pre = [op for op in circ.ops if op.name != "CX"]
    ops = pre + [Op("CX", g) for t in range(1, lay.depth + 1) for g in lay.gates[t]]
    bounds = [len(pre)]
    for t in range(1, lay.depth + 1):
        bounds.append(bounds[-1] + len(lay.gates[t]))
    return Circuit(ops, circ.num_qubits, list(circ.data)), bounds


# order among the couplings at one base position: bare checks (gadget order), then blocks (block order, coupling round
# by round), then late checks (after everything else, i.e. after all blocks)
_BLOCK_TICK, _LATE_TICK = 1e6, 2e6


def _tick(g: Gadget, gi: int, j: int) -> float:
    if g.ticks is not None:
        return g.ticks[j]
    if g.late:
        return _LATE_TICK + gi
    if g.block is not None:
        return _BLOCK_TICK + 1e3 * g.block + j
    return float(gi)


def build(
    base: Circuit, gadgets: list[Gadget], blocks: list[tuple[BlockSpec, Circuit]] | None = None
) -> tuple[Circuit, dict[int, int]]:
    """Compile a base prep with its gadgets and block ancilla circuits.

    ``blocks[b]`` = (BlockSpec, circuit of block b with one data qubit per ancilla qubit). Gadgets at one position come
    in list order before the base op there; late gadgets come at the end. Detectors: the base's own, one per bare-check
    or flag measurement, each block circuit's own, and per block one per row of its ancilla check matrix.

    Returns:
        The circuit and, per base position, the op index of the base op there (where a new gadget would go).
    """
    blocks = blocks or []
    ops: list[Op] = []
    tags: list[tuple] = []  # provenance of every op (stable across adding gadgets / flags)
    nq0 = base.num_qubits
    end = len(base.ops)
    n_flags = 1 + max((fid for g in gadgets for *_, fid in g.flags), default=-1)
    nq = nq0 + len(gadgets) + n_flags
    where: dict[int, int] = {}
    # block b: ancilla qubit a -> gadget; the block circuit's other qubits get fresh ids after the flags
    members: dict[int, dict[int, int]] = {}
    for gi, g in enumerate(gadgets):
        if g.block is not None:
            members.setdefault(g.block, {})[g.anc] = gi
    qmaps: dict[int, dict[int, int]] = {}
    for b, (_, circ_b) in enumerate(blocks):
        mem = members.get(b, {})
        if sorted(mem) != list(range(len(circ_b.data))):
            msg = f"block {b}: gadgets for ancilla qubits {sorted(mem)}, its circuit has {len(circ_b.data)}"
            raise ValueError(msg)
        qm = {circ_b.data[a]: nq0 + gi for a, gi in mem.items()}
        for q in range(circ_b.num_qubits):
            if q not in qm:
                qm[q] = nq
                nq += 1
        qmaps[b] = qm
    if set(members) - set(range(len(blocks))):
        msg = "gadget of a block that does not exist"
        raise ValueError(msg)
    # shared flags: a flag id used by several gadgets (same kind) is one flag qubit coupled to each of their ancillas
    # at its two coupling points; it is reset before its first flag CX and measured after its last one
    users: dict[int, list[int]] = {}
    for gi, g in enumerate(gadgets):
        for *_, fid in g.flags:
            users.setdefault(fid, []).append(gi)
    shared = {fid for fid, us in users.items() if len(us) > 1}
    for fid in shared:
        if len({gadgets[gi].kind for gi in users[fid]}) > 1:
            msg = f"shared flag {fid} on gadgets of different kinds"
            raise ValueError(msg)
    flag_cx_left = {fid: 2 * len(users[fid]) for fid in shared}
    # events per base position: (tick, gadget index, coupling index); couplings of a gadget in coupling order
    events: dict[int, list[tuple[float, int, int]]] = {}
    for gi, g in enumerate(gadgets):
        sl = [end] * len(g.support) if g.late else g.coupling_slots()
        if any(b < a for a, b in itertools.pairwise(sl)):
            msg = "gadget slots must be non-decreasing"
            raise ValueError(msg)
        for j, p in enumerate(sl):
            events.setdefault(p, []).append((_tick(g, gi, j), gi, j))

    def flag_cx(g: Gadget, gi: int, a: int, f: int, fid: int, j: int | str) -> None:
        if fid in shared and flag_cx_left[fid] == 2 * len(users[fid]):
            ops.append(Op("RX" if g.kind == "Z" else "R", (f,)))
            tags.append(("fr", fid))
        ops.append(Op("CX", (f, a)) if g.kind == "Z" else Op("CX", (a, f)))
        tags.append(("fc", fid, gi, j) if fid in shared else ("fc", fid, j))
        if fid in shared:
            flag_cx_left[fid] -= 1
            if flag_cx_left[fid] == 0:
                ops.append(Op("MX" if g.kind == "Z" else "M", (f,)))
                tags.append(("fm", fid))

    emitted: set[int] = set()

    def block_circuit(b: int) -> None:
        """Block b's ancilla circuit, once, right before the first coupling of any of its ancilla qubits."""
        emitted.add(b)
        qm = qmaps[b]
        for k, op in enumerate(blocks[b][1].ops):
            ops.append(Op(op.name, tuple(qm[q] for q in op.qubits)))
            tags.append(("a", b, k))

    drives = [g.driven(shared) for g in gadgets]
    for p in range(end + 1):
        for _, gi, j in sorted(events.get(p, []), key=operator.itemgetter(0, 1, 2)):
            g = gadgets[gi]
            a = nq0 + gi  # stable qubit ids: gadget ancillas, then flags by global flag id
            fl = [nq0 + len(gadgets) + fid for *_, fid in g.flags]
            fids = [fid for *_, fid in g.flags]
            if j == 0:
                if g.block is None:
                    ops.append(Op("R" if g.kind == "Z" else "RX", (a,)))
                    tags.append(("r", gi))
                elif g.block not in emitted:
                    block_circuit(g.block)
                for f, fid in zip(fl, fids, strict=True):
                    if fid not in shared:
                        ops.append(Op("RX" if g.kind == "Z" else "R", (f,)))
                        tags.append(("fr", fid))
            q = g.support[j]
            d = base.data[q]
            for (fj, fk, fid), f in zip(g.flags, fl, strict=True):
                if j in {fj, fk}:
                    flag_cx(g, gi, a, f, fid, j)
            src = (
                nq0 + len(gadgets) + drives[gi][j] if j in drives[gi] else a
            )  # takeover: the flag drives this coupling
            ops.append(Op("CX", (d, src)) if g.kind == "Z" else Op("CX", (src, d)))
            tags.append(("c", gi, j))
            if j == len(g.support) - 1:
                for (_fj, fk, fid), f in zip(g.flags, fl, strict=True):
                    if fk >= len(g.support):
                        flag_cx(g, gi, a, f, fid, "end")
                ops.append(Op("M" if g.kind == "Z" else "MX", (a,)))
                tags.append(("m", gi))
                for f, fid in zip(fl, fids, strict=True):
                    if fid not in shared:
                        ops.append(Op("MX" if g.kind == "Z" else "M", (f,)))
                        tags.append(("fm", fid))
        where[p] = len(ops)
        if p < end:
            ops.append(base.ops[p])
            tags.append(("b", p))
    circ = Circuit(ops, nq, list(base.data))
    # detectors, in record order of the tags
    rec: dict[tuple, int] = {}
    for op, tg in zip(ops, tags, strict=True):
        if op.name in MEASUREMENTS:
            rec[tg] = len(rec)
    dets: list[list[int]] = []
    if base.detectors is not None:
        # a base with its own detectors: its records renumbered (gadget measurements come between them)
        base_rec = [r for tg, r in rec.items() if tg[0] == "b"]
        dets += [[base_rec[i] for i in d] for d in base.detectors]
        own = set(base_rec)
    else:
        own = set()
    in_block = {rec["m", gi] for gi, g in enumerate(gadgets) if g.block is not None}
    for b, (spec, circ_b) in enumerate(blocks):
        b_rec = [r for tg, r in rec.items() if tg[0] == "a" and tg[1] == b]  # the block circuit's k-th measurement
        own |= set(b_rec)
        cd = [[i] for i in range(circ_b.num_measurements)] if circ_b.detectors is None else circ_b.detectors
        dets += [[b_rec[i] for i in d] for d in cd]
        m_of = {a: rec["m", gi] for a, gi in members.get(b, {}).items()}
        dets += [[m_of[int(a)] for a in np.nonzero(row)[0]] for row in spec.restricted()]
    dets += [[r] for r in range(len(rec)) if r not in own and r not in in_block]
    circ.detectors = dets
    circ.tags = tags
    return circ, where


def pulled_back_groups(
    base: Circuit, state: CSSState, positions: list[int], *, max_elems: int = 8192, seed: int = 0
) -> dict:
    """Per base position and kind ("Z", "X"): the state's stabilizer elements pulled back to it (rows over the data).

    All nonzero elements for small groups (``cover.EXACT_LIMIT``), else a sample per position
    (``cover.group_elements``). A base with its own ancillas: only elements without support on them at that position.
    """
    anc = [q for q in range(base.num_qubits) if q not in set(base.data)]
    out = {}
    for kind, basis in (("Z", gf2_row_basis(state.sz)), ("X", gf2_row_basis(state.sx))):
        r = basis.shape[0]
        if (1 << r) - 1 > EXACT_LIMIT:
            full = np.zeros((r, base.num_qubits), dtype=np.uint8)
            full[:, base.data] = basis
            cur = full.copy()
            snap = {len(base.ops): cur.copy()}
            for i in range(len(base.ops) - 1, -1, -1):
                op = base.ops[i]
                if op.name == "CX":
                    c, t = op.qubits
                    if kind == "Z":
                        cur[:, c] ^= cur[:, t]
                    else:
                        cur[:, t] ^= cur[:, c]
                snap[i] = cur.copy()
            for p in positions:
                bp = snap[p][:, base.data]
                if anc and snap[p][:, anc].any():
                    bp = (dual(snap[p][:, anc].T) @ snap[p] % 2)[:, base.data].astype(np.uint8)
                g, _ = group_elements(bp, max_elems, seed=seed + p)
                rb = bp.shape[0]
                out.setdefault(p, {})[kind] = ((((g[:, None] >> np.arange(rb)) & 1).astype(np.uint8) @ bp) % 2).astype(
                    np.uint8
                )
            continue
        coeff = np.arange(1, 1 << r)
        v = (((coeff[:, None] >> np.arange(r)) & 1).astype(np.uint8) @ basis) % 2
        full = np.zeros((len(coeff), base.num_qubits), dtype=np.uint8)
        full[:, base.data] = v
        # walk back through the base ops from the end
        cur = full.copy()
        snap = {len(base.ops): cur.copy()}
        for i in range(len(base.ops) - 1, -1, -1):
            op = base.ops[i]
            if op.name == "CX":
                c, t = op.qubits
                if kind == "Z":
                    cur[:, c] ^= cur[:, t]
                else:
                    cur[:, t] ^= cur[:, c]
            snap[i] = cur.copy()
        for p in positions:
            keep = ~snap[p][:, anc].any(axis=1) if anc else slice(None)
            out.setdefault(p, {})[kind] = snap[p][keep][:, base.data]
    return out


def apply_candidate(
    gadgets: list[Gadget],
    c: tuple,
    fid: int,
    *,
    late_at: int | None = None,
    img: dict | None = None,
) -> list[Gadget]:
    """The gadgets with one check candidate added.

    Candidates: ("S", kind, p, support) a stabilizer check at base position p (a late check if p == ``late_at``); ("G",
    kind, ((p, q), ...)) a circuit gauge operator, each coupling at its own location; ("T", kind, p1, p2, q) a two-time
    check of qubit q between p1 and p2 (``img``: forward images of Paulis); ("F", gi, j, k) flag ``fid`` on gadget gi
    spanning couplings j .. k-1.
    """
    out = [g.copy() for g in gadgets]
    if c[0] == "S":
        _, kind, p, sup = c
        out.append(Gadget(kind, 0, list(sup), late=True) if p == late_at else Gadget(kind, p, list(sup)))
    elif c[0] == "G":
        _, kind, e = c
        out.append(Gadget(kind, e[0][0], [q for _, q in e], slots=[pp for pp, _ in e]))
    elif c[0] == "T":
        _, kind, p1, p2, q = c
        assert img is not None, "two-time candidates need ``img``"
        tgt = [int(x) for x in np.nonzero(img[kind][p1][p2][q])[0]]
        out.append(Gadget(kind, p1, [q, *tgt], slots=[p1] + [p2] * len(tgt)))
    else:
        out[c[1]].flags.append((c[2], c[3], fid))
    return out


GAUGE_CHUNK = 4096  # gauge elements scored at once
STAB_CHUNK = 2048  # stabilizer elements scored at once
W_MAX = 8  # largest stabilizer-check weight among the candidates
SAFE_FROM = 4  # promotion: hook-safe candidates only from this round of a fault order on
MAX_COMBOS = 20000  # failing combinations a promotion round or greedy step scores at most (``combos.Combos``)
PER_POS = 200  # promotion: stabilizer candidates per position and kind (gauge candidates: ten times as many)

# candidates, their coverage of the failing combinations (combinations x candidates, uint8) and their costs
Block = tuple[list[tuple], np.ndarray, np.ndarray]


@dataclass
class CandidateSpace:
    """Where the check candidates of a preparation circuit go and what they can measure there."""

    base: Circuit  # gadget positions refer to its ops
    positions: list[int]  # base positions of the stabilizer-check candidates
    groups: dict  # per position the pulled-back stabilizer elements (``pulled_back_groups``)
    gauge: dict[str, list]  # per kind the circuit gauge elements (``gauge_elements``)
    last_loc: dict[int, int]  # per data index the base position right after its last op

    @property
    def end(self) -> int:
        """The base position after the prep."""
        return len(self.base.ops)


def candidate_space(
    base: Circuit,
    state: CSSState,
    *,
    layer: bool = True,
    positions: list[int] | None = None,
    gauge_w: int = 0,
    seed: int = 0,
) -> CandidateSpace:
    """The candidate space of both check engines (``promote`` and ``greedy_ft.greedy_ft``).

    ``layer``: put the ops in ASAP-layer order and place candidates at layer boundaries (else at every gate boundary).
    ``positions``: candidate positions instead of every boundary. ``gauge_w``: also circuit gauge elements of weight
    <= ``gauge_w`` (0: none). ``seed``: seed of the element samples of large groups.
    """
    base, bounds = layered_base(base) if layer else (base, list(range(len(base.ops) + 1)))
    end = len(base.ops)
    positions = sorted({*bounds, end}) if positions is None else sorted(set(positions))
    last_loc = {}
    for i, op in enumerate(base.ops):
        for q in op.qubits:
            if q in base.data:
                last_loc[base.data.index(q)] = i + 1
    return CandidateSpace(
        base=base,
        positions=positions,
        groups=pulled_back_groups(base, state, positions, seed=seed),
        gauge={k: (_gauge_cached(base, k, gauge_w) if gauge_w > 0 else []) for k in ("Z", "X")},
        last_loc=last_loc,
    )


class CheckCoverage:
    """Which failing fault combinations of a circuit from ``build`` each check candidate catches.

    A candidate catches a combination only if it detects it for every member of each fault class involved
    (``combos.Combos``).
    """

    def __init__(
        self,
        space: CandidateSpace,
        circ: Circuit,
        where: dict[int, int],
        gadgets: list[Gadget],
        state: CSSState,
        rep: FTReport,
        *,
        flag_on: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        """Build the fault table and failing combinations of ``circ`` = ``build(space.base, gadgets)`` (``where``).

        ``rep``: the exact check of ``circ`` with collected failures. ``flag_on``: gadgets that get flag candidates
        (default: all). ``seed``: the combination sample (``combos.Combos``).
        """
        self.space, self.where, self.gadgets = space, where, gadgets
        # a new flag (j, k) on a gadget flips iff the ancilla's frame differs right before couplings j and k: its two
        # flag CXs come last among the ops there and it touches nothing else. One probe per coupling (``flag_probes``)
        probes, self.probe_col = flag_probes(circ, gadgets, space.base.num_qubits, flag_on)
        # frames only at the base positions that candidates read (one copy per position: all of them can take GBs)
        need = {*space.positions, *space.last_loc.values()} | {
            p for els in space.gauge.values() for e in els for p, _ in e
        }
        self.tab = fault_table(
            circ, record_at=sorted({where[p] for p in need}), frame_qubits=space.base.data, probes=probes
        )
        self.combos = Combos(rep, fault_classes(circ, state), max_combos=MAX_COMBOS, seed=seed)

    def frame(self, p: int, kind: str) -> np.ndarray:
        """The frames a check of ``kind`` at base position ``p`` sees (faults x data, rows ``combos.faults``)."""
        fx, fz = self.tab.frame(self.where[p], self.space.base.data)
        return (fx if kind == "Z" else fz)[self.combos.faults]

    def stabilizers(self, *, safe: bool = False, per_pos: int | None = None, asap_end: bool = False) -> Iterator[Block]:
        """Blocks of stabilizer checks ("S", kind, p, support) that catch a failing combination.

        Of weight at most ``W_MAX``. ``safe``: hook-safe candidates only (weight <= 3, not at the end). ``per_pos``: per
        position and kind only the cheapest check per coverage pattern, at most that many. ``asap_end``: every check at
        the end also as a variant ("G", kind, locations) coupled to each qubit right after its last prep op (Criger et
        al.'s placement).
        """
        sp, combine = self.space, self.combos.combine
        for p in sp.positions:
            cap = (3 if p != sp.end else 0) if safe else W_MAX
            for kind in ("Z", "X"):
                rows = sp.groups[p][kind]
                w = rows.sum(axis=1)
                cand = rows[(w > 0) & (w <= cap)]
                if not len(cand):
                    continue
                f32 = self.frame(p, kind).astype(np.float32)
                chunks = []
                for lo in range(0, len(cand), STAB_CHUNK):
                    part = cand[lo : lo + STAB_CHUNK]
                    comb = combine(((f32 @ part.T.astype(np.float32)) % 2).astype(np.uint8))
                    useful = comb.any(axis=0)
                    if per_pos is None and not asap_end:
                        if useful.any():
                            yield self._stab_block(kind, p, part[useful], comb[:, useful])
                        continue
                    chunks.append((part[useful], comb[:, useful]))
                if per_pos is None and not asap_end:
                    continue
                cand = np.concatenate([c for c, _ in chunks])
                comb = np.concatenate([b for _, b in chunks], axis=1)
                if not len(cand):
                    continue
                if per_pos is not None:
                    order = np.argsort(cand.sum(axis=1), kind="stable")
                    packed = np.packbits(comb[:, order].T, axis=1)
                    _, first = np.unique(packed, axis=0, return_index=True)
                    keep = order[np.sort(first)][:per_pos]
                    cand, comb = cand[keep], comb[:, keep]
                if not (asap_end and p == sp.end):
                    yield self._stab_block(kind, p, cand, comb)
                    continue
                cands, cols = [], []
                frames: dict[int, np.ndarray] = {}
                for k in range(len(cand)):
                    sup = tuple(int(q) for q in np.nonzero(cand[k])[0])
                    cands.append(("S", kind, p, sup))
                    cols.append(comb[:, k])
                    e = tuple(sorted((sp.last_loc[q], q) for q in sup))
                    det = np.zeros(len(self.combos.faults), dtype=np.uint8)
                    for pp, q in e:
                        if pp not in frames:
                            frames[pp] = self.frame(pp, kind)
                        det ^= frames[pp][:, q]
                    col = combine(det[:, None])[:, 0]
                    if col.any():
                        cands.append(("G", kind, e))
                        cols.append(col)
                costs = [len(c[-1]) + 1 for c in cands]
                yield cands, np.stack(cols, axis=1), np.array(costs)

    @staticmethod
    def _stab_block(kind: str, p: int, cand: np.ndarray, comb: np.ndarray) -> Block:
        cands = [("S", kind, p, tuple(int(q) for q in np.nonzero(row)[0])) for row in cand]
        return cands, comb, cand.sum(axis=1).astype(np.int64) + 1

    def gauge(self, *, safe: bool = False, limit: int | None = None) -> Iterator[Block]:
        """Blocks of circuit gauge elements ("G", kind, locations) that catch a failing combination (Criger et al.).

        An element detects a fault iff the parity of the fault's frames at its locations is odd. ``safe``: weight <= 3
        only. ``limit``: per kind, lightest first, only the first element per coverage pattern, at most that many.
        """
        for kind in ("Z", "X"):
            els = self.space.gauge[kind] if not safe else [e for e in self.space.gauge[kind] if len(e) <= 3]
            if not els:
                continue
            locs = sorted({loc for e in els for loc in e})
            lidx = {loc: i for i, loc in enumerate(locs)}
            frames: dict[int, np.ndarray] = {}
            fr = np.zeros((len(self.combos.faults), len(locs)), dtype=np.uint8)
            for i, (pp, q) in enumerate(locs):
                if pp not in frames:
                    frames[pp] = self.frame(pp, kind)
                fr[:, i] = frames[pp][:, q]
            rows = np.concatenate([[lidx[loc] for loc in e] for e in els])
            colsi = np.concatenate([[k] * len(e) for k, e in enumerate(els)])
            inc = csc_matrix((np.ones(len(rows), dtype=np.int32), (rows, colsi)), shape=(len(locs), len(els)))
            frs = csr_matrix(fr.astype(np.int32))
            # in chunks: all elements at once can take several GB
            order = np.argsort([len(e) for e in els], kind="stable")
            seen: set[bytes] = set()
            for lo in range(0, len(order), GAUGE_CHUNK):
                part = order[lo : lo + GAUGE_CHUNK]
                comb = self.combos.combine((np.asarray((frs @ inc[:, part]).todense()) % 2).astype(np.uint8))
                pick = []
                for j in np.nonzero(comb.any(axis=0))[0]:
                    if limit is not None:
                        pat = np.packbits(comb[:, j]).tobytes()
                        if pat in seen:
                            continue
                        seen.add(pat)
                    pick.append(j)
                    if len(seen) == limit:
                        break
                if pick:
                    cands = [("G", kind, els[int(part[j])]) for j in pick]
                    yield cands, comb[:, pick], np.array([len(c[-1]) + 1 for c in cands])
                if len(seen) == limit:
                    break

    def flags(self, cost: int = 3) -> Iterator[Block]:
        """One block of flags ("F", gi, j, k) on the gadgets ``flag_on`` that catch a failing combination."""
        probe = self.tab.probe
        cands, dets = [], []
        for gi in sorted({gi for gi, _ in self.probe_col}):
            g = self.gadgets[gi]
            m = len(g.support)
            for j in range(m - 1):
                for k in range(j + 2, m + 1):
                    if any((j, k) == (a, b) for a, b, _ in g.flags):
                        continue
                    assert probe is not None, "the table was built with probes"
                    cands.append(("F", gi, j, k))
                    dets.append(
                        probe[self.combos.faults, self.probe_col[gi, j]]
                        ^ probe[self.combos.faults, self.probe_col[gi, k]]
                    )
        if not cands:
            return
        comb = self.combos.combine(np.stack(dets, axis=1))
        useful = np.nonzero(comb.any(axis=0))[0]
        if len(useful):
            yield [cands[i] for i in useful], comb[:, useful], np.full(len(useful), cost)


@dataclass
class PromotionLog:
    """Record of a promotion: per-round data, whether it ended FT, and alternative gadget sets."""

    rounds: list[dict] = field(default_factory=list)
    ok: bool = False
    alternatives: list[list[Gadget]] = field(default_factory=list)  # with ``alternatives`` > 0


def promote(
    base: Circuit,
    state: CSSState,
    t: int,
    *,
    positions: list[int] | None = None,
    rounds: int = 10,
    time_limit: float = 300.0,
    incremental: bool = False,
    blocks: list | None = None,
    gadgets0: list[Gadget] | None = None,
    gauge_w: int = 0,
    asap_end: bool = False,
    alternatives: int = 0,
    stage_round0: int = 0,
    layer: bool = True,
    cost_jitter: float = 0.0,
    seed: int = 0,
    late: bool = False,
    log: Callable[..., object] = _debug,
) -> tuple[Circuit, list[Gadget], PromotionLog]:
    """Criger's ILP promotion: add checks and flags to the scope's prep until it is FT.

    ``blocks``: the scope's blocks as (BlockSpec, compiled ancilla circuit), coupled through their gadgets in
    ``gadgets0``, the start gadgets (positions relative to the layered base). ``gauge_w`` > 0: also circuit gauge
    operators of weight <= gauge_w as candidates (``gauge_elements``). ``asap_end``: see
    ``CheckCoverage.stabilizers``. ``cost_jitter`` > 0: ILP costs multiplied by exp(cost_jitter * N(0, 1)) (seeded):
    another cheap set of checks per seed. A round covers at most ``MAX_COMBOS`` failing combinations
    (``combos.Combos``), the rest in later rounds (every round re-checks exactly); each such round adds a round to the
    budget, up to ``3 * rounds``. ``late``: checks at the end of the prep become late checks (after all blocks).
    """
    space = candidate_space(base, state, layer=layer, positions=positions, gauge_w=gauge_w, seed=seed)
    base, end = space.base, space.end
    if gauge_w > 0:
        log({"gauge_candidates": {k: len(v) for k, v in space.gauge.items()}})

    def full(gs: list[Gadget]) -> tuple[Circuit, dict[int, int]]:
        return build(base, gs, blocks)

    gadgets: list[Gadget] = [g.copy() for g in gadgets0 or []]
    next_fid = 1 + max((fid for g in gadgets for *_, fid in g.flags), default=-1)
    plog = PromotionLog()
    t0 = time.time()
    t_cur = 1 if incremental else t
    stage_round = stage_round0
    rounds_lim = rounds
    rnd = -1
    while rnd < rounds_lim:
        rnd += 1
        circ, where = full(gadgets)
        rep = check_ft(circ, state, t_cur, collect=True)
        while rep.ok and t_cur < t:  # Criger-style promotion: raise the target once it is met
            t_cur += 1
            stage_round = 0
            rep = check_ft(circ, state, t_cur, collect=True)
        entry = {
            "round": rnd,
            "t_cur": t_cur,
            "failures": dict(rep.failures),
            "n_gadgets": len(gadgets),
            "cost": sum(g.cost for g in gadgets),
            "s": round(time.time() - t0, 1),
        }
        plog.rounds.append(entry)
        log(entry)
        if rep.ok and t_cur == t:
            plog.ok = True
            return circ, gadgets, plog
        if rnd == rounds_lim:
            break
        safe = stage_round >= SAFE_FROM
        stage_round += 1
        cov = CheckCoverage(space, circ, where, gadgets, state, rep, seed=seed * 1000 + rnd)
        cb = cov.combos
        n_rows = len(cb)
        if cb.total > len(cb):
            log({"round": rnd, "combos": cb.total, "covered_this_round": len(cb)})
            rounds_lim = min(rounds_lim + 1, 3 * rounds)

        # bits: per candidate the failing combinations it catches (row indices), sparse
        cols, bits, costs = [], [], []
        found = itertools.chain(
            cov.stabilizers(safe=safe, per_pos=PER_POS, asap_end=asap_end),
            cov.gauge(safe=safe, limit=PER_POS * 10),
            cov.flags(),
        )
        for cands, comb, cost in found:
            cols += cands
            bits += [np.flatnonzero(comb[:, k]).astype(np.int32) for k in range(len(cands))]
            costs += [int(c) for c in cost]
        if not cols:
            log("no candidate detects the remaining failures")
            break
        indptr = np.concatenate([[0], np.cumsum([len(b) for b in bits])])
        a = csc_matrix(
            (np.ones(indptr[-1], dtype=np.int8), np.concatenate(bits), indptr), shape=(n_rows, len(bits))
        ).tocsr()
        coverable = np.diff(a.indptr) > 0

        def apply(gs: list[Gadget], c: tuple, fid: int) -> list[Gadget]:
            return apply_candidate(gs, c, fid, late_at=end if late else None)

        def deterministic(gs: list[Gadget]) -> bool:
            try:
                full(gs)[0].to_stim().detector_error_model()
            except ValueError:
                return False
            return True

        ilp_cost = np.array(costs, dtype=float)
        if cost_jitter:
            ilp_cost *= np.exp(cost_jitter * np.random.default_rng((seed, rnd)).standard_normal(len(costs)))

        # lazy conflict cuts: spread gauge checks can anticommute with other checks crossing them in time
        cuts: list[tuple[int, ...]] = []

        # ``solve_consistent`` reads this round's variables; it is defined and called in the same round
        # ruff: disable[function-uses-loop-variable]
        def solve_consistent(nogoods: list[list[int]]) -> list[int] | None:
            for _ in range(50):
                # cover the coverable failing combinations; never all of a cut (non-commuting checks) or a nogood
                sol = solve_cover(
                    a[coverable],
                    ilp_cost,
                    conflicts=[list(c) for c in cuts] + [list(ng) for ng in nogoods],
                    time_limit=time_limit,
                    seed=seed * 1000 + rnd,
                )
                if sol is None:
                    return None
                idx = sorted(sol, key=lambda i: costs[i])
                gs, fid, ok_idx, bad = gadgets, next_fid, [], None
                for i in idx:
                    trial = apply(gs, cols[i], fid)
                    if deterministic(trial):
                        gs, fid = trial, fid + (cols[i][0] == "F")
                        ok_idx.append(i)
                        continue
                    partner = None  # the earlier chosen check it conflicts with (else: an existing gadget)
                    for j in ok_idx:
                        g2 = apply(gadgets, cols[j], next_fid)
                        if not deterministic(apply(g2, cols[i], next_fid + (cols[j][0] == "F"))):
                            partner = j
                            break
                    bad = (i,) if partner is None else (partner, i)
                    break
                if bad is None:
                    return [int(i) for i in idx]
                cuts.append(bad)
            return None

        # ruff: enable[function-uses-loop-variable]

        chosen_idx = solve_consistent([])
        if chosen_idx is None:
            log(f"ILP failed: no consistent cover ({len(cuts)} cuts)")
            break
        if alternatives > 0:
            # one promotion round only: the best `alternatives` distinct consistent ILP solutions
            sols = [chosen_idx]
            while len(sols) < alternatives:
                nxt = solve_consistent(sols)
                if nxt is None:
                    break
                sols.append(nxt)
            for idx in sols:
                gs, fid = gadgets, next_fid
                for i in idx:
                    gs = apply(gs, cols[i], fid)
                    fid += cols[i][0] == "F"
                plog.alternatives.append(gs)
            entry["alternatives"] = [float(sum(costs[i] for i in idx)) for idx in sols]
            circ, _ = full(gadgets)
            return circ, gadgets, plog
        chosen = [cols[i] for i in chosen_idx]
        for c in chosen:
            gadgets = apply(gadgets, c, next_fid)
            next_fid += c[0] == "F"
        entry["cuts"] = len(cuts)
        entry["added"] = [
            c[0] + (c[1] if c[0] in {"S", "G"} else "") + (str(len(c[-1])) if c[0] in {"S", "G"} else "")
            for c in chosen
        ]
        entry["uncoverable"] = int((~coverable).sum())
    circ, _ = full(gadgets)
    return circ, gadgets, plog


def flag_probes(
    circ: Circuit, gadgets: list[Gadget], nq0: int, which: list[int] | None = None
) -> tuple[list[tuple[int, int, str]], dict]:
    """Frame probes for the flag trials (``fault_table(probes=...)``) and their index ``(gi, j) -> probe``.

    Per gadget gi (all, or ``which``) and coupling j (j = m: its measurement): the ancilla's Z (Z check) or X (X check)
    frame bit right before that op. A new flag (j, k) on gi sees probe (gi, j) XOR probe (gi, k).
    """
    assert circ.tags is not None, "a circuit from ``build`` has tags"
    pos_of = {tag: i for i, tag in enumerate(circ.tags)}
    probes, col = [], {}
    for gi in range(len(gadgets)) if which is None else which:
        g = gadgets[gi]
        for j in range(len(g.support) + 1):
            tag = ("c", gi, j) if j < len(g.support) else ("m", gi)
            col[gi, j] = len(probes)
            probes.append((pos_of[tag], nq0 + gi, "z" if g.kind == "Z" else "x"))
    return probes, col


def _gauge_cached(base: Circuit, kind: str, w_cap: int) -> list:
    key = memo.digest("gauge", str(base.to_stim()), tuple(base.data), kind, w_cap)
    hit = memo.get("gauge", key)
    if not isinstance(hit, list):
        hit = gauge_elements(base, kind, w_cap=w_cap)
        memo.put("gauge", key, hit)
    return hit


def gauge_elements(
    base: Circuit, kind: str, w_cap: int = 4, explore_cap: int | None = None, max_elems: int = 200000
) -> list[tuple[tuple[int, int], ...]]:
    """Low-weight elements of the circuit gauge group of a CNOT preparation circuit (Criger et al.).

    Locations: (p, q) = data qubit q right before base op p. Local generators (``kind`` "Z" or "X"): Z after |0>
    (X after |+>); for a CNOT c->t (b = before, a = after) Z_c(b) Z_c(a), Z_t(b) Z_c(a) Z_t(a), or X_c(b) X_c(a) X_t(a),
    X_t(b) X_t(a). Products of overlapping generators up to weight ``explore_cap``; returns those of weight <= ``w_cap``
    as sorted tuples of locations.
    """
    explore_cap = explore_cap if explore_cap is not None else w_cap + 2
    dpos = {d: i for i, d in enumerate(base.data)}
    loc_id: dict[tuple[int, int], int] = {}

    def lid(p: int, q: int) -> int:
        return loc_id.setdefault((p, q), len(loc_id))

    last: dict[int, int] = {}  # data index -> base position of its current location
    gens: list[frozenset[int]] = []
    # |+> qubits: RX, or R followed by H before any CNOT on the qubit
    plus: set[int] = set()
    touched: set[int] = set()
    for op in base.ops:
        if op.name == "CX":
            touched.update(op.qubits)
        elif op.name in {"RX", "H"} and op.qubits[0] in dpos and op.qubits[0] not in touched:
            plus.add(dpos[op.qubits[0]])
        elif op.name == "R" and op.qubits[0] in dpos and op.qubits[0] not in touched:
            plus.discard(dpos[op.qubits[0]])
    for i, op in enumerate(base.ops):
        if op.name in {"R", "RX", "H"}:
            q = dpos.get(op.qubits[0])
            if q is not None:
                last[q] = i + 1
        elif op.name == "CX" and op.qubits[0] in dpos and op.qubits[1] in dpos:
            c, t = dpos[op.qubits[0]], dpos[op.qubits[1]]
            cb, tb = lid(last[c], c), lid(last[t], t)
            ca, ta = lid(i + 1, c), lid(i + 1, t)
            if kind == "Z":
                gens += [frozenset({cb, ca}), frozenset({tb, ca, ta})]
            else:
                gens += [frozenset({cb, ca, ta}), frozenset({tb, ta})]
            last[c] = last[t] = i + 1
    # initialization generators at the first location of every data qubit
    first: dict[int, int] = {}
    for i, op in enumerate(base.ops):
        if op.name in {"R", "RX", "H"} and op.qubits[0] in dpos:
            first[dpos[op.qubits[0]]] = i + 1
    for q, p in first.items():
        is_plus = q in plus
        if (kind == "Z" and not is_plus) or (kind == "X" and is_plus):
            gens.append(frozenset({lid(p, q)}))
    # symmetric-difference closure restricted to small weight, via overlapping generators
    by_loc: dict[int, list[int]] = {}
    for gi, g in enumerate(gens):
        for l in g:
            by_loc.setdefault(l, []).append(gi)
    seen: set[frozenset[int]] = set()
    frontier = [g for g in gens if len(g) <= explore_cap]
    seen.update(frontier)
    while frontier and len(seen) < max_elems:
        nxt = []
        for e in frontier:
            for gi in {gi for l in e for gi in by_loc[l]}:
                f = e ^ gens[gi]
                if 0 < len(f) <= explore_cap and f not in seen:
                    seen.add(f)
                    nxt.append(f)
        frontier = nxt
    inv = {v: k for k, v in loc_id.items()}
    out = [tuple(sorted(inv[l] for l in e)) for e in seen if len(e) <= w_cap]
    return sorted(set(out), key=lambda e: (len(e), e))
