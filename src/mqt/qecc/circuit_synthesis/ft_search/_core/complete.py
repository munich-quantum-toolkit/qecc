# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Completion of a partial design (the rollout): every result is exactly checked FT.

1. Known protections (``parts``) on the open ancilla scopes, then a finisher on what is still open.
2. For designs with blocks also late checks on the data after the blocks, which see the faults of every ancilla.
3. The block step (``block_repairs``); from t = 3 on it repeats (``Profile.block_rounds``), since each block's ancilla
   must tolerate t faults itself.
4. Only if 1-3 give nothing: Criger's ILP promotion, if need be on the bare preparation circuit.

Steps 1-3 run for the run's smallest, main and largest depth weight, the main one first and the other two only within
the rollout budget. Every completion is polished.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from typing import TYPE_CHECKING

from . import parts, polish
from .circuit import metrics
from .config import profile
from .design import Design, ancilla_states, compile_node, make_node
from .evaluation import check_cached, evaluate_cached, ft_cached, proposals
from .guided import guided_block_preps
from .solvers import BlockSolver, CrigerSolver, QeccSolver, scope_slack
from .verify import weighted_failures

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import Context


def _attach_parts(design: Design, ctx: Context, lam: float, keep_blocks: bool = False) -> tuple[Design, int]:
    """Replace open ancilla scopes, shallowest first, by the cheapest known part of their state under ``lam``.

    Returns the design and the number of scopes replaced. ``keep_blocks``: scopes with blocks of their own stay.
    """
    d, n = design, 0
    for _ in range(8):
        try:
            ev = evaluate_cached(d)
        except Exception:  # ruff: ignore[blind-except]
            break
        if ev.ok:
            break
        done = False
        for p in sorted((p for p in ev.open_scopes if p), key=len):
            node = d.node(p)
            if node.opaque or (keep_blocks and node.blocks):
                continue
            known = parts.known(node, d.t)
            if not known:
                continue
            d = parts.install(d, p, parts.cheapest(known, ctx, lam, scope_slack(d, p)))
            n += 1
            done = True
            break  # scopes changed: evaluate again
        if not done:
            break
    return d, n


def _finish(
    design: Design,
    ctx: Context,
    lams: tuple[float, ...],
    deadline: float | None = None,
    on_out: Callable[[str, Design], object] | None = None,
) -> list[tuple[str, Design]]:
    """Finish a design per λ: known parts on the open ancillas, then the profile's finishers.

    With nested block rounds also once with the scopes that have blocks kept, so the nested structure is finished
    rather than swapped for known parts. ``deadline``: no further λ after it. ``on_out``: called with each result as
    soon as it is found. Returns (label, design) pairs.
    """
    outs: list[tuple[str, Design]] = []
    seen = set()

    def add(lab: str, d: Design) -> None:
        outs.append((lab, d))
        if on_out is not None:
            on_out(lab, d)

    keeps = (False, True) if profile(design.t).block_rounds > 1 else (False,)
    for lam in sorted(lams, key=lambda x: x != ctx.lam):
        if deadline is not None and outs and time.time() > deadline:
            break
        c = dataclasses.replace(ctx, lam=lam)  # the greedy's depth weight is this λ's
        for keep in keeps:
            d, n = _attach_parts(design, c, lam, keep_blocks=keep)
            if d.key() in seen:
                continue
            seen.add(d.key())
            pre = f"{'keep' if keep else 'parts'}{n}"
            if _ok(d):
                add(pre, d)
                continue
            for fin in profile(design.t).finishers:
                d2 = _greedy_fix(d, c) if fin == "greedy" else _criger_fix(d, c)[1]
                if d2 is not None:
                    add(f"{pre}+{fin}" if n or keep else fin, d2)
                    break
            if d.root.blocks and (deadline is None or time.time() < deadline):
                for lab, x in _joint(d, c):
                    add(f"{pre}+{lab}" if n or keep else lab, x)
    return outs


def _joint(design: Design, ctx: Context) -> list[tuple[str, Design]]:
    """Late checks on the data after the blocks, which see the faults of every ancilla.

    Directly, and after the greedy on the data preparation circuit. Returns (label, design) pairs.
    """
    late = CrigerSolver("greedy", 80, late=True)
    outs = [("late", x) for _, x in proposals(late, design, (), ctx) if _ok(x)]
    props = proposals(CrigerSolver("greedy", 80), design, (), ctx)
    if props:
        d1 = props[0][1]
        if _ok(d1):
            outs.append(("root", d1))
        else:
            outs += [("root+late", x) for _, x in proposals(late, d1, (), ctx) if _ok(x)]
    return outs


def _cost(d: Design, ctx: Context, lam: float | None = None) -> float:
    return ctx.cost(metrics(d.circuit()), lam)


def _ok(d: Design, *, fast: bool = False) -> bool:
    """Whether ``d`` is FT; ``fast``: the fast check first, the full evaluation (it records parts) only for FT ones."""
    try:
        return (not fast or ft_cached(d.circuit(), d.root.target, d.t)) and evaluate_cached(d).ok
    except Exception:  # ruff: ignore[blind-except]
        return False


def _complete_criger(design: Design, ctx: Context) -> tuple[str, Design | None]:
    """Criger's ILP promotion with the structure kept (``_criger_fix``), else on the bare data preparation circuit.

    Returns (label, design); the design is None if both fail.
    """
    if _ok(design, fast=True):
        return "as-is", design
    label, d = _criger_fix(design, ctx)
    if d is not None:
        return label, d
    # the bare data preparation circuit
    crig = CrigerSolver("complete")
    bare = Design(copy.deepcopy(design.root), design.t, ["root:(structure dropped)"])
    bare.root.gadgets, bare.root.blocks = [], []
    for label, solver in (("criger@bare", crig), ("qecc@bare", QeccSolver())):
        for _, x in sorted(proposals(solver, bare, (), ctx), key=lambda lx: _cost(lx[1], ctx)):
            if _ok(x, fast=True):
                return label, x
    return "failed", None


def _criger_fix(design: Design, ctx: Context) -> tuple[str, Design | None]:
    """Criger's ILP promotion with the structure kept: open ancilla scopes deepest first, then the data scope.

    Returns (label, design); the design is None if the promotion fails.
    """
    if _ok(design, fast=True):
        return "as-is", design
    crig = CrigerSolver("complete")
    d = design
    # 1. open ancilla scopes, deepest first
    for path in sorted((p for p, _ in d.nodes() if p), key=len, reverse=True):
        node = d.node(path)
        if node.opaque or check_cached(compile_node(node), node.target, d.t)[0] is True:
            continue
        props = proposals(crig, d, path, ctx)
        if props:
            d = min((x for _, x in props), key=lambda x: _cost(x, ctx))
    if _ok(d, fast=True):
        return "criger@ancillas", d
    # 2. data scope with the blocks fixed: checks in the preparation circuit or after the blocks, the cheaper FT one
    cands = [("criger@ancillas+data", x) for _, x in proposals(crig, d, (), ctx)]
    late = CrigerSolver("complete", late=True)
    cands += [("criger@ancillas+late", x) for _, x in proposals(late, d, (), ctx)]
    for label, x in sorted(cands, key=lambda lx: _cost(lx[1], ctx)):
        if _ok(x, fast=True):
            return label, x
    return "failed", None


def _greedy_fix(design: Design, ctx: Context, rounds: int = 6) -> Design | None:
    """The efficiency greedy on the open scopes, deepest first, until the design is FT."""
    greedy = CrigerSolver("greedy", 80)
    d = design
    for _ in range(rounds):
        try:
            ev = evaluate_cached(d)
        except Exception:  # ruff: ignore[blind-except]
            return None
        if ev.ok:
            return d
        progressed = False
        for path in sorted(ev.open_scopes, key=len, reverse=True):
            try:
                props = proposals(greedy, d, path, ctx)
            except Exception:  # ruff: ignore[blind-except, try-except-continue]  (a failing scope is skipped)
                continue
            if props:
                d = props[0][1]
                progressed = True
        if not progressed:
            return None
    return d if _ok(d) else None


def _unprotected(design: Design, open_scopes: list) -> list[tuple[tuple[int, ...], int]]:
    """Blocks whose ancilla is a plain prep that fails its local check, as (scope, block index)."""
    out = []
    for path, node in design.nodes():
        for i, (_, ch) in enumerate(node.blocks):
            if (*path, i) in open_scopes and not (ch.opaque or ch.gadgets or ch.blocks):
                out.append((path, i))
    return out


def block_repairs(
    design: Design, ctx: Context, deadline: float | None = None, nest: bool = False
) -> list[tuple[str, Design, tuple[float, ...]]]:
    """The rollout's block step: candidate chains that protect an unprotected block, as (label, design, λs).

    An unprotected block of type K sends back errors of type K: a block of the other type beside it catches them (the
    Steane-like pair), and a block of type K inside the new block's ancilla protects that ancilla. ``nest``: every
    unprotected block's ancilla gets a block of the other type inside it instead. A design without blocks gets the
    whole pair. Only the best chain under each λ gets its λs; the others are children for the tree.
    """
    try:
        ev = evaluate_cached(design)
    except Exception:  # ruff: ignore[blind-except]
        return []
    if ev.ok:
        return []
    todo = _unprotected(design, ev.open_scopes)
    chains = []  # (failures, plain?, label, design)
    if nest:  # only where the surrounding ancilla still fails
        todo = [(p, i) for p, i in todo if not p or p in ev.open_scopes]
    if todo and nest:
        chains = _nested_chains(design, todo, ctx)
    elif todo:
        chains = _pair_chains(design, *todo[0], ctx)
    elif not any(node.blocks for _, node in design.nodes()):
        for kind in ("Z", "X"):
            if chains and deadline is not None and time.time() >= deadline:
                break
            props = proposals(BlockSolver(kind, "weight"), design, (), ctx)
            if not props:
                continue
            lab, d1 = min(props, key=lambda lx: _cost(lx[1], ctx))
            got = _pair_chains(d1, (), len(d1.root.blocks) - 1, ctx)
            if not got:  # its ancilla needs no protection: the single block is the chain
                try:
                    got = [(weighted_failures(evaluate_cached(d1).raw, design.t), True, "", d1)]
                except Exception:  # ruff: ignore[blind-except]
                    got = []
            chains += [(wf, plain, f"{lab}|{lx}" if lx else lab, d) for wf, plain, lx, d in got]
    out: dict[str, tuple[str, Design, list[float]]] = {}
    for lam in sorted(ctx.extreme_lams(), key=lambda x: x != ctx.lam):
        if chains:
            _, _, lab, d2 = min(chains, key=lambda c: (c[0], c[1], _cost(c[3], ctx, lam)))
            out.setdefault(d2.key(), (lab, d2, []))[2].append(lam)
    for _, _, lab, d2 in sorted(chains, key=lambda c: (c[0], c[1], _cost(c[3], ctx))):
        out.setdefault(d2.key(), (lab, d2, []))
    return [(lab, d, tuple(ls)) for lab, d, ls in out.values()]


def ancilla_seeds(design: Design, ctx: Context) -> list[tuple[str, Design]]:
    """The ancillas of the first blocks of the block step, one design per ancilla state.

    Completing such a design records its FT versions as parts, which later rollouts install instead of building the
    protection again.
    """
    out, seen = [], set()
    for kind in ("Z", "X"):
        try:
            props = proposals(BlockSolver(kind, "weight"), design, (), ctx)
        except Exception:  # ruff: ignore[blind-except, try-except-continue]  (no cover of this type: nothing to seed)
            continue
        for lab, d1 in _kept(props, ctx):
            _, ch = d1.root.blocks[-1]
            k = parts.state_key(ch, design.t)
            if ch.opaque or k in seen:
                continue
            seen.add(k)
            out.append((lab, Design(copy.deepcopy(ch), design.t, [f"part:{lab}"])))
    return out


def _kept(props: list[tuple[str, Design]], ctx: Context) -> list[tuple[str, Design]]:
    """Block proposals worth a chain: the cheapest under each depth weight, then each cover's cheapest."""
    kept: dict[str, tuple[str, Design]] = {}
    for lam in ctx.extreme_lams():
        best = min(props, key=lambda lx: _cost(lx[1], ctx, lam), default=None)
        if best is not None:
            kept.setdefault(best[1].key(), best)
    for lab, d1 in sorted(props, key=lambda lx: _cost(lx[1], ctx)):
        if lab.split("]")[0] not in {x.split("]")[0] for x, _ in kept.values()}:
            kept.setdefault(d1.key(), (lab, d1))
    return list(kept.values())


def _nested_chains(
    design: Design, todo: list[tuple[tuple[int, ...], int]], ctx: Context
) -> list[tuple[float, bool, str, Design]]:
    """Chains that protect the ancillas of the blocks ``todo`` by a nested block of the other type.

    Per ancilla the proposal with the fewest local failures wins, then the cheapest under each depth weight (an ancilla
    that is FT on its own beats a cheaper one that needs a further level). Returns (weighted failures, True, label,
    design) per chain.
    """

    def local(x: Design, scope: tuple[int, ...]) -> float:
        node = x.node(scope)
        try:
            return weighted_failures(check_cached(compile_node(node), node.target, design.t)[2], design.t)
        except Exception:  # ruff: ignore[blind-except]
            return float("inf")

    chains, seen = [], set()
    for lam in ctx.extreme_lams():
        d, labs = design, []
        for path, i in todo:
            kind = "X" if d.node(path).blocks[i][0].kind == "Z" else "Z"
            scope = (*path, i)
            props = proposals(BlockSolver(kind, "weight"), d, scope, ctx)
            if props:
                lab, d = min(props, key=lambda lx, lam=lam, sc=scope: (local(lx[1], sc), _cost(lx[1], ctx, lam)))
                labs.append(lab)
        if not labs or d.key() in seen:
            continue
        seen.add(d.key())
        try:
            chains.append((weighted_failures(evaluate_cached(d).raw, design.t), True, "+".join(labs), d))
        except Exception:  # ruff: ignore[blind-except, try-except-continue]  (a chain that fails to compile is dropped)
            continue
    return chains


def _pair_chains(design: Design, path: tuple[int, ...], i: int, ctx: Context) -> list[tuple[float, bool, str, Design]]:
    """Chains that protect block ``i`` of scope ``path``, as (weighted failures, plain ancilla?, label, design).

    The new block's ancilla is the plain prep or a cancellation-guided prep (``guided``), each with a nested block.
    """

    def other(d: Design, path: tuple[int, ...], i: int) -> str:
        return "X" if d.node(path).blocks[i][0].kind == "Z" else "Z"

    props = proposals(BlockSolver(other(design, path, i), "weight"), design, path, ctx)
    chains = []
    for lab, d1 in _kept(props, ctx):
        j = len(d1.node(path).blocks) - 1
        node = d1.node(path)
        spec, ch = node.blocks[j]
        if ch.opaque or ch.gadgets or ch.blocks:
            continue
        variants = [(lab, d1, True)]
        try:
            full, target = ancilla_states(spec)
            for g, c in enumerate(guided_block_preps(node, j, design.t, k=2)):
                nn = copy.deepcopy(node)
                nn.blocks[j] = (spec, make_node(ch.name, c, full, target, origin="guided"))
                variants.append((f"{lab}+guided{g}", d1.with_node(path, nn, f"guided{g}"), False))
        except Exception:  # ruff: ignore[blind-except, try-except-pass]  (no guided prep: the plain one stays)
            pass
        for lv, dv, plain in variants:
            for l2, d2 in proposals(BlockSolver(other(dv, path, j), "weight"), dv, (*path, j), ctx):
                try:
                    wf = weighted_failures(evaluate_cached(d2).raw, design.t)
                except Exception:  # ruff: ignore[blind-except, try-except-continue]  (a chain that fails is dropped)
                    continue
                chains.append((wf, plain, f"{lv}|{l2}", d2))
    return chains


def complete(
    design: Design,
    ctx: Context,
    *,
    with_circuit: bool,
    polish_budget: float,
    partials: list | None = None,
    quick_budget: float | None = None,
    found: list | None = None,
) -> list[tuple[str, Design, Design]]:
    """FT completions of ``design`` as (label, design, base).

    Structured completions come first (base = itself), then circuit-polished ones (base = the structured design they
    were polished from). ``partials`` gets the partial designs with new blocks, for the tree. ``quick_budget``: rollout
    budget instead of ``Profile.rollout_budget``. ``found`` gets each FT completion as soon as it is verified, so a
    caller that stops this function at its time limit keeps them.
    """

    def keep(items: list[tuple[str, Design]]) -> list[tuple[str, Design]]:
        if found is not None:
            found.extend((lab, x, x) for lab, x in items if evaluate_cached(x).ok)
        return items

    if evaluate_cached(design).ok:
        raw = keep([("as-is", design)])
    else:
        c = dataclasses.replace(ctx, ilp_time=min(ctx.ilp_time, profile(design.t).rollout_ilp_time))
        budget = quick_budget if quick_budget is not None else profile(design.t).rollout_budget
        deadline = None if budget is None else time.time() + budget
        hard = None if budget is None else time.time() + 2 * budget  # lead chains: at most twice the budget
        raw = _finish(design, c, ctx.extreme_lams(), deadline=deadline, on_out=lambda lx, x: keep([(lx, x)]))
        # the block step, repeated on its chains for ``Profile.block_rounds`` rounds
        todo = [(design, "", tuple(c.extreme_lams()), profile(design.t).block_rounds)]
        while todo:
            d0, lab0, lams0, rounds = todo.pop(0)
            for k, (lab1, d1, lams1) in enumerate(block_repairs(d0, c, deadline, nest=bool(lab0))):
                lab = f"{lab0}||{lab1}" if lab0 else lab1
                lams = tuple(x for x in lams1 if x in lams0)
                # the round's lead chain goes on past the deadline, up to twice the budget
                lead = k == 0 and (bool(lab0) or not any("|" in x for x, _ in raw))
                lead = lead and (hard is None or time.time() < hard)
                in_time = deadline is None or time.time() < deadline
                if lams and (in_time or lead):
                    raw += [
                        (f"{lab}+{lx}", x)
                        for lx, x in _finish(
                            d1, c, lams, deadline=deadline, on_out=lambda lx, x, lab=lab: keep([(f"{lab}+{lx}", x)])
                        )
                    ]
                    if rounds > 1:
                        todo.append((d1, lab, lams, rounds - 1))
                if partials is not None:
                    partials.append((lab, d1))
        if not raw:
            label, d = _complete_criger(design, c)
            raw = keep([(label, d)] if d is not None else [])
    # the circuit stage only on the cheapest completion under each λ
    circ_for = {id(x) for x in ctx.best_per_lam(raw, lambda x: metrics(x[1].circuit()), ctx.extreme_lams())}
    seen, structured, opaque = set(), [], []
    for item in raw:
        label, d = item
        t0 = time.time()
        polished = polish_budget > 0
        if polished:
            s, circ = polish.quick(d, ctx, with_circuit=with_circuit and id(item) in circ_for, budget=polish_budget)
        else:  # no polish in the quick rollout
            s, circ = d, []
        tail = "+polish" if polished else ""
        for lab, x in [(label + tail, s), *[(label + "+polish:circuit", y) for y in circ]]:  # base of all: s
            k = x.key()
            if k in seen:
                continue
            seen.add(k)
            try:
                ok = evaluate_cached(x).ok
            except Exception:  # ruff: ignore[blind-except]
                ok = False
            if ok:
                (opaque if x.root.opaque else structured).append((lab, x, s))
        polish_budget = max(0.0, polish_budget - (time.time() - t0))
    return structured + opaque
