# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Memoized exact checks and solver proposals, and the evaluation of a design that records the parts it finds."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

from . import memo
from .design import Design, compile_node, evaluate, scope_name
from .verify import check_ft, check_noiseless, target_key, weighted_failures

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Protocol

    from .circuit import Circuit
    from .config import Context
    from .design import Evaluation, Node
    from .verify import CSSState

    class Solver(Protocol):
        """What ``proposals`` needs of a solver."""

        name: str

        @property
        def ctx_deps(self) -> tuple[str, ...]:
            """The ``Context`` fields its proposals depend on."""

        def propose(self, design: Design, path: tuple[int, ...], ctx: Context) -> Iterable[tuple[str, Node]]:
            """Yield (label, new node for the scope at ``path``)."""


def _solver_key(solver: Solver, design: Design, path: tuple[int, ...], ctx: Context) -> str:
    """Memo key of a solver's proposals: solver class and settings, ``ctx_deps`` fields, node and t."""
    cache_extra = getattr(solver, "cache_extra", None)
    extra = cache_extra(design, path) if cache_extra is not None else ()
    settings = sorted((k, repr(v)) for k, v in vars(solver).items())
    ctxk = tuple((k, repr(getattr(ctx, k))) for k in solver.ctx_deps)
    return memo.digest(type(solver).__name__, settings, ctxk, design.t, extra, design.node(path))


def _stored(circ: Circuit, state: CSSState, t: int) -> tuple[str, tuple | None]:
    """The memo key of the check of ``circ`` and its stored result (``check_cached``), if any."""
    key = memo.digest(str(circ.to_stim()), *target_key(state), t)
    return key, cast("tuple | None", memo.get("check", key))


def ft_cached(circ: Circuit, state: CSSState, t: int) -> bool:
    """Whether the compiled circuit is FT: the yes / no form of ``check_cached``, stopping at the first failure.

    A failure is stored only as a marker ("okc"), so a later ``check_cached`` still computes the counts.
    """
    key, hit = _stored(circ, state, t)
    if hit is not None:
        return hit[0] is True
    if memo.get("okc", key) is not None:
        return False
    try:
        check_noiseless(circ, state)
    except Exception:  # ruff: ignore[blind-except]
        return False
    ok = check_ft(circ, state, t, first_fail=True).ok
    if ok:
        memo.put("check", key, (True, dict.fromkeys(range(1, t + 1), 0), dict.fromkeys(range(1, t + 1), 0)))
    else:
        memo.put("okc", key, False)
    return ok


def check_cached(circ: Circuit, state: CSSState, t: int) -> tuple:
    """Noiseless and exact FT check of a compiled circuit, memoized: (ok, failures, raw failures).

    Raises:
        AssertionError: The circuit is not deterministic or prepares the wrong state.
    """
    key, hit = _stored(circ, state, t)
    if hit is None:
        try:
            check_noiseless(circ, state)
        except Exception as e:  # ruff: ignore[blind-except]
            hit = ("error", f"{type(e).__name__}: {str(e)[:300]}")
        else:
            rep = check_ft(circ, state, t)
            hit = (rep.ok, dict(rep.failures), dict(rep.raw_failures))
        memo.put("check", key, hit)
    if hit[0] == "error":
        raise AssertionError(hit[1])
    return hit


def score_cached(circ: Circuit, state: CSSState, t: int) -> float | None:
    """Weighted failure count (0 = FT, ``verify.weighted_failures``), None if not deterministic / wrong state."""
    try:
        return weighted_failures(check_cached(circ, state, t)[2], t)
    except (AssertionError, ValueError):
        return None


def proposals(solver: Solver, design: Design, path: tuple[int, ...], ctx: Context) -> list[tuple[str, Design]]:
    """Proposals of one solver on one scope as new designs, labelled ``<scope>:<solver label>``.

    The memo keeps the solver's (label, new node) pairs and applies them to the current design.
    """
    key = _solver_key(solver, design, path, ctx)
    hit = cast("list[tuple[str, Node]] | None", memo.get("prop", key))
    if hit is None:
        hit = list(solver.propose(design, path, ctx))
        memo.put("prop", key, hit)
    return [
        (f"{scope_name(path)}:{label}", design.with_node(path, copy.deepcopy(node), f"{scope_name(path)}:{label}"))
        for label, node in hit
    ]


def evaluate_cached(design: Design) -> Evaluation:
    """Exact evaluation with the memoized check; every subtree FT for its own target becomes a part (``parts.add``)."""
    from . import parts  # ruff: ignore[import-outside-top-level]  (parts imports this module)

    ev = evaluate(design, check_cached)
    for path, n in design.nodes():
        if (n.opaque and path) or (not path and not ev.ok) or (path and path in ev.open_scopes):
            continue
        if path and ev.ok and not check_cached(compile_node(n), n.target, design.t)[0]:
            continue  # FT as a whole, but this ancilla is not FT on its own (its faults are caught elsewhere)
        parts.add(n, design.t)
    return ev
