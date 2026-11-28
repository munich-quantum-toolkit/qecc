# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Direct tests of parts that a short search on a small code does not reach.

The search jobs run in this process; edge cases of the cover and set-cover helpers, split blocks, circuit conversion
and ancilla pruning.
"""

from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.ft_search import api
from mqt.qecc.circuit_synthesis.ft_search._core import memo, parts, search
from mqt.qecc.circuit_synthesis.ft_search._core.bellred import _safe, prune_ancillas
from mqt.qecc.circuit_synthesis.ft_search._core.blocks import BlockSpec, dual
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit, Op, layering
from mqt.qecc.circuit_synthesis.ft_search._core.complete import (
    _complete_criger,
    _criger_fix,
    _greedy_fix,
)
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context, profile
from mqt.qecc.circuit_synthesis.ft_search._core.cover import (
    anticommuting_element,
    parity_matrix,
    robust_cover_from_failures,
)
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.evaluation import evaluate_cached
from mqt.qecc.circuit_synthesis.ft_search._core.front import DMAX, Front
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import Gadget, build
from mqt.qecc.circuit_synthesis.ft_search._core.pool import kill_pool, process_pool
from mqt.qecc.circuit_synthesis.ft_search._core.setcover import ilp_cover, ls_cover, solve_cover
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import ancilla
from mqt.qecc.circuit_synthesis.ft_search._core.solvers.ancilla import BlockSolver
from mqt.qecc.circuit_synthesis.ft_search._core.solvers.common import circuit_choices, to_cnot_circuit, to_qecc_prep
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    CSSState,
    check_added,
    check_ft,
    check_noiseless,
    state_of_code,
)
from mqt.qecc.circuit_synthesis.ft_search.api import FrontCircuit, synthesize_ft_prep
from mqt.qecc.circuit_synthesis.ft_search.simulation import logical_error_rate
from mqt.qecc.circuit_synthesis.ft_search.synth import candidates

if TYPE_CHECKING:
    from collections.abc import Callable


def _steane_design() -> tuple[Design, CSSState]:
    state = state_of_code(CSSCode.from_code_name("Steane"))
    prep = candidates(state.sx, "greedy", 1)[0]
    return Design(make_node("D", prep, state, origin="prep0"), 1), state


def test_search_jobs_in_process() -> None:
    """Seed, part and deep node jobs on a Steane design: an ancilla seed, its FT versions, FT completions."""
    design, state = _steane_design()
    ctx, prof = Context(), profile(1)
    seeds = search.seed_job((design, ctx, prof))
    assert not seeds["errors"]
    assert seeds["seeds"]
    for _, seed in seeds["seeds"]:
        part = search.part_job((seed, ctx, prof))
        assert not part["errors"]
        key = parts.state_key(seed.root, seed.t)
        assert any(k == key for k, _ in parts.snapshot())  # its FT versions are known parts of its state
    deep = search.node_job((design, ctx, prof, True))
    assert not deep["errors"]
    assert deep["completions"]
    for _, d, _ in deep["completions"]:
        check_noiseless(d.circuit(), state)
        assert check_ft(d.circuit(), state, 1).ok


def test_finishers() -> None:
    """Criger's ILP and the greedy each make the Steane design FT."""
    design, _ = _steane_design()
    ctx = Context()
    label, d = _complete_criger(design, ctx)
    assert d is not None
    assert evaluate_cached(d).ok, label
    assert _criger_fix(design, ctx)[1] is not None
    fixed = _greedy_fix(design, ctx)
    assert fixed is not None
    assert evaluate_cached(fixed).ok


def test_job_error_is_reported() -> None:
    """A job that fails reports the error and its time instead of raising."""
    out = search.node_job((None, Context(), profile(1)))
    assert out["errors"]
    assert out["seconds"] >= 0


def test_anticommuting_element() -> None:
    """The element has odd overlap with the target; target 0 gives 0."""
    rng = np.random.default_rng(0)
    assert anticommuting_element(0, 5, rng) == 0
    for target in (1, 6, 0b10110):
        assert all((anticommuting_element(target, 5, rng) & target).bit_count() % 2 == 1 for _ in range(20))


def test_parity_matrix_without_bitwise_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shift-and-xor parity (numpy without ``bitwise_count``) gives the same matrix."""
    rng = np.random.default_rng(1)
    tg, el = rng.integers(0, 1 << 40, 50), rng.integers(0, 1 << 40, 30)
    expected = parity_matrix(tg, el)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert np.array_equal(parity_matrix(tg, el), expected)
    assert np.array_equal(expected, [[(int(a) & int(b)).bit_count() % 2 for b in el] for a in tg])


def test_robust_cover_without_failures() -> None:
    """No failing combination, or none that stays dangerous, needs no element."""
    basis = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    assert robust_cover_from_failures(basis, [], 1) == []
    assert robust_cover_from_failures(basis, [(1, 1)], 1) == []  # one fault with a weight-1 leader: harmless


def test_memo_drops_the_oldest_entry() -> None:
    """Beyond its limit a memo table forgets its oldest entry."""
    for i in range(3):
        memo.put("test", str(i), i, limit=2)
    assert memo.get("test", "0") is None
    assert memo.get("test", "2") == 2


def test_ls_cover_edge_cases() -> None:
    """No rows: empty cover; an uncoverable row: None; a sparse matrix with a conflict: a valid cover."""
    assert ls_cover(np.zeros((0, 3), dtype=np.uint8), [1, 1, 1]) == []
    assert ls_cover(np.array([[1, 0], [0, 0]], dtype=np.uint8), [1, 1]) is None
    n = 40
    a = np.zeros((n, n + 1), dtype=np.uint8)
    a[np.arange(n), np.arange(n)] = 1  # sparse: each row by its own column ...
    a[:, n] = 1  # ... or by the expensive last column
    cost = [1.0] * n + [100.0]
    sel = ls_cover(a, cost, conflicts=[(0, 1)], time_limit=1.0, patience=50)
    assert sel is not None
    assert a[:, sel].any(axis=1).all()
    assert not {0, 1} <= set(sel)


def test_ls_cover_conflict_with_the_only_column_of_a_row() -> None:
    """The greedy takes column 0 first, which conflicts with column 2, the only column of the last rows."""
    a = np.zeros((1001, 4), dtype=np.uint8)
    a[:501, [0, 1, 3]] = 1
    a[501:, 2] = 1
    sel = ls_cover(a, [1.0, 2.0, 1.0, 3.0], conflicts=[(0, 2)], time_limit=1.0, patience=50)
    assert sel == [1, 2]


def test_ilp_cover_edge_cases() -> None:
    """The ILP respects a conflict, finds no cover for an uncoverable row, and large problems go to local search."""
    a = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    sel = ilp_cover(a, [1, 1, 1], conflicts=[(0, 1)])
    assert sel is not None
    assert not {0, 1} <= set(sel)
    assert a[:, sel].any(axis=1).all()
    assert ilp_cover(np.array([[1, 0], [0, 0]], dtype=np.uint8), [1, 1]) is None
    big = np.eye(1001, dtype=np.uint8)
    assert solve_cover(big, np.ones(1001), time_limit=1.0) == list(range(1001))


def test_harmless_back_of_split_blocks() -> None:
    """A split block adds the pairs of ancilla qubits on one data qubit to the ancilla checks."""
    split = BlockSpec("Z", np.array([[1, 1, 0]]), gamma=(0, 1, 1), htilde=np.array([[1, 1, 0]]))
    assert np.array_equal(split.harmless_back(), [[1, 1, 0], [0, 1, 1]])
    plain = BlockSpec("Z", np.array([[1, 1, 0]]))
    assert np.array_equal(plain.harmless_back(), [[1, 1]])
    single = BlockSpec("Z", np.array([[1, 1]]), gamma=(0, 1), htilde=np.array([[1, 1]]))
    assert np.array_equal(single.harmless_back(), [[1, 1]])


def test_from_stim_edge_cases() -> None:
    """Annotations are dropped, measured qubits are not data by default, other gates raise."""
    c = Circuit.from_stim(stim.Circuit("R 0 1\nTICK\nCX 0 1\nI 0\nM 1"))
    assert c.data == [0]
    assert [op.name for op in c.ops] == ["R", "R", "CX", "M"]
    with pytest.raises(ValueError, match="unsupported instruction S"):
        Circuit.from_stim(stim.Circuit("S 0"))
    with pytest.raises(ValueError, match="unsupported instruction X"):
        Circuit.from_stim(stim.Circuit("R 0\nX 0"))
    with pytest.raises(ValueError, match="inverted"):
        Circuit.from_stim(stim.Circuit("R 0\nM !0"))
    untouched = Circuit.from_stim(stim.Circuit("H 0"), data=[0, 1, 2])  # qubits 1 and 2 only get the implicit reset
    assert untouched.num_qubits == 3
    assert [op.name for op in untouched.ops] == ["R", "R", "R", "H"]


def test_to_qecc_prep() -> None:
    """A CNOT preparation circuit converts to MQT QECC's form; other gates raise."""
    design, _ = _steane_design()
    sp = to_qecc_prep(design.root.base, 1)
    assert sp.num_qubits == 7
    with pytest.raises(ValueError, match="unexpected M"):
        to_qecc_prep(Circuit([Op("R", (0,)), Op("M", (0,))], 1, [0]), 1)


def test_cnot_circuit_initialization_in_order() -> None:
    """Resets and Hadamards fold in order; an initialization after a CNOT on its qubit raises."""
    c = to_cnot_circuit(Circuit([Op("R", (0, 1)), Op("H", (0,)), Op("H", (0,)), Op("H", (1,)), Op("CX", (1, 0))], 2))
    assert str(c.to_stim_circuit()).split("\n")[:2] == ["R 0", "RX 1"]
    with pytest.raises(ValueError, match="unexpected H"):
        to_cnot_circuit(Circuit([Op("R", (0, 1)), Op("CX", (0, 1)), Op("H", (0,))], 2))


def test_make_node_keeps_the_order_of_preps_it_cannot_layer() -> None:
    """A prep with a CZ, or with a Hadamard after a CNOT on its qubit, is not reordered."""
    state = state_of_code(CSSCode.from_code_name("Steane"))
    prep = candidates(state.sx, "greedy", 1)[0]
    j = next(i for i, op in enumerate(prep.ops) if op.name == "CX")
    c, t = prep.ops[j].qubits
    for ops in (
        [*prep.ops[:j], Op("H", (t,)), Op("CZ", (c, t)), Op("H", (t,)), *prep.ops[j + 1 :]],
        [*prep.ops, Op("H", (t,)), Op("CX", (c, t)), Op("CX", (c, t)), Op("H", (t,))],
    ):
        circ = Circuit(ops, prep.num_qubits, list(prep.data))
        check_noiseless(circ, state)
        node = make_node("D", circ, state)
        assert not node.layered
        assert node.base.ops == ops
    assert make_node("D", prep, state).layered


def test_block_cover_keeps_covers_when_a_later_sample_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A perturbed cover sample without a solution leaves the covers found before it."""
    design, _ = _steane_design()
    real = ancilla.robust_cover_from_failures

    def flaky(*args: object, seed: int = 0, **kwargs: object) -> list[int]:
        if seed:
            msg = "cover failed"
            raise RuntimeError(msg)
        return real(*args, seed=seed, **kwargs)  # ty: ignore[invalid-argument-type]  (passed through)

    monkeypatch.setattr(ancilla, "robust_cover_from_failures", flaky)
    assert list(BlockSolver("Z", samples=2).propose(design, (), Context()))


def test_prune_ancillas_removes_a_useless_check() -> None:
    """An ancilla that the check does not need is removed with its detector."""
    design, state = _steane_design()
    base = design.root.base
    n = base.num_qubits
    ops = [*base.ops, Op("R", (n,)), Op("CX", (0, n)), Op("CX", (1, n)), Op("M", (n,))]
    circ = Circuit(ops, n + 1, list(base.data), [[0]])
    pruned, removed = prune_ancillas(circ, lambda c: check_ft(c, state, 0).ok)
    assert removed == [n]
    assert pruned.num_measurements == 0


def test_circuit_choices_skip_a_failing_synthesizer() -> None:
    """A synthesis mode whose tool fails is skipped; the other modes still give circuits."""
    _, state = _steane_design()
    ctx = Context(synth=[("broken", 1), ("greedy", 2)], synthesizers={"broken": _broken}, pool=3)
    out = circuit_choices(state.sx, state, 1, ctx)
    assert out
    assert all(label.startswith("greedy") for label, _ in out)


def _broken(_h: np.ndarray, _k: int) -> list:
    msg = "tool failed"
    raise RuntimeError(msg)


def test_build_rejects_inconsistent_gadgets() -> None:
    """Inconsistent gadgets are errors.

    A gadget of a missing block, a block without all its ancilla gadgets, a flag shared by a Z and an X gadget, and
    decreasing slots.
    """
    design, _ = _steane_design()
    base = design.root.base
    end = len(base.ops)
    with pytest.raises(ValueError, match="block that does not exist"):
        build(base, [Gadget("Z", end, [0, 1], block=0)])
    spec = BlockSpec("Z", np.array([[1, 1, 0, 0, 0, 0, 0]]))
    two = Circuit([Op("R", (0,)), Op("R", (1,)), Op("CX", (0, 1))], 2, [0, 1])
    with pytest.raises(ValueError, match="gadgets for ancilla qubits"):
        build(base, [Gadget("Z", end, [0], block=0, anc=0)], [(spec, two)])
    with pytest.raises(ValueError, match="different kinds"):
        build(base, [Gadget("Z", end, [0, 1, 2], [(0, 2, 0)]), Gadget("X", end, [0, 1, 2], [(0, 2, 0)])])
    with pytest.raises(ValueError, match="non-decreasing"):
        build(base, [Gadget("Z", end, [0, 1], slots=[end, 0])])


def test_check_added_declines() -> None:
    """``check_added`` declines (None) where it cannot apply.

    Without new checks, without tags (a circuit not made by ``gauge.build``), with repeated tags and with a detector on
    old records.
    """
    design, state = _steane_design()
    base = design.root.base
    assert base.tags is None
    assert check_added(base, state, 1, base, check_ft(base, state, 1, collect=True)) is None
    assert base == Circuit(base.ops, base.num_qubits, base.data, base.detectors)  # tags are not compared
    end = len(base.ops)
    old, _ = build(base, [Gadget("Z", end, [0, 1, 2, 3])])
    report = check_ft(old, state, 1, collect=True)
    assert check_added(old, state, 1, old, report) is None  # nothing added
    new, _ = build(base, [Gadget("Z", end, [0, 1, 2, 3]), Gadget("Z", end, [2, 3, 4, 5])])
    assert check_added(new, state, 1, old, report) is not None
    assert new.tags is not None
    assert new.detectors is not None
    repeated = Circuit(new.ops, new.num_qubits, new.data, new.detectors)
    repeated.tags = [new.tags[0]] * len(new.tags)
    assert check_added(repeated, state, 1, old, report) is None
    mixed = Circuit(new.ops, new.num_qubits, new.data, [*new.detectors, [0, 1]])
    mixed.tags = list(new.tags)
    assert check_added(mixed, state, 1, old, report) is None


def test_small_error_paths() -> None:
    """Small error paths and edge cases.

    Layering of a measured circuit, a detector that is always 1, a grouped block's ancilla map, the dual of nothing, a
    failing check counted as not OK, and the lazy failure sequence.
    """
    measured = Circuit([Op("R", (0,)), Op("M", (0,))], 1, [])
    with pytest.raises(ValueError, match="layering supports"):
        layering(measured)
    always_one = Circuit([Op("R", (0,)), Op("X", (0,)), Op("M", (0,))], 1, [], [[0]])
    with pytest.raises(AssertionError, match="have value 1"):
        check_noiseless(always_one, state_of_code(CSSCode.from_code_name("Steane")))
    grouped = BlockSpec("Z", np.array([[1, 1, 1, 1]]), groups=((0, 1), (2, 3)), htilde=np.array([[1, 1]]))
    with pytest.raises(ValueError, match="grouped block"):
        _ = grouped.anc_data
    assert np.array_equal(dual(np.zeros((1, 3), dtype=np.uint8)), np.eye(3))

    def fails(_c: Circuit) -> bool:
        msg = "not deterministic"
        raise ValueError(msg)

    assert not _safe(fails, measured)
    design, state = _steane_design()
    report = check_ft(design.root.base, state, 1, collect=True)
    assert len(report.failing_reps) == len(list(report.failing_reps)) > 0
    assert report.failing_reps == list(report.failing_reps)


def test_no_start_circuit() -> None:
    """A synthesis mode that returns no circuit leaves no start circuit: a ValueError."""
    code = CSSCode.from_code_name("Steane")
    with pytest.raises(ValueError, match="no start circuit"):
        synthesize_ft_prep(code, synth=[("nothing", 1)], synthesizers={"nothing": _nothing}, seconds=1, workers=1)


def _nothing(_h: np.ndarray, _k: int) -> list:
    return []


def test_logical_error_rate_of_a_front_circuit() -> None:
    """A ``FrontCircuit`` is simulated like its stim circuit."""
    code = CSSCode.from_code_name("Steane")
    design, _ = _steane_design()
    fixed = _greedy_fix(design, Context())
    assert fixed is not None
    circ = fixed.circuit().to_stim()
    front = FrontCircuit(circ, 0, 0, 0, 0, 0)
    assert logical_error_rate(front, code, 0.0, shots=500, min_errors=0)[2] == 0


def test_reward_targets_of_a_very_deep_front() -> None:
    """A front deeper than the deepest target depth still gives a target."""
    front = Front()
    front.add(100, DMAX + 6)
    assert front.targets(1) == [DMAX]


def test_kill_pool_after_shutdown() -> None:
    """``kill_pool`` on a pool that is already shut down does nothing."""
    pool = process_pool(1)
    pool.shutdown()
    kill_pool(pool)


def test_start_circuits_per_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``synth=[(mode, k)]`` gives up to ``k`` start circuits of that mode."""
    seen: list = []

    def no_search(start: list, *_args: object, **_kwargs: object) -> list:
        seen.extend(start)
        return []

    monkeypatch.setattr(api, "search", no_search)
    with pytest.warns(UserWarning, match="no fault-tolerant circuit"):
        synthesize_ft_prep(CSSCode.from_code_name("Steane"), synth=[("random", 3)], seconds=1, workers=1)
    assert len(seen) == 3


def test_search_survives_a_broken_pool_on_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pool that broke since the last wait is replaced, and the search still returns its front."""
    real = search.process_pool
    broke: list = []

    def flaky_pool(workers: int) -> object:
        pool = real(workers)
        submit = pool.submit

        def flaky_submit(fn: Callable, *args: object) -> object:
            if not broke:
                broke.append(True)
                raise BrokenProcessPool
            return submit(fn, *args)

        pool.submit = flaky_submit  # ty: ignore[invalid-assignment]  (a pool that breaks once)
        return pool

    monkeypatch.setattr(search, "process_pool", flaky_pool)
    assert synthesize_ft_prep(CSSCode.from_code_name("Steane"), seconds=10, workers=1)
    assert broke


def test_a_crash_returns_the_front_so_far(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """An unexpected error in the search loop is logged; the front found so far comes back."""

    def crash(*_args: object, **_kwargs: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(search, "wait", crash)
    with pytest.warns(UserWarning, match="no fault-tolerant circuit"):
        assert synthesize_ft_prep(CSSCode.from_code_name("Steane"), seconds=60, workers=1) == []
    assert "search failed" in caplog.text
