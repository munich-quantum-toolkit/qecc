# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for synthesizing encoding circuits for CSS codes."""

from __future__ import annotations

import functools
import logging
import operator
from typing import TYPE_CHECKING

import numpy as np
import z3
from ldpc import mod2

from ..codes import InvalidCSSCodeError
from .synthesis_utils import build_css_circuit_from_cnot_list, heuristic_gaussian_elimination, optimal_elimination

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from qiskit.circuit import QuantumCircuit

    from ..codes.css_code import CSSCode

logger = logging.getLogger(__name__)


def heuristic_encoding_circuit(
    code: CSSCode, optimize_depth: bool = True, balance_checks: bool = False
) -> QuantumCircuit:
    """Synthesize an encoding circuit for the given CSS code using a heuristic greedy search.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        optimize_depth: Whether to optimize the depth of the circuit.
        balance_checks: Whether to balance the entries of the stabilizer matrix via row operations.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting encoding circuit synthesis.")

    checks, logicals, use_x_checks = _get_matrix_with_fewest_checks(code)
    n_checks = checks.shape[0]

    if balance_checks:
        _balance_matrix(logicals)

    checks, cnots = heuristic_gaussian_elimination(
        np.vstack((checks, logicals)),
        parallel_elimination=optimize_depth,
    )

    # after reduction there still might be some overlap between initialized qubits and encoding qubits, we simply perform CNOTs to correct this
    encoding_qubits = np.where(checks[n_checks:, :].sum(axis=0) != 0)[0]
    initialization_qubits = np.where(checks[:n_checks, :].sum(axis=0) != 0)[0]
    # remove encoding qubits from initialization qubits
    initialization_qubits = np.setdiff1d(initialization_qubits, encoding_qubits)
    rows = []  # type: list[int]
    qubit_to_row = {}
    for qubit in initialization_qubits:
        cand = np.where(checks[:n_checks, qubit] == 1)[0]
        np.setdiff1d(cand, np.array(rows))
        rows.append(cand[0])
        qubit_to_row[qubit] = cand[0]

    for init_qubit in initialization_qubits:
        for encoding_qubit in encoding_qubits:
            row = qubit_to_row[init_qubit]
            if checks[row, encoding_qubit] == 1:
                cnots.append((init_qubit, encoding_qubit))
                checks[row, encoding_qubit] = 0

    cnots = cnots[::-1]

    return _build_css_encoder_from_cnot_list(n_checks, checks, cnots, use_x_checks)


def gate_optimal_encoding_circuit(
    code: CSSCode,
    min_gates: int = 1,
    max_gates: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> QuantumCircuit:
    """Synthesize an encoding circuit for the given CSS code using the minimal number of gates.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        min_gates: The minimum number of gates to use in the circuit.
        max_gates: The maximum number of gates to use in the circuit.
        min_timeout: The minimum time to spend on the synthesis.
        max_timeout: The maximum time to spend on the synthesis.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting optimal encoding circuit synthesis.")
    checks, logicals, use_x_checks = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.shape[0]
    checks_and_logicals = np.vstack((checks, logicals))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(checks.shape[0], checks.shape[0] + logicals.shape[0])),
        rank=rank,
    )

    res = optimal_elimination(
        checks_and_logicals,
        termination_criteria,
        "column_ops",
        min_param=min_gates,
        max_param=max_gates,
        min_timeout=min_timeout,
        max_timeout=max_timeout,
    )
    if res is None:
        return None
    reduced_checks_and_logicals, cnots = res
    cnots = cnots[::-1]

    return _build_css_encoder_from_cnot_list(n_checks, reduced_checks_and_logicals, cnots, use_x_checks)


def depth_optimal_encoding_circuit(
    code: CSSCode,
    min_depth: int = 1,
    max_depth: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> QuantumCircuit:
    """Synthesize an encoding circuit for the given CSS code using minimal depth.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        min_depth: The minimum number of gates to use in the circuit.
        max_depth: The maximum number of gates to use in the circuit.
        min_timeout: The minimum time to spend on the synthesis.
        max_timeout: The maximum time to spend on the synthesis.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting optimal encoding circuit synthesis.")
    checks, logicals, use_x_checks = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.shape[0]
    checks_and_logicals = np.vstack((checks, logicals))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(checks.shape[0], checks.shape[0] + logicals.shape[0])),
        rank=rank,
    )
    res = optimal_elimination(
        checks_and_logicals,
        termination_criteria,
        "parallel_ops",
        min_param=min_depth,
        max_param=max_depth,
        min_timeout=min_timeout,
        max_timeout=max_timeout,
    )
    if res is None:
        return None
    reduced_checks_and_logicals, cnots = res
    cnots = cnots[::-1]

    return _build_css_encoder_from_cnot_list(n_checks, reduced_checks_and_logicals, cnots, use_x_checks)


def _get_matrix_with_fewest_checks(code: CSSCode) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8], bool]:
    """Return the stabilizer matrix with the fewest checks, the corresponding logicals and a bool indicating whether X- or Z-checks have been returned."""
    if code.Hx is None or code.Hz is None:
        msg = "The code must have both X and Z stabilizers defined."
        raise InvalidCSSCodeError(msg)

    use_x_checks = code.Hx.shape[0] < code.Hz.shape[0]
    checks = code.Hx if use_x_checks else code.Hz
    logicals = code.Lx if use_x_checks else code.Lz
    return checks, logicals, use_x_checks


def _final_matrix_constraint_partially_full_reduction(
    columns: npt.NDArray[z3.BoolRef | bool], full_reduction_rows: list[int], rank: int
) -> z3.BoolRef:
    assert len(columns.shape) == 3

    partial_reduction_rows = list(set(range(columns.shape[1])) - set(full_reduction_rows))

    # assert that the partial_reduction_rows are partially reduced, i.e. there are at least columns.shape[2] - (columns.shape[1] - len(full_reduction_rows)) non-zero columns
    partially_reduced = z3.PbEq(
        [(z3.Not(z3.Or(list(columns[-1, partial_reduction_rows, col]))), 1) for col in range(columns.shape[2])],
        columns.shape[2] - (rank - len(full_reduction_rows)),
    )

    # assert that there is no overlap between the full_reduction_rows and the partial_reduction_rows
    overlap_constraints = [True]
    for col in range(columns.shape[2]):
        has_entry_partial = z3.Or(list(columns[-1, partial_reduction_rows, col]))
        has_entry_full = z3.Or(list(columns[-1, full_reduction_rows, col]))
        overlap_constraints.append(z3.Not(z3.And(has_entry_partial, has_entry_full)))

    # assert that the full_reduction_rows are fully reduced
    fully_reduced = z3.PbEq(
        [
            (z3.PbEq([(columns[-1, row, col], 1) for col in range(columns.shape[2])], 1), 1)
            for row in full_reduction_rows
        ],
        len(full_reduction_rows),
    )

    return z3.And(fully_reduced, partially_reduced, z3.And(overlap_constraints))


def _build_css_encoder_from_cnot_list(
    n_checks: int, checks_and_logicals: npt.NDArray[np.int8], cnots: list[tuple[int, int]], use_x_checks: bool
) -> tuple[QuantumCircuit, list[int]]:
    encoding_qubits = np.where(checks_and_logicals[n_checks:, :].sum(axis=0) != 0)[0]
    if use_x_checks:
        hadamards = np.where(checks_and_logicals[:n_checks, :].sum(axis=0) != 0)[0]
    else:
        hadamards = np.where(checks_and_logicals[:n_checks, :].sum(axis=0) == 0)[0]
        cnots = [(j, i) for i, j in cnots]

    hadamards = np.setdiff1d(hadamards, encoding_qubits)
    circ = build_css_circuit_from_cnot_list(checks_and_logicals.shape[1], cnots, list(hadamards))
    return circ, encoding_qubits


def _balance_matrix(m: npt.NDArray[np.int8]) -> None:
    """Balance the columns of the matrix.

    Try to balance the number of 1's in each column via row operations without increasing the row-weight.
    """
    variance = np.var(m.sum(axis=0))
    reduced = False

    while not reduced:
        reduced = True
        # compute row operations that do not increase the row-weights
        row_ops = []
        for i, row_1 in enumerate(m):
            for j, row_2 in enumerate(m):
                if i == j:
                    continue
                s = (row_1 + row_2) % 2
                if s.sum() > row_1.sum() or s.sum() > row_2.sum():
                    continue
                # compute associated column weights
                m[j] = s  # noqa: B909

                new_variance = np.var(m.sum(axis=0))
                if new_variance < variance:
                    row_ops.append((i, j, new_variance))

                m[j] = row_2  # noqa: B909
        # sort by lowest variance
        row_ops.sort(key=operator.itemgetter(2))
        # apply best row operation
        if row_ops:
            i, j = row_ops[0][:2]
            m[i] = (m[i] + m[j]) % 2
            reduced = False
            variance = row_ops[0][2]
