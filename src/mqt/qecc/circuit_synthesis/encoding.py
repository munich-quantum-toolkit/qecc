# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for synthesizing encoding circuits for CSS codes."""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim
import z3

from ..codes import CSSCode
from ..codes.core.pauli import CheckMatrix, StabilizerTableau, complete_stabilizer_tableau_with_destabilizers
from .circuits import CliffordIsometry, CNOTCircuit
from .operations import CNOT
from .synthesis import SynthesisConfig, synthesize_cnot, synthesize_non_css
from .synthesis_utils import build_css_encoder_from_cnot_list, optimal_elimination
from .transvection import lexicographical_compare_np, score_symplectic

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ..codes import StabilizerCode


logger = logging.getLogger(__name__)


def depth_optimal_encoding_circuit_non_css(
    code: StabilizerCode,
    max_depth: int,
    max_two_qubit_gates: int | None = None,
    *,
    exact_two_qubit_count: bool = False,
) -> CliffordIsometry | str:
    """Synthesize an encoding circuit for a stabilizer code using Z3 SMT solver.

    This function searches for a reduction circuit that maps the target code tableau
    to a terminal identity-isometry tableau, allowing stabilizer row additions, physical
    qubit permutation, and X- or Z-type terminal ancilla pivots. The returned circuit
    is the inverse reduction with resets on ancilla qubits.

    The solver enforces the following constraints:
    - Each depth layer uses gates from {I, H, S, SQRT_X, CX, CZ}
    - Every qubit has exactly one gate role per layer
    - Terminal condition: stabilizers have exactly m pivot half-columns
    - Logical qubits map to non-pivot physical qubits with canonical X/Z pairs
    - Optional two-qubit gate budget (equality or inequality depending on exact_two_qubit_count)

    Args:
        code: The stabilizer code to synthesize an encoding circuit for.
        max_depth: Maximum circuit depth to search within.
        max_two_qubit_gates: Optional constraint on the total number of two-qubit gates.
            If None, no gate count constraint is applied.
        exact_two_qubit_count: If True and max_two_qubit_gates is set, the solver enforces
            exactly max_two_qubit_gates two-qubit gates. If False, the solver enforces at
            most max_two_qubit_gates.

    Returns:
        A CliffordIsometry representing the synthesized encoding circuit if a solution is found.
        Returns a string error message if the solver fails:
        - "UNSAT": No solution exists within the given constraints
        - "UNKNOWN: <reason>": Solver could not determine satisfiability
        - "MODEL_ERROR: <detail>": Solution found but model extraction failed

    Raises:
        ValueError: If the code has an incorrect number of independent stabilizers or if
            the stabilizer tableau is malformed.
    """
    n = code.n
    k = code.k
    m = n - k

    assert code.x_logicals is not None
    assert code.z_logicals is not None

    stabilizers = code.symplectic.astype(int)
    x_logicals = code.x_logicals.tableau.matrix.astype(int)
    z_logicals = code.z_logicals.tableau.matrix.astype(int)

    stabilizer_rank = mod2.rank(stabilizers)
    if stabilizer_rank != m:
        if stabilizer_rank < m:
            msg = f"Expected {m} independent stabilizers, but rank is only {stabilizer_rank}."
            raise ValueError(msg)
        independent_stabilizers = np.empty((0, stabilizers.shape[1]), dtype=int)
        for row in stabilizers:
            candidate = np.vstack([independent_stabilizers, row])
            if mod2.rank(candidate) > independent_stabilizers.shape[0]:
                independent_stabilizers = candidate
            if independent_stabilizers.shape[0] == m:
                break
        stabilizers = independent_stabilizers

    row_count = 2 * k + m

    initial_x = np.vstack([
        x_logicals[:, :n],
        z_logicals[:, :n],
        stabilizers[:, :n],
    ])
    initial_z = np.vstack([
        x_logicals[:, n:],
        z_logicals[:, n:],
        stabilizers[:, n:],
    ])

    solver = z3.Solver()

    # ---------------------------------------------------------------------
    # Small Z3 helpers
    # ---------------------------------------------------------------------

    def z3_or(items: list[z3.BoolRef]) -> z3.BoolRef:
        return z3.Or(*items) if items else z3.BoolVal(False)

    def add_cardinality_eq(items: list[z3.BoolRef], value: int) -> None:
        solver.add(z3.Sum([z3.If(lit, 1, 0) for lit in items]) == value)

    def add_cardinality_le(items: list[z3.BoolRef], value: int) -> None:
        solver.add(z3.Sum([z3.If(lit, 1, 0) for lit in items]) <= value)

    def add_exactly_one(items: list[z3.BoolRef]) -> None:
        add_cardinality_eq(items, 1)

    def model_bool(model: z3.ModelRef, lit: z3.BoolRef) -> bool:
        return bool(z3.is_true(model.eval(lit, model_completion=True)))

    # ---------------------------------------------------------------------
    # Tableau variables
    # ---------------------------------------------------------------------

    tableau_x: list[list[list[z3.BoolRef]]] = [
        [[z3.Bool(f"x_{depth}_{row}_{q}") for q in range(n)] for row in range(row_count)]
        for depth in range(max_depth + 1)
    ]

    tableau_z: list[list[list[z3.BoolRef]]] = [
        [[z3.Bool(f"z_{depth}_{row}_{q}") for q in range(n)] for row in range(row_count)]
        for depth in range(max_depth + 1)
    ]

    for row in range(row_count):
        for q in range(n):
            solver.add(tableau_x[0][row][q] == bool(initial_x[row, q]))
            solver.add(tableau_z[0][row][q] == bool(initial_z[row, q]))

    # ---------------------------------------------------------------------
    # Gate variables
    # ---------------------------------------------------------------------

    idle: list[list[z3.BoolRef]] = [[z3.Bool(f"idle_{depth}_{q}") for q in range(n)] for depth in range(max_depth)]

    h_gate: list[list[z3.BoolRef]] = [[z3.Bool(f"h_{depth}_{q}") for q in range(n)] for depth in range(max_depth)]

    s_gate: list[list[z3.BoolRef]] = [[z3.Bool(f"s_{depth}_{q}") for q in range(n)] for depth in range(max_depth)]

    sqrt_x_gate: list[list[z3.BoolRef]] = [
        [z3.Bool(f"sqrt_x_{depth}_{q}") for q in range(n)] for depth in range(max_depth)
    ]

    cx_gate: list[dict[tuple[int, int], z3.BoolRef]] = []
    for depth in range(max_depth):
        cx_layer: dict[tuple[int, int], z3.BoolRef] = {}
        for control in range(n):
            for target in range(n):
                if control != target:
                    cx_layer[control, target] = z3.Bool(f"cx_{depth}_{control}_{target}")
        cx_gate.append(cx_layer)

    cz_gate: list[dict[tuple[int, int], z3.BoolRef]] = []
    for depth in range(max_depth):
        cz_layer: dict[tuple[int, int], z3.BoolRef] = {}
        for q1 in range(n):
            for q2 in range(q1 + 1, n):
                cz_layer[q1, q2] = z3.Bool(f"cz_{depth}_{q1}_{q2}")
        cz_gate.append(cz_layer)

    def cz_var(depth: int, q1: int, q2: int) -> z3.BoolRef:
        lo, hi = sorted((q1, q2))
        return cz_gate[depth][lo, hi]

    # ---------------------------------------------------------------------
    # Layer consistency: every qubit has exactly one role per layer.
    # ---------------------------------------------------------------------

    for depth in range(max_depth):
        for q in range(n):
            incident_cx: list[z3.BoolRef] = []

            for other in range(n):
                if other == q:
                    continue
                incident_cx.extend([
                    cx_gate[depth][q, other],
                    cx_gate[depth][other, q],
                ])

            incident_cz: list[z3.BoolRef] = [cz_var(depth, q, other) for other in range(n) if other != q]

            add_exactly_one([
                idle[depth][q],
                h_gate[depth][q],
                s_gate[depth][q],
                sqrt_x_gate[depth][q],
                *incident_cx,
                *incident_cz,
            ])

    # ---------------------------------------------------------------------
    # Optional two-qubit-gate budget.
    # ---------------------------------------------------------------------

    all_two_qubit_gates: list[z3.BoolRef] = []
    for depth in range(max_depth):
        all_two_qubit_gates.extend(cx_gate[depth].values())
        all_two_qubit_gates.extend(cz_gate[depth].values())

    if max_two_qubit_gates is not None:
        if exact_two_qubit_count:
            add_cardinality_eq(all_two_qubit_gates, max_two_qubit_gates)
        else:
            add_cardinality_le(all_two_qubit_gates, max_two_qubit_gates)

    # ---------------------------------------------------------------------
    # Tableau transition constraints.
    # ---------------------------------------------------------------------

    for depth in range(max_depth):
        next_depth = depth + 1

        # Single-qubit gates and idles.
        for q in range(n):
            for row in range(row_count):
                old_x = tableau_x[depth][row][q]
                old_z = tableau_z[depth][row][q]
                new_x = tableau_x[next_depth][row][q]
                new_z = tableau_z[next_depth][row][q]

                # I
                solver.add(
                    z3.Implies(
                        idle[depth][q],
                        z3.And(new_x == old_x, new_z == old_z),
                    )
                )

                # H: X <-> Z
                solver.add(
                    z3.Implies(
                        h_gate[depth][q],
                        z3.And(new_x == old_z, new_z == old_x),
                    )
                )

                # S: Z ^= X
                solver.add(
                    z3.Implies(
                        s_gate[depth][q],
                        z3.And(new_x == old_x, new_z == z3.Xor(old_z, old_x)),
                    )
                )

                # SQRT_X: X ^= Z
                solver.add(
                    z3.Implies(
                        sqrt_x_gate[depth][q],
                        z3.And(new_x == z3.Xor(old_x, old_z), new_z == old_z),
                    )
                )

        # CX(control -> target):
        #   X_target ^= X_control
        #   Z_control ^= Z_target
        for (control, target), gate in cx_gate[depth].items():
            for row in range(row_count):
                old_x_control = tableau_x[depth][row][control]
                old_z_control = tableau_z[depth][row][control]
                old_x_target = tableau_x[depth][row][target]
                old_z_target = tableau_z[depth][row][target]

                new_x_control = tableau_x[next_depth][row][control]
                new_z_control = tableau_z[next_depth][row][control]
                new_x_target = tableau_x[next_depth][row][target]
                new_z_target = tableau_z[next_depth][row][target]

                solver.add(
                    z3.Implies(
                        gate,
                        z3.And(
                            new_x_control == old_x_control,
                            new_z_control == z3.Xor(old_z_control, old_z_target),
                            new_x_target == z3.Xor(old_x_target, old_x_control),
                            new_z_target == old_z_target,
                        ),
                    )
                )

        # CZ(q1, q2):
        #   X unchanged
        #   Z_q1 ^= X_q2
        #   Z_q2 ^= X_q1
        for (q1, q2), gate in cz_gate[depth].items():
            for row in range(row_count):
                old_x_q1 = tableau_x[depth][row][q1]
                old_z_q1 = tableau_z[depth][row][q1]
                old_x_q2 = tableau_x[depth][row][q2]
                old_z_q2 = tableau_z[depth][row][q2]

                new_x_q1 = tableau_x[next_depth][row][q1]
                new_z_q1 = tableau_z[next_depth][row][q1]
                new_x_q2 = tableau_x[next_depth][row][q2]
                new_z_q2 = tableau_z[next_depth][row][q2]

                solver.add(
                    z3.Implies(
                        gate,
                        z3.And(
                            new_x_q1 == old_x_q1,
                            new_x_q2 == old_x_q2,
                            new_z_q1 == z3.Xor(old_z_q1, old_x_q2),
                            new_z_q2 == z3.Xor(old_z_q2, old_x_q1),
                        ),
                    )
                )

    # ---------------------------------------------------------------------
    # Terminal condition.
    #
    # Stabilizers have exactly m pivot half-columns.
    # Each pivot may be X-type or Z-type.
    #
    # Logical qubit i selects one non-pivot physical qubit q:
    #   X_i = X_q up to stabilizers
    #   Z_i = Z_q up to stabilizers
    #
    # On Z-pivot ancillas, logical Z-support is allowed because it can be
    # removed by stabilizer row additions. Logical X-support is forbidden.
    #
    # On X-pivot ancillas, logical X-support is allowed because it can be
    # removed by stabilizer row additions after terminal H normalization.
    # Logical Z-support is forbidden.
    # ---------------------------------------------------------------------

    final_depth = max_depth
    stabilizer_rows: list[int] = list(range(2 * k, 2 * k + m))

    x_pivot: list[z3.BoolRef] = [z3.Bool(f"x_pivot_{q}") for q in range(n)]
    z_pivot: list[z3.BoolRef] = [z3.Bool(f"z_pivot_{q}") for q in range(n)]

    for q in range(n):
        solver.add(x_pivot[q] == z3_or([tableau_x[final_depth][row][q] for row in stabilizer_rows]))
        solver.add(z_pivot[q] == z3_or([tableau_z[final_depth][row][q] for row in stabilizer_rows]))

        # This rewrite supports X- or Z-type terminal pivots.
        # It intentionally excludes Y-type pivots.
        solver.add(z3.Not(z3.And(x_pivot[q], z_pivot[q])))

    add_cardinality_eq([*x_pivot, *z_pivot], m)

    logical_at: list[list[z3.BoolRef]] = [[z3.Bool(f"logical_{i}_at_{q}") for q in range(n)] for i in range(k)]

    for logical in range(k):
        x_row = logical
        z_row = k + logical

        add_exactly_one(logical_at[logical])

        for q in range(n):
            is_pivot = z3.Or(x_pivot[q], z_pivot[q])

            # Selected non-pivot column carries the canonical X/Z pair.
            solver.add(
                z3.Implies(
                    logical_at[logical][q],
                    z3.And(
                        z3.Not(is_pivot),
                        tableau_x[final_depth][x_row][q],
                        z3.Not(tableau_z[final_depth][x_row][q]),
                        z3.Not(tableau_x[final_depth][z_row][q]),
                        tableau_z[final_depth][z_row][q],
                    ),
                )
            )

            # Non-pivot columns not selected by this logical pair vanish.
            solver.add(
                z3.Implies(
                    z3.And(z3.Not(is_pivot), z3.Not(logical_at[logical][q])),
                    z3.And(
                        z3.Not(tableau_x[final_depth][x_row][q]),
                        z3.Not(tableau_z[final_depth][x_row][q]),
                        z3.Not(tableau_x[final_depth][z_row][q]),
                        z3.Not(tableau_z[final_depth][z_row][q]),
                    ),
                )
            )

            # Z-pivot ancilla: only Z-support may remain in logical rows.
            solver.add(
                z3.Implies(
                    z_pivot[q],
                    z3.And(
                        z3.Not(tableau_x[final_depth][x_row][q]),
                        z3.Not(tableau_x[final_depth][z_row][q]),
                    ),
                )
            )

            # X-pivot ancilla: only X-support may remain in logical rows.
            solver.add(
                z3.Implies(
                    x_pivot[q],
                    z3.And(
                        z3.Not(tableau_z[final_depth][x_row][q]),
                        z3.Not(tableau_z[final_depth][z_row][q]),
                    ),
                )
            )

    # Distinct logical qubits must select distinct physical qubits.
    for q in range(n):
        add_cardinality_le([logical_at[logical][q] for logical in range(k)], 1)

    # ---------------------------------------------------------------------
    # Solve.
    # ---------------------------------------------------------------------

    status = solver.check()

    if status == z3.unsat:
        return "UNSAT"

    if status == z3.unknown:
        return f"UNKNOWN: {solver.reason_unknown()}"

    model = solver.model()

    # ---------------------------------------------------------------------
    # Extract reduction circuit.
    # ---------------------------------------------------------------------

    reduction = stim.Circuit()

    for depth in range(max_depth):
        # Single-qubit gates.
        for q in range(n):
            if model_bool(model, h_gate[depth][q]):
                reduction.append("H", [q])
            elif model_bool(model, s_gate[depth][q]):
                reduction.append("S", [q])
            elif model_bool(model, sqrt_x_gate[depth][q]):
                reduction.append("SQRT_X", [q])

        # Two-qubit gates.
        for (control, target), gate in cx_gate[depth].items():
            if model_bool(model, gate):
                reduction.append("CX", [control, target])

        for (q1, q2), gate in cz_gate[depth].items():
            if model_bool(model, gate):
                reduction.append("CZ", [q1, q2])

    # Normalize terminal X-pivot ancillas to Z-pivot ancillas.
    # In the final encoder this becomes initial H preparation of those ancillas.
    x_pivot_ancillas: list[int] = [q for q in range(n) if model_bool(model, x_pivot[q])]

    if x_pivot_ancillas:
        reduction.append("H", x_pivot_ancillas)

    # Extract which physical qubits carry the input logical qubits.
    encoding_qubits: list[int] = []
    for logical in range(k):
        chosen: list[int] = [q for q in range(n) if model_bool(model, logical_at[logical][q])]
        if len(chosen) != 1:
            return "MODEL_ERROR: logical placement was not unique"
        encoding_qubits.append(chosen[0])

    encoding_qubit_set: set[int] = set(encoding_qubits)
    ancilla_qubits: list[int] = [q for q in range(n) if q not in encoding_qubit_set]

    # ---------------------------------------------------------------------
    # Build encoder from inverse reduction.
    # ---------------------------------------------------------------------

    encoder_circuit = stim.Circuit()

    if ancilla_qubits:
        encoder_circuit.append("RZ", ancilla_qubits)

    encoder_circuit += reduction.inverse()

    # ---------------------------------------------------------------------
    # Pauli sign correction.
    #
    # This preserves the same convention as your previous implementation.
    # The SMT model ignores Pauli signs, so the resulting binary tableau may
    # need a final Pauli correction.
    # ---------------------------------------------------------------------

    inverse_unitary = reduction.inverse()
    stim_tableau = inverse_unitary.to_tableau().to_numpy()

    x_part = stim_tableau[2].astype(int)
    z_part = stim_tableau[3].astype(int)
    signs = stim_tableau[-1].astype(int)

    signed_tableau = np.hstack((x_part, z_part, np.array([signs]).T))

    if not np.all(signed_tableau[:, -1] == 0):
        kernel = mod2.nullspace(signed_tableau)

        if kernel.size == 0:
            return "MODEL_ERROR: could not find Pauli sign correction"

        correction_symplectic = kernel[-1]

        if correction_symplectic[-1] != 1:
            return "MODEL_ERROR: invalid Pauli sign-correction kernel"

        z_correction = correction_symplectic[:n]
        x_correction = correction_symplectic[n:-1]

        for q, (xv, zv) in enumerate(zip(x_correction, z_correction, strict=False)):
            if xv == 1 and zv == 1:
                encoder_circuit.append("Y", [q])
            elif xv == 1:
                encoder_circuit.append("X", [q])
            elif zv == 1:
                encoder_circuit.append("Z", [q])

    return CliffordIsometry.from_stim_circuit(encoder_circuit)


def gate_optimal_encoding_circuit(
    code: CSSCode,
    min_gates: int = 1,
    max_gates: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> CNOTCircuit | None:
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
    checks, logicals = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.num_rows()

    checks_and_logicals = np.vstack((checks.matrix, logicals.matrix))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(n_checks, n_checks + logicals.num_rows())),
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
    if checks.type == "Z":
        cnots = [(j, i) for i, j in cnots]

    return build_css_encoder_from_cnot_list(
        CheckMatrix(reduced_checks_and_logicals[:n_checks], pauli_type=checks.type),
        CheckMatrix(reduced_checks_and_logicals[n_checks:], pauli_type=logicals.type),
        cnots,
    )


def depth_optimal_encoding_circuit(
    code: CSSCode,
    min_depth: int = 1,
    max_depth: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> CNOTCircuit | None:
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
    checks, logicals = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.num_rows()
    checks_and_logicals = np.vstack((checks.matrix, logicals.matrix))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(n_checks, n_checks + logicals.num_rows())),
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
    if checks.type == "Z":
        cnots = [(j, i) for i, j in cnots]

    return build_css_encoder_from_cnot_list(
        CheckMatrix(reduced_checks_and_logicals[:n_checks], pauli_type=checks.type),
        CheckMatrix(reduced_checks_and_logicals[n_checks:], pauli_type=logicals.type),
        cnots,
    )


def _get_matrix_with_fewest_checks(code: CSSCode) -> tuple[CheckMatrix, CheckMatrix]:
    """Return the stabilizer matrix with the fewest checks, the corresponding logicals and a bool indicating whether X- or Z-checks have been returned."""
    use_x_checks = code.Hx.shape[0] < code.Hz.shape[0]
    checks = code.Hx if use_x_checks else code.Hz
    logicals = code.Lx if use_x_checks else code.Lz
    type_ = "X" if use_x_checks else "Z"
    return CheckMatrix(checks, type_), CheckMatrix(logicals, type_)


def _final_matrix_constraint_partially_full_reduction(
    columns: npt.NDArray[np.bool_], full_reduction_rows: list[int], rank: int
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


def gottesman_encoding_circuit(tableau: StabilizerTableau | Sequence[str]) -> CliffordIsometry:
    """Synthesize encoding circuit for a stabilizer code as described in chapter 6.4 of Gottesman's book.

    Assumes all signs of the stabilizers are +1.

    Args:
        tableau: The stabilizer tableau of the code to synthesize the encoding circuit for.

    Returns:
        stim circuit implementing the encoding and a list of qubits that are used to encode the logical qubits.
    """
    if isinstance(tableau, Sequence):
        tableau = StabilizerTableau.from_pauli_strings(tableau)  # ty: ignore[invalid-argument-type]

    nq = tableau.n
    mat = tableau.tableau.matrix.copy()
    x_part = mat[:, :nq]
    z_part = mat[:, nq:]

    circ = stim.Circuit()
    n_rows = mat.shape[0]

    initialized = []
    for row in range(n_rows):
        # find row with either x_part[row][i] = 1 or z_part[row][i] = 1
        pivot = row
        column = row

        while column < nq and x_part[pivot][column] != 1 and z_part[pivot][column] != 1:
            found_pivot = False
            for p in range(row, n_rows):
                if x_part[p][column] == 1 or z_part[p][column] == 1:
                    pivot = p
                    found_pivot = True
                    break
            if not found_pivot:
                column += 1
                pivot = row
        if column >= nq:
            # No valid pivot found, invalid tableau
            msg = "Invalid tableau: could not find a valid pivot."
            raise ValueError(msg)
        initialized.append(column)
        # swap to row i
        t = x_part[pivot].copy()
        x_part[pivot] = x_part[row]
        x_part[row] = t

        t = z_part[pivot].copy()
        z_part[pivot] = z_part[row]
        z_part[row] = t

        if x_part[row][column] == 0:
            circ.append("H", [column])
            t = x_part[:, column].copy()
            x_part[:, column] = z_part[:, column]
            z_part[:, column] = t

        # reduce column
        for q in np.where(x_part[row])[0]:
            if q == column:
                continue
            circ.append("CX", [column, q])
            x_part[:, q] ^= x_part[:, column]
            z_part[:, column] ^= z_part[:, q]

        if z_part[row][column] == 1:
            circ.append("S", [column])
            z_part[:, column] ^= x_part[:, column]

        for q in np.where(z_part[row])[0]:
            if q == column:
                continue
            circ.append("CZ", [column, q])
            z_part[:, q] ^= x_part[:, column]
            z_part[:, column] ^= x_part[:, q]

        # reduce stabilizers below row
        x_part[:, column] = 0
        x_part[row, column] = 1

    circ.append("H", initialized)
    circ = circ.inverse()

    signs = [s.sign for s in circ.to_tableau().to_stabilizers()]
    for row, sign in enumerate(signs):
        if sign == -1:
            circ.insert(0, stim.CircuitInstruction("X", [row]))
    iso = CliffordIsometry.from_stim_circuit(circ)
    for q in initialized:
        iso.initialize_qubit(q, basis="Z")
    return iso


def synthesize_clifford(
    tableau: StabilizerTableau,
    use_cnots_if_css: bool = True,
    config: SynthesisConfig | None = None,
) -> CliffordIsometry:
    """Synthesize a stim circuit implementing a Clifford operation to minimize two-qubit gate count.

    Args:
        tableau: The stabilizer tableau representing the Clifford operation to synthesize.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the tableau is CSS.
        config: Configuration options for the synthesis process.

    Returns:
        A CliffordIsometry representing the synthesized Clifford operation that implements
        the same operation as the input tableau. The synthesis aims to minimize the two-qubit
        gate count. If the tableau is CSS and use_cnots_if_css is True, the circuit uses only
        CNOT gates; otherwise, a general Clifford synthesis is performed.
    """
    if tableau.is_css() and use_cnots_if_css:
        x_checks, z_checks = tableau.to_css()
        assert isinstance(config, SynthesisConfig) or config is None, (
            "CNOTSynthesisConfig must be provided when use_cnots_if_css is True."
        )
        logicals = x_checks if x_checks.num_rows() <= z_checks.num_rows() else z_checks
        return cnot_encoding_circuit(
            CheckMatrix(np.empty((0, tableau.n), dtype=np.int8), pauli_type=logicals.type),
            logicals,
            config=config,
        )

    assert isinstance(config, SynthesisConfig) or config is None, (
        "CliffordSynthesisConfig must be provided when use_cnots_if_css is False."
    )
    ops, _ = synthesize_non_css(
        tableau,
        config=config,
    )
    return CliffordIsometry.from_stim_circuit(ops.to_circuit_inverse())


def synthesize_encoding_circuit(
    code: StabilizerCode,
    config: SynthesisConfig | None = None,
    use_cnots_if_css: bool = True,
) -> CliffordIsometry:
    """Synthesize an encoding circuit for the given stabilizer code.

    Args:
        code: The stabilizer code to synthesize the encoding circuit for.
        config: Configuration options for the synthesis process.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the code is CSS.

    Returns:
        A CliffordIsometry that implements the encoding circuit for the given stabilizer code.
    """
    if use_cnots_if_css and isinstance(code, CSSCode):
        x_checks = CheckMatrix(code.Hx, pauli_type="X")
        z_checks = CheckMatrix(code.Hz, pauli_type="Z")
        x_logicals = CheckMatrix(code.Lx, pauli_type="X")
        z_logicals = CheckMatrix(code.Lz, pauli_type="Z")
        checks, logicals = (
            (x_checks, x_logicals) if x_checks.num_rows() <= z_checks.num_rows() else (z_checks, z_logicals)
        )

        assert isinstance(config, SynthesisConfig) or config is None, (
            "CNOTSynthesisConfig must be provided when use_cnots_if_css is True."
        )
        return cnot_encoding_circuit(checks, logicals, config=config)

    assert isinstance(config, SynthesisConfig) or config is None, (
        "CliffordSynthesisConfig must be provided when use_cnots_if_css is False."
    )

    log_mat: npt.NDArray[np.int8] = np.vstack((code.x_logicals.tableau.matrix, code.z_logicals.tableau.matrix))
    log_phase: npt.NDArray[np.int8] = np.hstack((code.x_logicals.phase, code.z_logicals.phase))

    return encoder_from_stabilizers_and_logicals(code.generators, StabilizerTableau(log_mat, log_phase), config=config)


def resynthesize_stim_circuit(
    circ: stim.Circuit,
    use_cnots_if_css: bool = True,
    config: SynthesisConfig | None = None,
) -> stim.Circuit:
    """Resynthesize a stim circuit implementing a Clifford operation to minimize two-qubit gate count.

    Args:
        circ: The stim.Circuit to resynthesize.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the circuit is CSS.
        config: Configuration options for the synthesis process.

    Returns:
        A stim.Circuit that implements the same operation as the input circuit but with potentially fewer two
    """
    tableau = StabilizerTableau.from_stim_circuit(circ)
    return synthesize_clifford(tableau, use_cnots_if_css=use_cnots_if_css, config=config).to_stim_circuit()


def encoder_from_stabilizers_and_logicals(
    stabilizers: StabilizerTableau,
    logicals: StabilizerTableau,
    optimize_tableau_before_synthesis: bool = True,
    config: SynthesisConfig | None = None,
) -> CliffordIsometry:
    """Synthesize an encoding circuit for a stabilizer code given its stabilizers and logicals as tableaux.

    Args:
        stabilizers: A tableau representing the stabilizers of the code.
        logicals: A tableau representing the logical operators of the code.
        optimize_tableau_before_synthesis: Whether to perform row operations on the combined tableau to optimize it for synthesis before synthesizing the circuit.
        config: Configuration options for the synthesis process.

    Returns:
        A CliffordIsometry that implements the encoding circuit for the given stabilizer code.
    """
    if stabilizers.n != logicals.n:
        msg = "Stabilizers and logicals must have the same number of qubits."
        raise ValueError(msg)
    if stabilizers.num_rows() + logicals.num_rows() > stabilizers.n * 2:
        msg = "The total number of stabilizers and logicals must be less than or equal to 2n."
        raise ValueError(msg)

    full_tableau = combine_stabilizer_and_logical_tableau(stabilizers, logicals)
    stab_indices = list(range(logicals.num_rows() // 2, logicals.num_rows() // 2 + stabilizers.num_rows()))
    if optimize_tableau_before_synthesis:
        optimized_tableau = optimize_tableau(full_tableau, stab_rows=stab_indices)
    else:
        optimized_tableau = full_tableau

    iso = synthesize_clifford(
        optimized_tableau,
        use_cnots_if_css=False,
        config=config,
    )
    iso.initialize_qubits(stab_indices, basis="Z")
    return iso


def optimize_tableau(tableau: StabilizerTableau, stab_rows: list[int]) -> StabilizerTableau:
    """Optimize a stabilizer tableau by performing row operations to reduce the cost of the initial tableau for synthesis."""
    tab = tableau.copy()

    best = (tab, score_symplectic(tab)[0])
    improved = True
    half = tableau.num_rows() // 2
    x_logical_rows = [i for i in range(half) if i not in stab_rows]
    z_logical_rows = [i + half for i in range(half) if i not in stab_rows]
    logical_rows = x_logical_rows + z_logical_rows
    k = len(logical_rows) // 2
    while improved:
        improved = False
        for i in range(len(stab_rows)):
            for j in range(len(stab_rows)):
                if i == j:
                    continue
                tab = tableau.copy()
                mat = tab.tableau.matrix
                destabs = mat[:half][stab_rows]
                stabs = mat[half:][stab_rows]
                stabs[i] ^= stabs[j]
                destabs[j] ^= destabs[i]
                mat[:half][stab_rows] = destabs
                mat[half:][stab_rows] = stabs
                new_score, _ = score_symplectic(StabilizerTableau(mat, tableau.phase.copy()))
                if lexicographical_compare_np(new_score, best[1]):
                    best = (tab, new_score)
                    improved = True
            for j in range(len(logical_rows)):
                tab = tableau.copy()
                mat = tab.tableau.matrix
                destabs = mat[:half][stab_rows]
                stabs = mat[half:][stab_rows]

                other_log = mat[logical_rows[(j + k) % (2 * k)]]

                destabs[i] ^= other_log
                logj = mat[logical_rows[j]]

                logj ^= stabs[i]
                mat[:half][stab_rows] = destabs
                mat[logical_rows[j]] = logj
                new_score, _ = score_symplectic(StabilizerTableau(mat, tableau.phase.copy()))
                if lexicographical_compare_np(new_score, best[1]):
                    best = (tab, new_score)
                    improved = True
        tableau = best[0]

    return best[0]


def combine_stabilizer_and_logical_tableau(
    stabilizers: StabilizerTableau, logicals: StabilizerTableau
) -> StabilizerTableau:
    """Combine a stabilizer tableau and a logical tableau, then complete with destabilizers.

    Args:
        stabilizers: A tableau representing the stabilizers of the code (without destabilizers).
        logicals: A tableau containing logical operators.

    Returns:
        A combined tableau with destabilizers added, suitable for circuit synthesis.
    """
    if stabilizers.n != logicals.n:
        msg = "Stabilizers and logicals must act on the same number of qubits."
        raise ValueError(msg)

    m = stabilizers.num_rows()

    # Combine stabilizers and logicals into a single tableau
    x_logicals = logicals.tableau.matrix[: logicals.num_rows() // 2]
    z_logicals = logicals.tableau.matrix[logicals.num_rows() // 2 :]
    x_logicals_phase = logicals.phase[: logicals.num_rows() // 2]
    z_logicals_phase = logicals.phase[logicals.num_rows() // 2 :]
    combined_matrix = np.vstack([x_logicals, z_logicals, stabilizers.tableau.matrix])

    combined_phase = np.hstack([x_logicals_phase, z_logicals_phase, stabilizers.phase])
    combined_tableau = StabilizerTableau(combined_matrix, combined_phase)

    # Complete with destabilizers for the stabilizers only
    # The stabilizer rows are at indices 0 to m-1
    stab_rows = list(range(logicals.num_rows(), logicals.num_rows() + m))
    return complete_stabilizer_tableau_with_destabilizers(combined_tableau, stab_rows)


def _remove_redundant_stabilizers(checks: CheckMatrix) -> CheckMatrix:
    """Remove redundant stabilizers from the check matrix without impacting stabilizer weight."""
    rnk = mod2.rank(checks.matrix)
    if rnk == checks.num_rows():
        return checks
    independent_checks = np.array([checks.matrix[0]])
    prev_rnk = 1
    for row in checks.matrix[1:]:
        stacked = np.vstack((independent_checks, row))
        new_rnk = mod2.rank(stacked)
        if new_rnk > prev_rnk:
            independent_checks = stacked
            prev_rnk = new_rnk
        if prev_rnk == rnk:
            break
    return CheckMatrix(independent_checks, pauli_type=checks.type)


def cnot_encoding_circuit(
    checks: CheckMatrix, logicals: CheckMatrix, config: SynthesisConfig | None = None
) -> CNOTCircuit:
    """Synthesize an encoding circuit for the given CSS code using a heuristic greedy search.

    Args:
        checks: The stabilizer check matrix of the CSS code.
        logicals: The logical operator matrix of the CSS code.
        config: The configuration for the CNOT synthesis process.

    Returns:
        The synthesized encoding circuit.
    """
    logger.info("Starting encoding circuit synthesis.")

    if config is None:
        config = SynthesisConfig()

    checks = _remove_redundant_stabilizers(checks)
    n_stab = checks.num_rows()

    if checks.type != logicals.type:
        msg = f"Check matrix and logical matrix must have the same Pauli type. Got checks.type={checks.type}, logicals.type={logicals.type}"
        raise ValueError(msg)

    if checks.num_qubits() != logicals.num_qubits():
        msg = f"Check matrix and logical matrix must have the same number of qubits. Got checks: {checks.num_qubits()} qubits, logicals: {logicals.num_qubits()} qubits"
        raise ValueError(msg)

    mat = CheckMatrix(np.vstack((checks.matrix, logicals.matrix)), pauli_type=checks.type)

    ops, reduced_checks = synthesize_cnot(mat, config=config, n_stabs=n_stab)
    assert isinstance(reduced_checks, CheckMatrix)

    cnots = [(c.control, c.target) for c in reversed(ops) if isinstance(c, CNOT)]

    x_qubits = set()
    for row in range(n_stab):
        for col in range(reduced_checks.num_qubits()):
            if reduced_checks.matrix[row, col] == 1:
                x_qubits.add(col)

    logical_qubits = set()
    for row in range(n_stab, reduced_checks.num_rows()):
        for col in range(reduced_checks.num_qubits()):
            if reduced_checks.matrix[row, col] == 1:
                logical_qubits.add(col)

    z_qubits = set(range(reduced_checks.num_qubits())) - x_qubits - logical_qubits
    if checks.type == "Z":
        z_qubits, x_qubits = x_qubits, z_qubits

    return CNOTCircuit.from_cnot_list(cnots, z_qubits, x_qubits)
