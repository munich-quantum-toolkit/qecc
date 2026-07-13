# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Encoding interface for exact synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import z3

from ...codes.core.pauli import CheckMatrix, StabilizerTableau
from .css_utils import determine_css_initializations
from .encoding_depth import encode_clifford_depth, encode_css_depth
from .encoding_gate_count import encode_clifford_gate_count, encode_css_gate_count
from .extraction import (
    extract_clifford_depth_circuit,
    extract_clifford_gate_count_circuit,
    extract_cnot_depth_circuit,
    extract_cnot_gate_count_circuit,
)
from .symmetry import (
    add_clifford_depth_symmetry_breaking,
    add_clifford_gate_count_symmetry_breaking,
    add_css_depth_symmetry_breaking,
    add_css_gate_count_symmetry_breaking,
)

if TYPE_CHECKING:
    from ..circuits import CliffordIsometry, CNOTCircuit
    from .gate_operations import SymbolicGateOperation
    from .vars import CliffordDepthVars, CliffordGateCountVars


class SynthesisEncoding(ABC):
    """Abstract base class for synthesis encodings.

    Concrete subclasses are initialised with their specific configuration (gate set,
    permutation flags, etc.).  A call to ``encode`` populates internal variable state
    so that ``extract_circuit`` and ``compute_actual_resources`` can be called on a
    satisfying model without threading the variable dict through the call chain.
    """

    gate_set: dict[str, type[SymbolicGateOperation]]

    @abstractmethod
    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
    ) -> z3.Solver:
        """Build Z3 encoding for the given resource bound.

        Intermediate SAT variables are stored on the instance for later retrieval
        by ``extract_circuit`` and ``compute_actual_resources``.

        Args:
            target: Target tableau or check matrix.
            k: Number of logical qubits.
            bound: Resource bound (number of gates or circuit depth).

        Returns:
            Configured Z3 solver.
        """

    @abstractmethod
    def extract_circuit(self, model: z3.ModelRef) -> CliffordIsometry | CNOTCircuit:
        """Extract a circuit from a satisfying SAT model.

        Must be called after a successful ``encode``.

        Args:
            model: Satisfying Z3 model.

        Returns:
            Extracted circuit.
        """

    @abstractmethod
    def compute_actual_resources(self, model: z3.ModelRef, /) -> int:
        """Compute the actual resource usage from a satisfying model.

        Must be called after a successful ``encode``.

        Args:
            model: Satisfying Z3 model.

        Returns:
            Actual resource count (gates or depth).
        """


def _compute_pivot_qubits(model: z3.ModelRef, n: int, k: int, bound: int) -> list[int]:
    """Determine which qubits are pivot (stabilizer) qubits from the satisfying model.

    Reads the final tableau Z variables and returns columns where any stabilizer
    row has Z support — these are the qubits that need to be reset to |0⟩.
    """
    num_stab = n - k
    pivot_qubits = []
    for q in range(n):
        for row in range(2 * k, 2 * k + num_stab):
            z_var = z3.Bool(f"tz_{bound}_{row}_{q}")
            if model.eval(z_var, model_completion=True):
                pivot_qubits.append(q)
                break
    return pivot_qubits


class CliffordGateCountEncoding(SynthesisEncoding):
    """Gate-count encoding for general Clifford isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        allow_qubit_permutation: bool = True,
        use_symmetry_breaking: bool = False,
    ) -> None:
        """Initialise gate-count Clifford encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            allow_qubit_permutation: Allow final qubit permutation in the terminal constraint.
            use_symmetry_breaking: Add symmetry-breaking constraints to prune the search space.
        """
        self.gate_set = gate_set
        self.allow_qubit_permutation = allow_qubit_permutation
        self.use_symmetry_breaking = use_symmetry_breaking
        self._n = 0
        self._bound = 0
        self._k = 0
        self._vars: CliffordGateCountVars | None = None

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
    ) -> z3.Solver:
        """Build gate-count encoding for a Clifford circuit."""
        if not isinstance(target, StabilizerTableau):
            msg = "CliffordGateCountEncoding requires StabilizerTableau"
            raise TypeError(msg)
        self._n = target.n
        self._bound = bound
        self._k = k
        self._vars = encode_clifford_gate_count(target, k, bound, self.allow_qubit_permutation, self.gate_set)
        if self.use_symmetry_breaking:
            add_clifford_gate_count_symmetry_breaking(self._vars.solver, bound, self._vars)
        return self._vars.solver

    def extract_circuit(self, model: z3.ModelRef) -> CliffordIsometry:
        """Extract a Clifford circuit from a gate-count model."""
        assert self._vars is not None
        pivot_qubits = _compute_pivot_qubits(model, self._n, self._k, self._bound) if self._k < self._n else None
        return extract_clifford_gate_count_circuit(model, self._n, self._bound, self._vars, self._k, pivot_qubits)

    def compute_actual_resources(self, model: z3.ModelRef) -> int:
        """Compute actual gate count."""
        assert self._vars is not None
        count = 0
        for slot in range(self._bound):
            all_bools = [sel[slot] for sel in self._vars.gate_sel.values()]
            if model.eval(z3.Or(*all_bools), model_completion=True):
                count += 1
        return count


class CliffordDepthEncoding(SynthesisEncoding):
    """Depth encoding for general Clifford isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        allow_qubit_permutation: bool = True,
        use_symmetry_breaking: bool = False,
        max_two_qubit_gates: int | None = None,
    ) -> None:
        """Initialise depth Clifford encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            allow_qubit_permutation: Allow final qubit permutation in the terminal constraint.
            use_symmetry_breaking: Add symmetry-breaking constraints to prune the search space.
            max_two_qubit_gates: Upper bound on the total number of two-qubit gates across all
                layers.  Useful for obtaining shallow circuits without an excessive gate count.
        """
        self.gate_set = gate_set
        self.allow_qubit_permutation = allow_qubit_permutation
        self.use_symmetry_breaking = use_symmetry_breaking
        self.max_two_qubit_gates = max_two_qubit_gates
        self._n = 0
        self._bound = 0
        self._k = 0
        self._vars: CliffordDepthVars | None = None

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
    ) -> z3.Solver:
        """Build depth encoding for a Clifford circuit."""
        if not isinstance(target, StabilizerTableau):
            msg = "CliffordDepthEncoding requires StabilizerTableau"
            raise TypeError(msg)
        self._n = target.n
        self._bound = bound
        self._k = k
        self._vars = encode_clifford_depth(target, k, bound, self.allow_qubit_permutation, self.gate_set)
        if self.use_symmetry_breaking:
            add_clifford_depth_symmetry_breaking(self._vars.solver, bound, self._vars)
        if self.max_two_qubit_gates is not None:
            two_q_flat = [
                v
                for gate_name, all_layer_vars in self._vars.gate_vars.items()
                if self._vars.gate_set[gate_name].IS_TWO_QUBIT
                for layer_vars in all_layer_vars
                for v in layer_vars
            ]
            if two_q_flat:
                self._vars.solver.add(z3.AtMost(*two_q_flat, self.max_two_qubit_gates))
        return self._vars.solver

    def extract_circuit(self, model: z3.ModelRef) -> CliffordIsometry:
        """Extract a Clifford circuit from a depth model."""
        assert self._vars is not None
        pivot_qubits = _compute_pivot_qubits(model, self._n, self._k, self._bound) if self._k < self._n else None
        return extract_clifford_depth_circuit(model, self._n, self._bound, self._vars, self._k, pivot_qubits)

    def compute_actual_resources(self, model: z3.ModelRef) -> int:
        """Compute actual circuit depth."""
        assert self._vars is not None
        actual_depth = 0
        for layer in range(self._bound):
            layer_active = any(
                model.eval(v, model_completion=True)
                for gate_name, all_layer_vars in self._vars.gate_vars.items()
                if gate_name != "ID"
                for v in all_layer_vars[layer]
            )
            if layer_active:
                actual_depth += 1
        return actual_depth


class CSSGateCountEncoding(SynthesisEncoding):
    """Gate-count encoding for CSS CNOT isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        m_x: int,
        use_symmetry_breaking: bool = False,
    ) -> None:
        """Initialise gate-count CSS encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            m_x: Number of independent X-stabilizer generators (rank of H_X).
            use_symmetry_breaking: Add symmetry-breaking constraints to prune the search space.
        """
        self.gate_set = gate_set
        self._m_x = m_x
        self.use_symmetry_breaking = use_symmetry_breaking
        self._n = 0
        self._bound = 0
        self._k = 0
        self._num_rows = 0
        self._is_x_type = True
        self._alpha_vars: list[z3.BitVecRef] = []
        self._beta_vars: list[z3.BitVecRef] = []

    def _apply_symmetry_breaking(self, solver: z3.Solver) -> None:
        add_css_gate_count_symmetry_breaking(solver, self._bound, self._alpha_vars, self._beta_vars)

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
    ) -> z3.Solver:
        """Build gate-count encoding for a CSS CNOT circuit."""
        if not isinstance(target, CheckMatrix):
            msg = "CSSGateCountEncoding requires CheckMatrix"
            raise TypeError(msg)
        self._n = target.num_qubits()
        self._bound = bound
        self._k = k
        self._num_rows = target.num_rows()
        self._is_x_type = target.is_x_type()
        solver, alpha_vars, beta_vars = encode_css_gate_count(target, k, self._m_x, bound, self.gate_set)
        self._alpha_vars = alpha_vars
        self._beta_vars = beta_vars
        if self.use_symmetry_breaking:
            self._apply_symmetry_breaking(solver)
        return solver

    def extract_circuit(self, model: z3.ModelRef) -> CNOTCircuit:
        """Extract a CSS CNOT circuit from a gate-count model."""
        matrix_vars_final = np.array(
            [[z3.Bool(f"m_{self._bound}_{row}_{q}") for q in range(self._n)] for row in range(self._num_rows)],
            dtype=object,
        )
        init_x, init_z = determine_css_initializations(
            model,
            self._n,
            self._num_rows,
            self._k,
            matrix_vars_final,
            self._is_x_type,
        )
        return extract_cnot_gate_count_circuit(
            model,
            self._n,
            self._bound,
            self._alpha_vars,
            self._beta_vars,
            init_x,
            init_z,
        )

    def compute_actual_resources(self, _model: z3.ModelRef) -> int:
        """Compute actual gate count.

        In the CSS gate-count encoding every slot is occupied by a CNOT, so the
        actual count equals the bound used during encoding.
        """
        return self._bound


class CSSDepthEncoding(SynthesisEncoding):
    """Depth encoding for CSS CNOT isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        m_x: int,
        use_symmetry_breaking: bool = False,
        max_two_qubit_gates: int | None = None,
    ) -> None:
        """Initialise depth CSS encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            m_x: Number of independent X-stabilizer generators (rank of H_X).
            use_symmetry_breaking: Add symmetry-breaking constraints to prune the search space.
            max_two_qubit_gates: Upper bound on the total number of CNOT gates across all layers.
                Useful for obtaining shallow circuits without an excessive CNOT count.
        """
        self.gate_set = gate_set
        self._m_x = m_x
        self.use_symmetry_breaking = use_symmetry_breaking
        self.max_two_qubit_gates = max_two_qubit_gates
        self._n = 0
        self._bound = 0
        self._k = 0
        self._num_rows = 0
        self._is_x_type = True
        self._cx_vars: list[list[z3.BoolRef]] = []
        self._id_vars: list[list[z3.BoolRef]] = []

    def _apply_symmetry_breaking(self, solver: z3.Solver) -> None:
        add_css_depth_symmetry_breaking(solver, self._n, self._bound, self._cx_vars, self._id_vars)

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
    ) -> z3.Solver:
        """Build depth encoding for a CSS CNOT circuit."""
        if not isinstance(target, CheckMatrix):
            msg = "CSSDepthEncoding requires CheckMatrix"
            raise TypeError(msg)
        self._n = target.num_qubits()
        self._bound = bound
        self._k = k
        self._num_rows = target.num_rows()
        self._is_x_type = target.is_x_type()
        solver, cx_vars, id_vars = encode_css_depth(target, k, self._m_x, bound, self.gate_set)
        self._cx_vars = cx_vars
        self._id_vars = id_vars
        if self.use_symmetry_breaking:
            self._apply_symmetry_breaking(solver)
        if self.max_two_qubit_gates is not None:
            cx_flat = [cx_var for layer_cx in cx_vars for cx_var in layer_cx]
            if cx_flat:
                solver.add(z3.AtMost(*cx_flat, self.max_two_qubit_gates))
        return solver

    def extract_circuit(self, model: z3.ModelRef) -> CNOTCircuit:
        """Extract a CSS CNOT circuit from a depth model."""
        matrix_vars_final = np.array(
            [[z3.Bool(f"m_{self._bound}_{row}_{q}") for q in range(self._n)] for row in range(self._num_rows)],
            dtype=object,
        )
        init_x, init_z = determine_css_initializations(
            model,
            self._n,
            self._num_rows,
            self._k,
            matrix_vars_final,
            self._is_x_type,
        )
        return extract_cnot_depth_circuit(
            model,
            self._n,
            self._bound,
            self._cx_vars,
            init_x,
            init_z,
        )

    def compute_actual_resources(self, model: z3.ModelRef) -> int:
        """Compute actual circuit depth."""
        return sum(
            1
            for layer in range(self._bound)
            if any(
                model.eval(self._cx_vars[layer][cx_idx], model_completion=True)
                for cx_idx in range(len(self._cx_vars[layer]))
            )
        )
