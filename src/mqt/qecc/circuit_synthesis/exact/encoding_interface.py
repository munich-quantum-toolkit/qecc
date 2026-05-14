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

from ...codes.pauli import CheckMatrix, StabilizerTableau
from .css_utils import determine_css_initializations
from .encoding_depth import encode_clifford_depth, encode_css_depth
from .encoding_gate_count import encode_clifford_gate_count, encode_css_gate_count
from .extraction import (
    extract_clifford_depth_circuit,
    extract_clifford_gate_count_circuit,
    extract_cnot_depth_circuit,
    extract_cnot_gate_count_circuit,
)

if TYPE_CHECKING:
    from ..circuits import CliffordIsometry, CNOTCircuit
    from .gate_operations import SymbolicGateOperation


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
    def compute_actual_resources(self, model: z3.ModelRef) -> int:
        """Compute the actual resource usage from a satisfying model.

        Must be called after a successful ``encode``.

        Args:
            model: Satisfying Z3 model.

        Returns:
            Actual resource count (gates or depth).
        """


class CliffordGateCountEncoding(SynthesisEncoding):
    """Gate-count encoding for general Clifford isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        allow_qubit_permutation: bool = True,
    ) -> None:
        """Initialise gate-count Clifford encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            allow_qubit_permutation: Allow final qubit permutation in the terminal constraint.
        """
        self.gate_set = gate_set
        self.allow_qubit_permutation = allow_qubit_permutation
        self._n = 0
        self._bound = 0
        self._k = 0
        self._h_vars: list[z3.BoolRef] = []
        self._s_vars: list[z3.BoolRef] = []
        self._c_vars: list[z3.BoolRef] = []
        self._alpha_vars: list[z3.BitVecRef] = []
        self._beta_vars: list[z3.BitVecRef] = []

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
        solver, h_vars, s_vars, c_vars, alpha_vars, beta_vars = encode_clifford_gate_count(
            target, k, bound, self.allow_qubit_permutation, self.gate_set
        )
        self._h_vars = h_vars
        self._s_vars = s_vars
        self._c_vars = c_vars
        self._alpha_vars = alpha_vars
        self._beta_vars = beta_vars
        return solver

    def extract_circuit(self, model: z3.ModelRef) -> CliffordIsometry:
        """Extract a Clifford circuit from a gate-count model."""
        return extract_clifford_gate_count_circuit(
            model,
            self._n,
            self._bound,
            self._h_vars,
            self._s_vars,
            self._c_vars,
            self._alpha_vars,
            self._beta_vars,
            self._k,
        )

    def compute_actual_resources(self, model: z3.ModelRef) -> int:
        """Compute actual gate count."""
        return sum(
            1
            for slot in range(self._bound)
            if model.eval(
                z3.Or(self._h_vars[slot], self._s_vars[slot], self._c_vars[slot]),
                model_completion=True,
            )
        )


class CliffordDepthEncoding(SynthesisEncoding):
    """Depth encoding for general Clifford isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        allow_qubit_permutation: bool = True,
    ) -> None:
        """Initialise depth Clifford encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            allow_qubit_permutation: Allow final qubit permutation in the terminal constraint.
        """
        self.gate_set = gate_set
        self.allow_qubit_permutation = allow_qubit_permutation
        self._n = 0
        self._bound = 0
        self._k = 0
        self._h_vars: list[list[z3.BoolRef]] = []
        self._s_vars: list[list[z3.BoolRef]] = []
        self._cx_vars: list[list[z3.BoolRef]] = []

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
        solver, h_vars, s_vars, cx_vars, _id_vars = encode_clifford_depth(
            target, k, bound, self.allow_qubit_permutation, self.gate_set
        )
        self._h_vars = h_vars
        self._s_vars = s_vars
        self._cx_vars = cx_vars
        return solver

    def extract_circuit(self, model: z3.ModelRef) -> CliffordIsometry:
        """Extract a Clifford circuit from a depth model."""
        return extract_clifford_depth_circuit(
            model,
            self._n,
            self._bound,
            self._h_vars,
            self._s_vars,
            self._cx_vars,
            self._k,
        )

    def compute_actual_resources(self, model: z3.ModelRef) -> int:
        """Compute actual circuit depth."""
        actual_depth = 0
        for layer in range(self._bound):
            layer_has_gate = any(
                model.eval(self._h_vars[layer][q], model_completion=True)
                or model.eval(self._s_vars[layer][q], model_completion=True)
                for q in range(self._n)
            )
            if not layer_has_gate:
                layer_has_gate = any(
                    model.eval(self._cx_vars[layer][cx_idx], model_completion=True)
                    for cx_idx in range(len(self._cx_vars[layer]))
                )
            if layer_has_gate:
                actual_depth += 1
        return actual_depth


class CSSGateCountEncoding(SynthesisEncoding):
    """Gate-count encoding for CSS CNOT isometry synthesis."""

    def __init__(
        self,
        gate_set: dict[str, type[SymbolicGateOperation]],
        m_x: int,
    ) -> None:
        """Initialise gate-count CSS encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            m_x: Number of independent X-stabilizer generators (rank of H_X).
        """
        self.gate_set = gate_set
        self._m_x = m_x
        self._n = 0
        self._bound = 0
        self._k = 0
        self._num_rows = 0
        self._is_x_type = True
        self._alpha_vars: list[z3.BitVecRef] = []
        self._beta_vars: list[z3.BitVecRef] = []

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
    ) -> None:
        """Initialise depth CSS encoding.

        Args:
            gate_set: Gate set to use for synthesis.
            m_x: Number of independent X-stabilizer generators (rank of H_X).
        """
        self.gate_set = gate_set
        self._m_x = m_x
        self._n = 0
        self._bound = 0
        self._k = 0
        self._num_rows = 0
        self._is_x_type = True
        self._cx_vars: list[list[z3.BoolRef]] = []

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
        solver, cx_vars, _id_vars = encode_css_depth(target, k, self._m_x, bound, self.gate_set)
        self._cx_vars = cx_vars
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
