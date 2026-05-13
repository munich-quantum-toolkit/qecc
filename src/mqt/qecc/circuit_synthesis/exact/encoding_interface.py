# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Encoding interface for exact synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import z3

    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from ..circuits import CliffordIsometry, CNOTCircuit


class SynthesisEncoding(ABC):
    """Abstract base class for synthesis encodings."""

    @abstractmethod
    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
        **options: Any,
    ) -> tuple[z3.Solver, dict[str, Any]]:
        """Build Z3 encoding for given bound.

        Args:
            target: Target tableau or check matrix.
            k: Number of logical qubits.
            bound: Resource bound (gates or depth).
            **options: Additional encoding options.

        Returns:
            Tuple of (solver, variables_dict).
        """

    @abstractmethod
    def extract_circuit(
        self,
        model: z3.ModelRef,
        n: int,
        bound: int,
        variables: dict[str, Any],
        k: int,
    ) -> CliffordIsometry | CNOTCircuit:
        """Extract circuit from SAT model.

        Args:
            model: Z3 model from satisfiable formula.
            n: Number of qubits.
            bound: Resource bound used.
            variables: Variables dictionary from encode().
            k: Number of logical qubits.

        Returns:
            Extracted circuit.
        """

    @abstractmethod
    def compute_actual_resources(
        self,
        model: z3.ModelRef,
        bound: int,
        variables: dict[str, Any],
        n: int,
    ) -> int:
        """Compute actual resource usage from SAT model.

        Args:
            model: Z3 model.
            bound: Maximum bound.
            variables: Variables dictionary from encode().
            n: Number of qubits.

        Returns:
            Actual resource count (gates or depth).
        """


class CliffordGateCountEncoding(SynthesisEncoding):
    """Gate-count encoding for Clifford circuits."""

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
        **options: Any,
    ) -> tuple[z3.Solver, dict[str, Any]]:
        """Build gate-count encoding for Clifford circuit."""
        from ...codes.pauli import StabilizerTableau
        from .encoding_gate_count import encode_clifford_gate_count

        if not isinstance(target, StabilizerTableau):
            msg = "CliffordGateCountEncoding requires StabilizerTableau"
            raise TypeError(msg)

        allow_qubit_permutation = options.get("allow_qubit_permutation", True)

        solver, h_vars, s_vars, c_vars, alpha_vars, beta_vars = encode_clifford_gate_count(
            target,
            k,
            bound,
            allow_qubit_permutation,
        )

        variables = {
            "h_vars": h_vars,
            "s_vars": s_vars,
            "c_vars": c_vars,
            "alpha_vars": alpha_vars,
            "beta_vars": beta_vars,
        }

        return solver, variables

    def extract_circuit(
        self,
        model: z3.ModelRef,
        n: int,
        bound: int,
        variables: dict[str, Any],
        k: int,
    ) -> CliffordIsometry | CNOTCircuit:
        """Extract Clifford circuit from gate-count model."""
        from .extraction import extract_clifford_gate_count_circuit

        return extract_clifford_gate_count_circuit(
            model,
            n,
            bound,
            variables["h_vars"],
            variables["s_vars"],
            variables["c_vars"],
            variables["alpha_vars"],
            variables["beta_vars"],
            k,
        )

    def compute_actual_resources(
        self,
        model: z3.ModelRef,
        bound: int,
        variables: dict[str, Any],
        n: int,
    ) -> int:
        """Compute actual gate count."""
        import z3

        return sum(
            1
            for slot in range(bound)
            if model.eval(
                z3.Or(variables["h_vars"][slot], variables["s_vars"][slot], variables["c_vars"][slot]),
                model_completion=True,
            )
        )


class CliffordDepthEncoding(SynthesisEncoding):
    """Depth encoding for Clifford circuits."""

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
        **options: Any,
    ) -> tuple[z3.Solver, dict[str, Any]]:
        """Build depth encoding for Clifford circuit."""
        from ...codes.pauli import StabilizerTableau
        from .encoding_depth import encode_clifford_depth

        if not isinstance(target, StabilizerTableau):
            msg = "CliffordDepthEncoding requires StabilizerTableau"
            raise TypeError(msg)

        allow_qubit_permutation = options.get("allow_qubit_permutation", True)

        solver, h_vars, s_vars, cx_vars, id_vars = encode_clifford_depth(
            target,
            k,
            bound,
            allow_qubit_permutation,
        )

        variables = {
            "h_vars": h_vars,
            "s_vars": s_vars,
            "cx_vars": cx_vars,
            "id_vars": id_vars,
        }

        return solver, variables

    def extract_circuit(
        self,
        model: z3.ModelRef,
        n: int,
        bound: int,
        variables: dict[str, Any],
        k: int,
    ) -> CliffordIsometry | CNOTCircuit:
        """Extract Clifford circuit from depth model."""
        from .extraction import extract_clifford_depth_circuit

        return extract_clifford_depth_circuit(
            model,
            n,
            bound,
            variables["h_vars"],
            variables["s_vars"],
            variables["cx_vars"],
            k,
        )

    def compute_actual_resources(
        self,
        model: z3.ModelRef,
        bound: int,
        variables: dict[str, Any],
        n: int,
    ) -> int:
        """Compute actual depth."""
        actual_depth = 0
        for layer in range(bound):
            layer_has_gate = False
            for q in range(n):
                if model.eval(variables["h_vars"][layer][q], model_completion=True) or model.eval(
                    variables["s_vars"][layer][q], model_completion=True
                ):
                    layer_has_gate = True
                    break
            if not layer_has_gate:
                for cx_idx in range(len(variables["cx_vars"][layer])):
                    if model.eval(variables["cx_vars"][layer][cx_idx], model_completion=True):
                        layer_has_gate = True
                        break
            if layer_has_gate:
                actual_depth += 1
        return actual_depth


class CSSGateCountEncoding(SynthesisEncoding):
    """Gate-count encoding for CSS CNOT circuits."""

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
        **options: Any,
    ) -> tuple[z3.Solver, dict[str, Any]]:
        """Build gate-count encoding for CSS circuit."""
        from ...codes.pauli import CheckMatrix
        from .encoding_gate_count import encode_css_gate_count

        if not isinstance(target, CheckMatrix):
            msg = "CSSGateCountEncoding requires CheckMatrix"
            raise TypeError(msg)

        m_x = options.get("m_x")
        if m_x is None:
            msg = "m_x must be provided for CSS encoding"
            raise ValueError(msg)

        solver, alpha_vars, beta_vars = encode_css_gate_count(
            target,
            k,
            m_x,
            bound,
        )

        variables = {
            "alpha_vars": alpha_vars,
            "beta_vars": beta_vars,
            "n": target.num_qubits(),
            "num_rows": target.num_rows(),
            "is_x_type": target.is_x_type(),
        }

        return solver, variables

    def extract_circuit(
        self,
        model: z3.ModelRef,
        n: int,
        bound: int,
        variables: dict[str, Any],
        k: int,
    ) -> CliffordIsometry | CNOTCircuit:
        """Extract CSS circuit from gate-count model."""
        import numpy as np
        import z3

        from .extraction import extract_cnot_gate_count_circuit

        num_rows = variables["num_rows"]
        matrix_vars_final = np.array(
            [[z3.Bool(f"m_{bound}_{row}_{q}") for q in range(n)] for row in range(num_rows)], dtype=object
        )

        from .search import _determine_css_initializations

        init_x, init_z = _determine_css_initializations(
            model,
            n,
            num_rows,
            k,
            matrix_vars_final,
            variables["is_x_type"],
        )

        return extract_cnot_gate_count_circuit(
            model,
            n,
            bound,
            variables["alpha_vars"],
            variables["beta_vars"],
            init_x,
            init_z,
        )

    def compute_actual_resources(
        self,
        model: z3.ModelRef,
        bound: int,
        variables: dict[str, Any],
        n: int,
    ) -> int:
        """Compute actual gate count (all gates are used in CSS gate-count encoding)."""
        return bound


class CSSDepthEncoding(SynthesisEncoding):
    """Depth encoding for CSS CNOT circuits."""

    def encode(
        self,
        target: StabilizerTableau | CheckMatrix,
        k: int,
        bound: int,
        **options: Any,
    ) -> tuple[z3.Solver, dict[str, Any]]:
        """Build depth encoding for CSS circuit."""
        from ...codes.pauli import CheckMatrix
        from .encoding_depth import encode_css_depth

        if not isinstance(target, CheckMatrix):
            msg = "CSSDepthEncoding requires CheckMatrix"
            raise TypeError(msg)

        m_x = options.get("m_x")
        if m_x is None:
            msg = "m_x must be provided for CSS encoding"
            raise ValueError(msg)

        solver, cx_vars, id_vars = encode_css_depth(
            target,
            k,
            m_x,
            bound,
        )

        variables = {
            "cx_vars": cx_vars,
            "id_vars": id_vars,
            "n": target.num_qubits(),
            "num_rows": target.num_rows(),
            "is_x_type": target.is_x_type(),
        }

        return solver, variables

    def extract_circuit(
        self,
        model: z3.ModelRef,
        n: int,
        bound: int,
        variables: dict[str, Any],
        k: int,
    ) -> CliffordIsometry | CNOTCircuit:
        """Extract CSS circuit from depth model."""
        import numpy as np
        import z3

        from .extraction import extract_cnot_depth_circuit

        num_rows = variables["num_rows"]
        matrix_vars_final = np.array(
            [[z3.Bool(f"m_{bound}_{row}_{q}") for q in range(n)] for row in range(num_rows)], dtype=object
        )

        from .search import _determine_css_initializations

        init_x, init_z = _determine_css_initializations(
            model,
            n,
            num_rows,
            k,
            matrix_vars_final,
            variables["is_x_type"],
        )

        return extract_cnot_depth_circuit(
            model,
            n,
            bound,
            variables["cx_vars"],
            init_x,
            init_z,
        )

    def compute_actual_resources(
        self,
        model: z3.ModelRef,
        bound: int,
        variables: dict[str, Any],
        n: int,
    ) -> int:
        """Compute actual depth."""
        actual_depth = 0
        for layer in range(bound):
            layer_has_gate = False
            for cx_idx in range(len(variables["cx_vars"][layer])):
                if model.eval(variables["cx_vars"][layer][cx_idx], model_completion=True):
                    layer_has_gate = True
                    break
            if layer_has_gate:
                actual_depth += 1
        return actual_depth
