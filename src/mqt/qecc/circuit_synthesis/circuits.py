# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveResetInZeroState

from ..codes import CSSCode
from .circuit_utils import compose_circuits

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    import numpy.typing as npt


class CNOTCircuit:
    """Represents a restricted quantum circuit composed of CNOT gates with optional qubit initialization."""

    def __init__(self) -> None:
        """Initialize an empty CNOT circuit."""
        self.cnots: list[tuple[int, int]] = []
        self.initializations: dict[int, str] = {}  # Dictionary mapping qubit index to initialization type ('Z' or 'X')

    def add_cnot(self, control: int, target: int) -> None:
        """Add a single CNOT gate to the circuit.

        Args:
            control: The control qubit index.
            target: The target qubit index.
        """
        if control < 0 or target < 0:
            msg = "Control and target qubits must have non-negative indices."
            raise ValueError(msg)
        if control == target:
            msg = "Control and target qubits cannot be the same."
            raise ValueError(msg)
        self.cnots.append((control, target))

    def add_cnots(self, cnot_pairs: Iterable[tuple[int, int]]) -> None:
        """Add multiple CNOT gates to the circuit.

        Args:
            cnot_pairs: An iterable of (control, target) pairs.
        """
        for control, target in cnot_pairs:
            self.add_cnot(control, target)

    def initialize_qubit(self, qubit: int, basis: str) -> None:
        """Initialize a qubit in the specified basis.

        Args:
            qubit: The qubit index to initialize.
            basis: The basis for initialization ('Z' or 'X').
        """
        if qubit < 0:
            msg = "Qubit index must be non-negative."
            raise ValueError(msg)
        if basis.capitalize() not in {"Z", "X"}:
            msg = "Initialization basis must be 'Z' or 'X'."
            raise ValueError(msg)
        self.initializations[qubit] = basis

    def is_initialized(self, qubit: int) -> bool:
        """Check if a qubit is initialized.

        Args:
            qubit: The qubit index to check.

        Returns:
            True if the qubit is initialized, False otherwise.
        """
        return qubit in self.initializations

    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the CNOT circuit to a stim.Circuit.

        Returns:
            A stim.Circuit representation of the CNOT circuit.
        """
        stim_circuit = stim.Circuit()

        # Add initializations
        for qubit, basis in self.initializations.items():
            stim_circuit.append("R" + basis, [qubit])

        # Add CNOT gates
        stim_circuit.append_operation("CX", [qubit for pair in self.cnots for qubit in pair])

        return stim_circuit

    def to_qiskit_circuit(self, remove_resets: bool = True) -> QuantumCircuit:
        """Convert the CNOT circuit to a qiskit.QuantumCircuit.

        Args:
            remove_resets: If set to `True`, removes resets in the |0> state from the circuit.

        Returns:
            A qiskit.QuantumCircuit representation of the CNOT circuit.
        """
        circ = QuantumCircuit.from_qasm_str(self.to_stim_circuit().to_qasm(open_qasm_version=2))
        if remove_resets:
            return RemoveResetInZeroState()(circ)
        return circ

    @classmethod
    def from_qiskit_circuit(
        cls, circ: QuantumCircuit, init_all: bool = False, initialized_qubits: Iterable[int] | None = None
    ) -> CNOTCircuit:
        """Construct a CNOT circuit from a qiskit `QuantumCircuit` object.

        Generally, circ must contain only CNOT gates. The only exception to this is if `initialized_qubits` is given and the first gate on a qubit is a Hadamard gate. Then the qubit is initialized in |+>.

        Args:
            circ: The `QuantumCircuit` to construct the CNOT circuit from.
            init_all: If set to `True`, all qubits are initialized.
            initialized_qubits: Qubits to initialized.

        Returns:
            CNOTCircuit representation of the input circuit.
        """
        cnot_circuit = cls()
        if initialized_qubits is None:
            initialized_qubits = set()
        else:
            for qubit in initialized_qubits:
                cnot_circuit.initialize_qubit(qubit, "Z")

        # Initialize all qubits if `init_all` is True
        if init_all:
            for qubit in range(circ.num_qubits):
                cnot_circuit.initialize_qubit(qubit, "Z")

        initialized = [False for _ in range(circ.num_qubits)]
        # Parse the circuit
        for instruction in circ.data:
            gate = instruction.operation
            qubits = [circ.find_bit(q)[0] for q in instruction.qubits]

            if gate.name == "h" and len(qubits) == 1:
                qubit = qubits[0]
                # Handle Hadamard gates for initialization
                if initialized[qubit]:
                    msg = f"Hadamard gate on qubit that is already initialized: {qubit}."
                    raise ValueError(msg)
                if qubit in initialized_qubits or init_all:
                    cnot_circuit.initialize_qubit(qubit, "X")
                    initialized[qubit] = True
                else:
                    msg = f"Hadamard gate on uninitialized qubit {qubit}."
                    raise ValueError(msg)
            elif gate.name == "cx" and len(qubits) == 2:
                # Handle CNOT gates
                cnot_circuit.add_cnot(qubits[0], qubits[1])
                initialized[qubits[0]] = True
                initialized[qubits[1]] = True
            else:
                msg = f"Unsupported gate {gate.name} in the circuit."
                raise ValueError(msg)

        return cnot_circuit

    @classmethod
    def from_stim_circuit(cls, circ: stim.Circuit) -> CNOTCircuit:
        """Construct a CNOT circuit from a `stim.Circuit` object.

        Generally, circ must contain only CNOT gates and initializations in the Z- or X-basis.

        Args:
            circ: The `stim.Circuit` to construct the CNOT circuit from.

        Returns:
            CNOTCircuit representation of the input circuit.
        """
        # determine which qubits are initialized in what basis.
        cnot_circuit = cls()
        initialized = [False for _ in range(circ.num_qubits)]
        for gate in circ:
            name = gate.name
            for grp in gate.target_groups():
                if name in {"R", "RZ"}:
                    q = grp[0].qubit_value
                    if initialized[q]:
                        msg = f"Qubit {q} reset during circuit."
                        raise ValueError(msg)
                    cnot_circuit.initialize_qubit(grp[0].qubit_value, basis="Z")
                    initialized[q] = True

                elif name == "RX":
                    q = grp[0].qubit_value
                    if initialized[q]:
                        msg = f"Qubit {q} reset during circuit."
                        raise ValueError(msg)
                    cnot_circuit.initialize_qubit(grp[0].qubit_value, basis="X")
                    initialized[q] = True
                elif name == "CX":
                    control, target = grp[0].qubit_value, grp[1].qubit_value
                    cnot_circuit.add_cnot(control, target)
                    initialized[control] = True
                    initialized[target] = True
                else:
                    msg = f"Unsupported gate {name} in the circuit."
                    raise ValueError(msg)

        return cnot_circuit

    @classmethod
    def from_cnot_list(
        cls, cnots: Iterable[tuple[int, int]], initialize_z: Iterable[int], initialize_x: Iterable[int]
    ) -> CNOTCircuit:
        """Construct CNOT circuit from list of CNOTs.

        Args:
            cnots: Control, target pairs defining CNOT interactions.
            initialize_z: Qubits that should be initialized in the Z-basis
            initialize_x: Qubits that should be initialized in the X-basis

        Returns:
            CNOT circuit
        """
        cnot_circuit = cls()
        cnot_circuit.add_cnots(cnots)
        for q in initialize_z:
            cnot_circuit.initialize_qubit(q, "Z")
        for q in initialize_x:
            cnot_circuit.initialize_qubit(q, "X")
        cnot_circuit._check_valid()
        return cnot_circuit

    def is_state(self) -> bool:
        """Check if all qubits used in the circuit are initialized.

        Returns:
            True if all qubits involved in CNOT operations are initialized, False otherwise.
        """
        used_qubits = {qubit for control, target in self.cnots for qubit in (control, target)}
        return used_qubits.issubset(self.initializations.keys())

    def num_qubits(self) -> int:
        """Return the number of qubits used in the circuit.

        The number of qubits is determined by the highest index of any CNOT control or target qubit,
        """
        cnot_indices = [qubit for control, target in self.cnots for qubit in (control, target)]
        init_indices = list(self.initializations.keys())
        return max(cnot_indices + init_indices, default=0) + 1

    def num_inputs(self) -> int:
        """Get the number of uninitialized qubits, i.e., the inputs of the isometry.

        Returns:
            The number of uninitialized_qubits.
        """
        return self.num_qubits() - len(self.initializations)

    def get_plus_initialized(self) -> list[int]:
        """Get the list of qubits initialized in the |+> state.

        Returns:
            A list of qubit indices initialized in the |+> state.
        """
        return [qubit for qubit, basis in self.initializations.items() if basis.upper() == "X"]

    def get_zero_initialized(self) -> list[int]:
        """Get the list of qubits initialized in the |0> state.

        Returns:
            A list of qubit indices initialized in the |0> state.
        """
        return [qubit for qubit, basis in self.initializations.items() if basis.upper() == "Z"]

    def get_uninitialized(self) -> list[int]:
        """Get the list of uninitialized qubits.

        Returns:
            A list of uninitialized qubits.
        """
        return [qubit for qubit in range(self.num_qubits()) if qubit not in self.initializations]

    def draw(self, *args, **kwargs):  # noqa: ANN003, ANN002, ANN201
        """Draw the circuit using Qiskit visualization tools.

        Args:
            *args: Positional arguments for the Qiskit draw method.
            **kwargs: Keyword arguments for the Qiskit draw method.
        """
        return self.to_qiskit_circuit().draw(*args, **kwargs)

    def _propagate_paulis(self, xs: list[int], zs: list[int]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        x = np.zeros((len(xs), self.num_qubits()), dtype=np.int8)
        z = np.zeros((len(zs), self.num_qubits()), dtype=np.int8)
        for i, qubit in enumerate(xs):
            x[i, qubit] = 1
        for i, qubit in enumerate(zs):
            z[i, qubit] = 1

        for ctrl, trgt in self.cnots:
            x[:, trgt] ^= x[:, ctrl]
            z[:, ctrl] ^= z[:, trgt]

        return x, z

    def get_code(self) -> CSSCode:
        """Get CSS code defined by the circuit.

        A CNOT circuit with |0> and |+> initializations is the encoding isometry of some CSS code.
        The code is obtained by propagating the stabilizers of the initialized qubits to the end of the circuit.
        Qubits initialized in |+> define X-type stabilizers, while qubits initialized in |0> define Z-type stabilizers.

        Returns:
            A CSSCode object representing the code defined by the circuit.
        """
        pluses = self.get_plus_initialized()
        zeros = self.get_zero_initialized()
        hx, hz = self._propagate_paulis(pluses, zeros)
        return CSSCode(hx, hz)

    def num_cnots(self) -> int:
        """Get number of CNOT gates in the circuit."""
        return len(self.cnots)

    def depth(self) -> int:
        """Get the depth of the circuit.

        Returns:
            The depth of the circuit.
        """
        path_lengths = np.zeros(self.num_qubits(), dtype=int)
        for control, target in self.cnots:
            new_path_length = max(path_lengths[control], path_lengths[target]) + 1
            path_lengths[target] = new_path_length
            path_lengths[control] = new_path_length
        return int(np.max(path_lengths))

    def get_logical_x(self) -> dict[int, npt.NDArray[np.int8]]:
        """Get logical X operators of the isometry.

        Returns:
            A dictionary mapping input qubits to their X-logicals.
        """
        return {qubit: logicals[0] for qubit, logicals in self.get_logicals().items()}

    def get_logical_z(self) -> dict[int, npt.NDArray[np.int8]]:
        """Get logical Z operators of the isometry.

        Returns:
            A dictionary mapping input qubits to their Z-logicals.
        """
        return {qubit: logicals[1] for qubit, logicals in self.get_logicals().items()}

    def get_logicals(self) -> dict[int, tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]]:
        """Get logical operators of the isomety.

        Returns:
            A dictionary mapping input qubits to their X-logicals and Z-logicals.
        """
        if self.is_state():
            return {}

        inputs = self.get_uninitialized()
        x, z = self._propagate_paulis(inputs, inputs)
        return {qubit: (x[i], z[i]) for i, qubit in enumerate(inputs)}

    def copy(self) -> CNOTCircuit:
        """Create a copy of the CNOT circuit.

        Returns:
            A new CNOTCircuit instance with the same CNOT gates and initializations.
        """
        new_circuit = CNOTCircuit()
        new_circuit.cnots = self.cnots.copy()
        new_circuit.initializations = self.initializations.copy()
        return new_circuit

    def relabel_qubits(self, mapping: dict[int, int]) -> None:
        """Relabel the qubits in the circuit according to a given mapping.

        Args:
            mapping: A dictionary mapping old qubit indices to new qubit indices.
        """
        self.cnots = [(mapping[control], mapping[target]) for control, target in self.cnots]
        self.initializations = {mapping[q]: basis for q, basis in self.initializations.items()}
        self._check_valid()

    def _check_valid(self) -> None:
        """Check if the circuit is valid.

        Raises:
            ValueError: If the circuit contains invalid CNOT gates or initializations.
        """
        for control, target in self.cnots:
            if control < 0 or target < 0:
                msg = f"Invalid CNOT gate with negative indices: ({control}, {target})"
                raise ValueError(msg)
            if control == target:
                msg = f"CNOT gate with control and target being the same qubit: ({control}, {target})"
                raise ValueError(msg)

        for qubit, basis in self.initializations.items():
            if qubit < 0:
                msg = f"Invalid initialization on negative qubit index: {qubit}"
                raise ValueError(msg)
            if basis.upper() not in {"Z", "X"}:
                msg = f"Invalid initialization basis '{basis}' for qubit {qubit}"
                raise ValueError(msg)


def compose_cnot_circuits(
    circ1: CNOTCircuit, circ2: CNOTCircuit, wiring: dict[int, int] | None = None
) -> tuple[CNOTCircuit, dict[int, int], dict[int, int]]:
    """Compose two CNOT circuits.

    The circuits are composed only along the qubits that are connected by the `wiring` dict.
    All other qubits are assumed to be unconnected.
    If wire is None, then the circuits are simply vertically stacked.

    Args:
        circ1: The first CNOT circuit.
        circ2: The second CNOT circuit.
        wiring: Optional dict mapping outputs of `circ1` to inputs of `circ2`.

    Returns:
        A tuple containing the composed CNOT circuit and two mappings:
        - mapping1: Maps qubits of circ1 to the composed circuit.
        - mapping2: Maps qubits of circ2 to the composed circuit.
    """
    if wiring is None:
        wiring = {}

    # make sure that wires are not connected to initialized qubits in circ2
    if any(q in circ2.initializations for q in wiring.values()):
        msg = "Cannot compose circuits with wiring that connects to initialized qubits in circ2."
        raise ValueError(msg)

    composed, mapping1, mapping2 = compose_circuits(circ1.to_stim_circuit(), circ2.to_stim_circuit(), wiring)
    return CNOTCircuit.from_stim_circuit(composed), mapping1, mapping2
