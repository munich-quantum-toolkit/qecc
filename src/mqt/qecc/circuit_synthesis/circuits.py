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

from ..codes import CSSCode, StabilizerCode
from ..codes.pauli import Pauli
from .circuit_utils import compose_circuits, num_two_qubit_gates, two_qubit_gate_depth

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    import numpy.typing as npt


class CliffordIsometry:
    """Circuit representation of a Clifford encoding isometry."""

    def __init__(self) -> None:
        """Initialize trivial isometry."""
        self._inputs: list[int] = []
        self._outputs: list[int] = []
        self._ancillas: set[int] = set()
        self._initializations: dict[int, str] = {}
        self._circ = stim.Circuit()

    def get_logical_x(self, idx: int) -> Pauli:
        """Get logical X operator of logical input at index.

        Args:
            idx: Index of the logical operator.

        Returns:
            Logical X operator.
        """
        if idx >= self.num_inputs():
            msg = "Given index is not a logical qubit index."
            raise ValueError(msg)
        tab = self.to_stim_circuit().to_tableau(ignore_reset=True)
        pauli_stim = tab.x_output(self._inputs[idx])
        return Pauli.from_stim(pauli_stim)

    def get_logical_z(self, idx: int) -> Pauli:
        """Get logical Z operator of logical input at index.

        Args:
            idx: Index of the logical operator.

        Returns:
            Logical Z operator.
        """
        if idx >= self.num_inputs():
            msg = "Given index is not a logical qubit index."
            raise ValueError(msg)

        tab = self.to_stim_circuit().to_tableau(ignore_reset=True)
        pauli_stim = tab.z_output(self._inputs[idx])
        return Pauli.from_stim(pauli_stim)

    def get_logical(self, idx: int) -> tuple[Pauli, Pauli]:
        """Get logical operators of logical input at index.

        Args:
            idx: Index of the logical qubit.

        Returns:
            Logical X and Z operators.
        """
        return self.get_logical_x(idx), self.get_logical_z(idx)

    def get_all_logicals(self) -> list[tuple[Pauli, Pauli]]:
        """Get logical X- and Z-operators of all logical qubits."""
        return [self.get_logical(i) for i in range(self.num_inputs())]

    def logical_to_input_mapping(self, code: StabilizerCode) -> list[int] | None:
        """Get mapping from logical qubits of the code to input qubits of the isometry.

        Args:
            code: Stabilizer code.

        Returns:
            A list mapping logical qubits of the code to input qubits of the isometry, or None if no such mapping exists, i.e. the code does not match the code represented by the isometry.
        """
        circuit_code = self.get_code()
        logical_mapping = code.get_logical_mapping(circuit_code)  # which circuit logicals map to which code logicals
        if logical_mapping is None:
            return None
        return [self._inputs[logical_mapping[i]] for i in range(len(logical_mapping))]

    def get_code(self) -> StabilizerCode:
        """Get the stabilizer code defined by the isometry.

        Returns:
            Stabilizer code.
        """
        # remove resets and remember basis
        circ_no_reset = stim.Circuit()
        basis = {}
        for gate in self.to_stim_circuit():
            if gate.name not in {"R", "RX", "RZ"}:
                circ_no_reset.append(gate)
            else:
                for grp in gate.target_groups():
                    q = grp[0].qubit_value
                    if gate.name in {"R", "RZ"}:
                        basis[q] = "Z"
                    elif gate.name == "RX":
                        basis[q] = "X"

        tab_no_reset = circ_no_reset.to_tableau()
        stabilizers = []
        for q in range(self.num_outputs()):
            if q not in self._inputs:
                pauli_stim = tab_no_reset.z_output(q) if basis.get(q, "Z") == "Z" else tab_no_reset.x_output(q)
                stabilizers.append(Pauli.from_stim(pauli_stim))
        logicals = self.get_all_logicals()
        return StabilizerCode(
            stabilizers, z_logicals=[log[1] for log in logicals], x_logicals=[log[0] for log in logicals]
        )

    @classmethod
    def from_stim_circuit(cls, circ: stim.Circuit) -> CliffordIsometry:
        """Construct Clifford isometry from stim Circuit.

        Args:
            circ: Stim Circuit representing the isometry.

        Returns:
            Clifford isometry.
        """
        iso = CliffordIsometry()
        n = circ.num_qubits
        iso._outputs = list(range(n))
        stripped = stim.Circuit()

        for gate in circ:
            name = gate.name
            if name in {"R", "RX", "RZ"}:
                for grp in gate.target_groups():
                    q = grp[0].qubit_value
                    iso._ancillas.add(q)
                    iso._initializations[q] = "X" if name == "RX" else "Z"
            else:
                stripped.append(gate)

        iso._circ = stripped
        iso._inputs = [q for q in iso._outputs if q not in iso._ancillas]

        return iso

    def to_stim_circuit(self, with_resets: bool = True) -> stim.Circuit:
        """Get the stim Circuit implementing the isometry.

        Args:
            with_resets: If set to `True`, includes resets in the |0> and |+> states for initialized qubits in the stim circuit.

        Returns:
            A stim.Circuit representation of the isometry.
        """
        if not with_resets:
            return self._circ.copy()
        result = stim.Circuit()

        for qubit, basis in self._initializations.items():
            result.append("R" + basis, [qubit])

        result += self._circ

        return result

    def outputs(self) -> list[int]:
        """Get output qubits."""
        return self._outputs

    def inputs(self) -> list[int]:
        """Get input qubits."""
        return self._inputs

    def draw(self, *args, **kwargs):  # noqa: ANN003, ANN002, ANN201
        """Draw the circuit using Qiskit visualization tools.

        Args:
            *args: Positional arguments for the Qiskit draw method.
            **kwargs: Keyword arguments for the Qiskit draw method.
        """
        return self.to_qiskit_circuit().draw(*args, **kwargs)

    def to_qiskit_circuit(self, remove_resets: bool = True) -> QuantumCircuit:
        """Convert the isometry to a qiskit.QuantumCircuit.

        Args:
            remove_resets: If set to `True`, removes resets in the |0> state from the circuit.

        Returns:
            A qiskit.QuantumCircuit representation of the isometry.
        """
        circ = QuantumCircuit.from_qasm_str(self.to_stim_circuit().to_qasm(open_qasm_version=2))
        if remove_resets:
            return RemoveResetInZeroState()(circ)
        return circ

    def num_inputs(self) -> int:
        """Get number of logical inputs."""
        return len(self._inputs)

    def num_outputs(self) -> int:
        """Get number of physical outputs."""
        return len(self._outputs)

    def initialize_qubit(self, qubit: int, basis: str) -> None:
        """Initialize a qubit in the specified basis.

        Args:
            qubit: The qubit index to initialize.
            basis: The basis for initialization ('Z' or 'X').
        """
        if qubit < 0:
            msg = "Qubit index must be non-negative."
            raise ValueError(msg)
        normalized_basis = basis.upper()
        if normalized_basis not in {"Z", "X"}:
            msg = "Initialization basis must be 'Z' or 'X'."
            raise ValueError(msg)

        if qubit in self._inputs:
            self._inputs.remove(qubit)
        self._initializations[qubit] = normalized_basis
        self._ancillas.add(qubit)

    def initialize_qubits(self, qubits: Iterable[int], basis: str) -> None:
        """Initialize multiple qubits in the specified basis.

        Args:
            qubits: An iterable of qubit indices to initialize.
            basis: The basis for initialization ('Z' or 'X').
        """
        for qubit in qubits:
            self.initialize_qubit(qubit, basis)

    def get_plus_initialized(self) -> list[int]:
        """Get the list of qubits initialized in the |+> state.

        Returns:
            A list of qubit indices initialized in the |+> state.
        """
        return [qubit for qubit, basis in self._initializations.items() if basis.upper() == "X"]

    def get_zero_initialized(self) -> list[int]:
        """Get the list of qubits initialized in the |0> state.

        Returns:
            A list of qubit indices initialized in the |0> state.
        """
        return [qubit for qubit, basis in self._initializations.items() if basis.upper() == "Z"]

    def get_uninitialized(self) -> list[int]:
        """Get the list of uninitialized qubits.

        Returns:
            A list of uninitialized qubits.
        """
        return [qubit for qubit in range(self.num_qubits()) if qubit not in self._initializations]

    def num_qubits(self) -> int:
        """Get the total number of qubits in the isometry.

        Returns:
            The total number of qubits.
        """
        return int(self.to_stim_circuit().num_qubits)

    def is_state(self) -> bool:
        """Check if all qubits used in the circuit are initialized.

        Returns:
            True if all qubits are initialized, False otherwise.
        """
        return len(self._inputs) == 0

    def num_two_qubit_gates(self) -> int:
        """Get the number of two-qubit gates in the circuit.

        Returns:
            The number of two-qubit gates.
        """
        return num_two_qubit_gates(self.to_stim_circuit())

    def depth(self) -> int:
        """Get the depth of the circuit.

        Returns:
            The depth of the circuit.
        """
        return two_qubit_gate_depth(self.to_stim_circuit())

    def get_initialized(self) -> dict[int, str]:
        """Get the initialized qubits and their initialization basis.

        Returns:
            A dictionary mapping qubit indices to their initialization basis ('Z' or 'X').
        """
        return self._initializations.copy()

    def is_initialized(self, qubit: int) -> bool:
        """Check if a qubit is initialized.

        Args:
            qubit: The qubit index to check.

        Returns:
            True if the qubit is initialized, False otherwise.
        """
        return qubit in self._initializations


class CNOTCircuit(CliffordIsometry):
    """Represents a restricted quantum circuit composed of CNOT gates with optional qubit initialization."""

    def __init__(self) -> None:
        """Initialize an empty CNOT circuit."""
        super().__init__()
        self.cnots: list[tuple[int, int]] = []
        self._initializations: dict[int, str] = {}

    def _add_input(self, qubit: int) -> None:
        """Add a qubit to the inputs if it is not already initialized or an input.

        Args:
            qubit: The qubit index to add as an input.
        """
        if qubit not in self._inputs and qubit not in self._initializations:
            self._inputs.append(qubit)

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
        self._add_input(control)
        self._add_input(target)

        self.cnots.append((control, target))

    def add_cnots(self, cnot_pairs: Iterable[tuple[int, int]]) -> None:
        """Add multiple CNOT gates to the circuit.

        Args:
            cnot_pairs: An iterable of (control, target) pairs.
        """
        for control, target in cnot_pairs:
            self.add_cnot(control, target)

    def to_stim_circuit(self, with_resets: bool = True) -> stim.Circuit:
        """Convert the CNOT circuit to a stim.Circuit.

        Args:
            with_resets: If set to `True`, includes resets in the |0> and |+> states for initialized qubits in the stim circuit.

        Returns:
            A stim.Circuit representation of the CNOT circuit.
        """
        stim_circuit = stim.Circuit()

        if with_resets:
            for qubit, basis in self._initializations.items():
                stim_circuit.append("R" + basis, [qubit])

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

        if init_all:
            for qubit in range(circ.num_qubits):
                cnot_circuit.initialize_qubit(qubit, "Z")

        initialized = [False for _ in range(circ.num_qubits)]
        for instruction in circ.data:
            gate = instruction.operation
            qubits = [circ.find_bit(q)[0] for q in instruction.qubits]

            if gate.name == "h" and len(qubits) == 1:
                qubit = qubits[0]
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
        for q in initialize_z:
            cnot_circuit.initialize_qubit(q, "Z")
        for q in initialize_x:
            cnot_circuit.initialize_qubit(q, "X")
        cnot_circuit.add_cnots(cnots)
        cnot_circuit._check_valid()
        return cnot_circuit

    def num_qubits(self) -> int:
        """Return the number of qubits used in the circuit.

        The number of qubits is determined by the highest index of any CNOT control or target qubit,
        """
        cnot_indices = [qubit for control, target in self.cnots for qubit in (control, target)]
        init_indices = list(self._initializations.keys())
        return max(cnot_indices + init_indices, default=0) + 1

    def outputs(self) -> list[int]:
        """Get output qubits.

        Returns:
            List of output qubit indices (all qubits used in the circuit).
        """
        return list(range(self.num_qubits()))

    def num_outputs(self) -> int:
        """Get number of physical outputs.

        Returns:
            Number of output qubits.
        """
        return self.num_qubits()

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

        logicals = self.get_logicals_css()
        lx = np.array([logicals[i][0] for i in self.inputs()])
        lz = np.array([logicals[i][1] for i in self.inputs()])

        return CSSCode(hx, hz, Lx=lx, Lz=lz)

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

    def get_logical_xs_css(self) -> dict[int, npt.NDArray[np.int8]]:
        """Get logical X operators of the isometry.

        Returns:
            A dictionary mapping input qubits to their X-logicals.
        """
        return {qubit: logicals[0] for qubit, logicals in self.get_logicals_css().items()}

    def get_logical_zs_css(self) -> dict[int, npt.NDArray[np.int8]]:
        """Get logical Z operators of the isometry.

        Returns:
            A dictionary mapping input qubits to their Z-logicals.
        """
        return {qubit: logicals[1] for qubit, logicals in self.get_logicals_css().items()}

    def get_logicals_css(self) -> dict[int, tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]]:
        """Get logical operators of the isomety.

        Returns:
            A dictionary mapping input qubits to their X-logicals and Z-logicals.
        """
        if self.is_state():
            return {}

        in_ = self.inputs()
        x, z = self._propagate_paulis(in_, in_)
        return {qubit: (x[i], z[i]) for i, qubit in enumerate(in_)}

    def copy(self) -> CNOTCircuit:
        """Create a copy of the CNOT circuit.

        Returns:
            A new CNOTCircuit instance with the same CNOT gates and initializations.
        """
        new_circuit = CNOTCircuit()
        new_circuit.cnots = self.cnots.copy()
        new_circuit._initializations = self._initializations.copy()
        new_circuit._inputs = self._inputs.copy()
        return new_circuit

    def relabel_qubits(self, mapping: dict[int, int]) -> None:
        """Relabel the qubits in the circuit according to a given mapping.

        Args:
            mapping: A dictionary mapping old qubit indices to new qubit indices.
        """
        self.cnots = [(mapping[control], mapping[target]) for control, target in self.cnots]
        self._initializations = {mapping[q]: basis for q, basis in self._initializations.items()}
        self._inputs = [mapping[q] for q in self._inputs]
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

        for qubit, basis in self._initializations.items():
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

    if any(q in circ2.get_initialized() for q in wiring.values()):
        msg = "Cannot compose circuits with wiring that connects to initialized qubits in circ2."
        raise ValueError(msg)

    composed, mapping1, mapping2 = compose_circuits(circ1.to_stim_circuit(), circ2.to_stim_circuit(), wiring)
    return CNOTCircuit.from_stim_circuit(composed), mapping1, mapping2


def compose(enc1: CliffordIsometry, enc2: CliffordIsometry, wiring: dict[int, int] | None = None) -> CliffordIsometry:
    """Compose two isometries to construct new isometry.

    Args:
        enc1: Left isometry (outputs serve as inputs to enc2)
        enc2: Right isometry
        wiring: Optional dict mapping outputs of `enc1` to inputs of `enc2`. If None, the inputs and outputs are connected in ascending order.

    Returns:
        Composed isometry.
    """
    if wiring is None:
        if enc1.num_outputs() != enc2.num_inputs():
            msg = "Cannot compose isometries with incompatible numbers of inputs and outputs."
            raise ValueError(msg)
        wiring = dict(zip(enc1.outputs(), enc2.inputs(), strict=True))
    else:
        enc1_outputs_set = set(enc1.outputs())
        enc2_inputs_set = set(enc2.inputs())

        invalid_keys = [k for k in wiring if k not in enc1_outputs_set]
        if invalid_keys:
            msg = f"Wiring keys {invalid_keys} are not valid outputs of enc1."
            raise ValueError(msg)

        invalid_values = [v for v in wiring.values() if v not in enc2_inputs_set]
        if invalid_values:
            msg = f"Wiring values {invalid_values} are not valid inputs of enc2."
            raise ValueError(msg)

        wiring_values = list(wiring.values())
        if len(wiring_values) != len(set(wiring_values)):
            msg = "Wiring cannot map multiple outputs to the same input."
            raise ValueError(msg)

    composed, _, _ = compose_circuits(enc1.to_stim_circuit(), enc2.to_stim_circuit(), wiring)
    return CliffordIsometry.from_stim_circuit(composed)
