# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility methods for circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from stim import Circuit, CircuitInstruction, CircuitRepeatBlock

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

STIM_MEASUREMENTS = {"MR", "MRX", "MRY", "MRZ"}


def get_stim_qubit_values(operation: CircuitInstruction | CircuitRepeatBlock) -> list[int]:
    """Get the qubit values from a stim operation."""
    if isinstance(operation, CircuitRepeatBlock):
        raise NotImplementedError

    qubits = [target.qubit_value for target in operation.targets_copy()]
    assert all(qubit is not None for qubit in qubits)
    return cast("list[int]", qubits)


def relabel_qubits(circ: Circuit, qubit_mapping: dict[int, int] | int) -> Circuit:
    """Relabels the qubits in a stim circuit based on the given mapping.

    Parameters:
        circ: The original stim circuit.
        qubit_mapping: Either a dictionary mapping original qubit indices to new qubit indices or a constant offset to add to all qubit indices.

    Returns:
        A new stim circuit with qubits relabeled.
    """
    new_circ = Circuit()
    for op in circ:
        assert isinstance(op, CircuitInstruction)
        if isinstance(qubit_mapping, dict):
            relabelled_qubits = [qubit_mapping[q.value] for q in op.targets_copy()]
        else:
            relabelled_qubits = [q.value + qubit_mapping for q in op.targets_copy()]
        new_circ.append(op.name, relabelled_qubits)
    return new_circ


def qiskit_to_stim_circuit(qc: QuantumCircuit) -> Circuit:
    """Convert a Qiskit circuit to a Stim circuit."""
    single_qubit_gate_map = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "S_DAG",
        "sx": "SQRT_X",
        "measure": "MR",
        "reset": "R",
    }
    stim_circuit = Circuit()
    for gate in qc:
        op = gate.operation.name
        qubit = qc.find_bit(gate.qubits[0])[0]
        if op in single_qubit_gate_map:
            stim_circuit.append(single_qubit_gate_map[op], [qubit])
        elif op == "cx":
            target = qc.find_bit(gate.qubits[1])[0]
            stim_circuit.append("CX", [qubit, target])
        elif op == "barrier":
            stim_circuit.append("TICK")  # ty: ignore[invalid-argument-type]
        else:
            msg = f"Unsupported gate: {op}"
            raise ValueError(msg)
    return stim_circuit


def compact_stim_circuit(circ: Circuit, scheduling_method: str = "asap") -> Circuit:
    """Move circuit instructions to the front and ignore TICKS.

    Args:
        circ: stim circuit to compact
        scheduling_method: Either "asap" (as soon as possible) or "alap" (as late as possible).

    Returns:
         A compacted stim circuit.
    """
    compact_circ = Circuit()
    for layer in collect_circuit_layers(circ, scheduling_method):
        compact_circ += layer
    return compact_circ


def compose_compact_stim_circuits(circs: list[Circuit], align: str = "start") -> Circuit:
    """Compose and compact multiple stim circuits.

    Args:
        circs: List of stim circuits to compose and compact.
        align: Either "start" (align at the start) or "end" (align at the end).

    Returns:
        A composed and compacted stim circuit.
    """
    if align not in {"start", "end"}:
        msg = "align must be 'start' or 'end'."
        raise ValueError(msg)

    composed_layers: list[Circuit] = []
    for circ in circs:
        layers = collect_circuit_layers(circ, "asap" if align == "start" else "alap")
        if align == "end":
            layers = layers[::-1]
        for i, layer_circ in enumerate(layers):
            if i < len(composed_layers):
                composed_layers[i] += layer_circ
            else:
                composed_layers.append(layer_circ)

    if align == "end":
        composed_layers.reverse()

    compose_circ = Circuit()
    for layer in composed_layers:
        compose_circ += layer
    return compose_circ


def collect_circuit_layers(circ: Circuit, scheduling_method: str = "asap") -> list[Circuit]:
    """Collect all layers that can be executed in parallel.

    Args:
        circ: Stim circuit to process.
        scheduling_method: Either "asap" (as soon as possible) or "alap" (as late as possible).

    Returns:
        list of circuit layers. All instructions in one layer can be executed in parallel. It holds that circ=sum(collect_circuit_layers(circ)).
    """
    if scheduling_method not in {"asap", "alap"}:
        msg = "scheduling_method must be 'asap' or 'alap'."
        raise ValueError(msg)

    # Copy the circuit and separate all instructions by ticks
    circ_copy = Circuit()
    for instr in circ:
        assert isinstance(instr, CircuitInstruction)
        for grp in instr.target_groups():
            qubits = [q.qubit_value for q in grp]
            assert all(qubit is not None for qubit in qubits)
            circ_copy.append(instr.name, cast("list[int]", qubits))
            circ_copy.append("TICK", [])

    if scheduling_method == "alap":
        circ_copy = circ_copy[::-1]  # Reverse the circuit for ALAP scheduling

    # Now work with the copied circuit
    circ = circ_copy
    n_qubits = circ.num_qubits
    layers = []

    while len(circ) > 0:
        layer = Circuit()
        qubit_layer_used = [False] * n_qubits  # Track used qubits in this layer
        instr_to_delete = []  # Track instructions to delete after adding them to the layer
        idx = 0

        while idx < len(circ):
            instr = circ[idx]

            # Skip TICK instructions
            while instr is not None and instr.name == "TICK" and idx < len(circ):
                circ.pop(idx)
                instr = circ[idx] if idx < len(circ) else None

            if instr is None:  # No more instructions to process
                break

            qubits = get_stim_qubit_values(instr)

            # Check if any qubit from this instruction is already used in the layer
            if not any(qubit_layer_used[q] for q in qubits):
                layer.append(instr.name, qubits)
                instr_to_delete.append(idx)  # Mark this instruction for removal

            # Mark the qubits used in this instruction
            for q in qubits:
                qubit_layer_used[q] = True

            idx += 1

        # Add the layer to the list
        layers.append(layer)

        # Remove the instructions that were added to the layer
        for n_deleted, gate_idx in enumerate(instr_to_delete):
            circ.pop(gate_idx - n_deleted)

    if scheduling_method == "alap":
        layers.reverse()  # Reverse the layers back for ALAP scheduling

    return layers


def depth(circ: Circuit) -> int:
    """Calculate the depth of a stim circuit.

    Args:
        circ: The stim circuit to analyze.

    Returns:
        The depth of the circuit.
    """
    return len(collect_circuit_layers(circ, scheduling_method="asap"))


def remove_single_qubit_gates(circ: Circuit) -> Circuit:
    """Remove all single-qubit gates from a stim circuit.

    Args:
        circ: The stim circuit to filter.

    Returns:
        A new stim circuit with single-qubit gates removed.
    """
    new_circ = Circuit()
    for op in circ:
        assert isinstance(op, CircuitInstruction)
        if all(len(grp) == 1 for grp in op.target_groups()):
            continue
        new_circ.append(op.name, get_stim_qubit_values(op))
    return new_circ


def remove_swap_gates(circ: Circuit) -> Circuit:
    """Remove all SWAP gates from a stim circuit.

    Args:
        circ: The stim circuit to filter.

    Returns:
        A new stim circuit with SWAP gates removed.
    """
    new_circ = Circuit()
    for op in circ:
        if op.name == "SWAP":
            continue
        new_circ.append(op.name, get_stim_qubit_values(op))
    return new_circ


def two_qubit_gate_depth(circ: Circuit, *, count_swaps: bool = False) -> int:
    """Calculate the two-qubit gate depth of a stim circuit.

    Args:
        circ: The stim circuit to analyse.
        count_swaps: If ``True``, SWAP gates are included in the depth
            calculation. Defaults to ``False``.

    Returns:
        The two-qubit gate depth of the circuit.
    """
    circ = remove_single_qubit_gates(circ)
    if not count_swaps:
        circ = remove_swap_gates(circ)
    return depth(circ)


def unmeasured_qubits(circ: Circuit) -> list[int]:
    """Return a list of qubits that are not measured in circ."""
    measured_qubits: set[int] = set()

    for instr in circ:
        if instr.name in STIM_MEASUREMENTS:
            measured_qubits.update(get_stim_qubit_values(instr))

    all_qubits = set(range(circ.num_qubits))
    return list(all_qubits - measured_qubits)


def measured_qubits(circ: Circuit) -> list[int]:
    """Return a list of qubits that are measured in circ.

    The qubits are in the ordered according to when they are measured.
    """
    measured_qubits: list[int] = []

    for instr in circ:
        if instr.name in STIM_MEASUREMENTS:
            measured_qubits.extend(get_stim_qubit_values(instr))

    return measured_qubits


def compose_circuits(
    circ1: Circuit, circ2: Circuit, wiring: dict[int, int] | None = None
) -> tuple[Circuit, dict[int, int], dict[int, int]]:
    """Compose two Stim circuits.

    The circuits are composed only along the qubits that are connected by the `wiring` dict.
    All other qubits are assumed to be unconnected.
    If wire is None, then the circuits are simply vertically stacked.

    Args:
        circ1: The first stim circuit.
        circ2: The second stim circuit.
        wiring: Optional dict mapping outputs of `circ1` to inputs of `circ2`.

    Returns:
        A tuple containing the composed stim circuit and two mappings:
        - mapping1: Maps qubits of circ1 to the composed circuit.
        - mapping2: Maps qubits of circ2 to the composed circuit.
    """
    if wiring is None:
        wiring = {}

    connected = wiring.keys()
    non_connected_circ1 = set(range(circ1.num_qubits)) - set(connected)
    non_connected_circ2 = set(range(circ2.num_qubits)) - set(wiring.values())
    # map non-connected of circ1 to the first n_connected qubits
    non_connected_mapping1 = {q: i for i, q in enumerate(non_connected_circ1)}
    # map non-connected of circ 2 to the qubits n_connected...n_connected + len(circ2)-1
    non_connected_mapping2 = {q: i + len(non_connected_circ1) for i, q in enumerate(non_connected_circ2)}
    # map connected qubits to the last n_connected qubits
    connected_mapping1 = {q: i + len(non_connected_circ1) + len(non_connected_circ2) for i, q in enumerate(connected)}
    connected_mapping2 = {
        wiring[q]: i + len(non_connected_circ1) + len(non_connected_circ2) for i, q in enumerate(connected)
    }

    mapping1 = {**non_connected_mapping1, **connected_mapping1}
    mapping2 = {**non_connected_mapping2, **connected_mapping2}

    composed = circ1.copy()
    composed = relabel_qubits(composed, mapping1)
    circ2_relabelled = relabel_qubits(circ2, mapping2)
    composed += circ2_relabelled
    return composed, mapping1, mapping2


def num_two_qubit_gates(circ: Circuit, *, count_swaps: bool = False) -> int:
    """Return the number of two-qubit gates in a stim circuit.

    Args:
        circ: The stim circuit to analyse.
        count_swaps: If ``True``, SWAP gates are counted as two-qubit gates.
            Defaults to ``False``.

    Returns:
        The number of two-qubit gates.
    """
    num_tqg = 0
    for op in circ:
        assert isinstance(op, CircuitInstruction)
        if op.name == "SWAP" and not count_swaps:
            continue
        for grp in op.target_groups():
            if len(grp) == 2:
                num_tqg += 1

    return num_tqg
