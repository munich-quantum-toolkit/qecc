# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helper functions for DAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, TGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

pos = tuple[int, int]


def count_cx_gates_per_layer(dag: DAGCircuit) -> list[int]:
    """Count CX gates per DAG layer.

    If a layer has no CX layers, it is skipped.
    """
    layer_counts = []
    for layer in dag.layers():
        graph = layer["graph"]
        ops = [op for op in graph.op_nodes() if op.name == "cx"]
        if len(ops) > 0:
            layer_counts.append(len(ops))
    return layer_counts


def pairs_into_dag_agnostic(pairs: list[tuple[int, int] | int], q: int) -> DAGCircuit:
    """Assumes tuples of int, and int values (i.e. agnostic of layout)."""
    circ = QuantumCircuit(q)
    for pair in pairs:
        if isinstance(pair, int):
            circ.t(pair)
        elif isinstance(pair, tuple):
            circ.cx(pair[0], pair[1])
        else:
            msg = f"Expected int or tuple[int,int] got {pair!r} ({type(pair).__name__})"  # type: ignore[unreachable]
            raise TypeError(msg)
    # dag
    return circuit_to_dag(circ)


def terminal_pairs_into_dag(terminal_pairs: list[tuple[pos, pos] | pos], layout: dict[int, pos]) -> DAGCircuit:
    """Given (a subset of) terminal pairs used in TeleportationRouter translate it back to integer qubit labels using layout and transform it into its dag.

    Args:
        terminal_pairs (list[tuple[pos, pos]  |  pos]): A list of gates in terms where quibt labels are their positions (type pos) on the grid. 'pos' is a T gate and 'tuple[pos,pos]' a CNOT gate.
        layout (dict[int, pos]): keys are integer logical qubit labels and values are the `pos` values on the respective logical qubit on the grid.

    Returns:
        DAGCircuit: DAG of the circuit.
    """
    # translate into integers
    layout_rev: dict[pos, int] = {j: i for i, j in layout.items()}
    terminal_pairs_trans: list[int | tuple[int, int]] = []
    for pair in terminal_pairs:
        if isinstance(pair[0], int):  # t gate
            terminal_pairs_trans.append(layout_rev[pair])
        elif isinstance(pair[0], tuple):  # cnot gate
            terminal_pairs_trans.append((layout_rev[pair[0]], layout_rev[pair[1]]))
        else:
            msg = "other gate types not implemented yet"  # type: ignore[unreachable]
            raise NotImplementedError(msg)
    # create qiskit circuit instance
    circ = QuantumCircuit(len(layout))
    for pair2 in terminal_pairs_trans:
        if isinstance(pair2, int):
            circ.t(pair2)
        elif isinstance(pair2, tuple):
            circ.cx(pair2[0], pair2[1])
    # dag
    return circuit_to_dag(circ)


def extract_layer_from_dag_agnostic(dag: DAGCircuit, layer: int) -> list[tuple[int, int] | int]:
    """Extracts a layer from the dag without assuming a layout. i.e. returns tuple[int,int] and int."""
    layers = list(dag.layers())
    if layer >= len(layers):
        msg = f"layer {layer} is larger than the number of layers."
        raise ValueError(msg)
    chosen = layers[layer]["graph"]
    circuit = dag_to_circuit(chosen)
    terminal_pairs_trans: list[int | tuple[int, int]] = []
    for ci in circuit.data:
        instr = ci.operation
        qargs = ci.qubits
        idx = [circuit.find_bit(q).index for q in qargs]
        if instr.name == "cx":
            terminal_pairs_trans.append(cast("tuple[int,int]", (idx[0], idx[1])))
        elif instr.name == "t":
            terminal_pairs_trans.append(cast("int", idx[0]))
        else:
            msg = "Gates other than T or CNOT are not implemented yet"
            raise NotImplementedError(msg)
    return terminal_pairs_trans


def extract_layer_from_dag(dag: DAGCircuit, layout: dict[int, pos], layer: int) -> list[tuple[pos, pos] | pos]:
    """Extracts a layer from the dag and translates it into a terminal pairs list depending on the given layout."""
    layers = dag.layers()
    chosen = list(layers)[layer]["graph"]
    circuit = dag_to_circuit(chosen)
    # translate the qiskit circuit into a terminal pairs trans structure
    terminal_pairs_trans: list[tuple[int, int] | int] = []
    for ci in circuit.data:
        instr = ci.operation
        qargs = ci.qubits
        idx = [circuit.find_bit(q).index for q in qargs]
        if instr.name == "cx":
            terminal_pairs_trans.append((idx[0], idx[1]))
        elif instr.name == "t":
            terminal_pairs_trans.append(idx[0])
        else:
            msg = "Gates other than T or CNOT are not implemented yet"
            raise NotImplementedError(msg)
    # translate into the given layout
    terminal_pairs: list[tuple[pos, pos] | pos] = []
    for pair in terminal_pairs_trans:
        if isinstance(pair, int):
            terminal_pairs.append(layout[pair])
        elif isinstance(pair, tuple):
            terminal_pairs.append((layout[pair[0]], layout[pair[1]]))
        else:
            msg = "no other gates implemented yet despite T and CNOT"  # type: ignore[unreachable]
            raise NotImplementedError(msg)

    return terminal_pairs


def push_remainder_into_layers_dag(
    dag: DAGCircuit,
    terminal_pairs_remainder: list[tuple[pos, pos] | pos],
    layout: dict[int, pos],
    current_layer: list[tuple[pos, pos] | pos],
) -> tuple[list[list[tuple[pos, pos] | pos]], DAGCircuit]:
    """This step could be avoided in principle but i do not want to change the structure of utils too much.

    Take the dag, remove the very first layer and add terminal pairs remainder as operationis in front.
    """
    # remove layer 0
    first_layer = next(iter(dag.layers()))["graph"]
    nodes_to_remove = list(first_layer.op_nodes())
    assert len(nodes_to_remove) == len(current_layer), (
        "mismatch between current_layer and the 0th layer extracted in helper"
    )
    layers = list(dag.layers())
    layers.pop(0)
    new_dag = dag.copy_empty_like()
    for layer in list(dag.layers())[
        1:
    ]:  # the [1:] appears redundant, but without it, the overall routing ends up in an endless loop.
        new_dag.compose(layer["graph"], inplace=True)
    dag = new_dag
    # translate terminal_pairs_remainder
    layout_rev: dict[pos, int] = {j: i for i, j in layout.items()}
    # translate and add to dag in front
    for pair in terminal_pairs_remainder:
        if isinstance(pair[0], tuple):  # cnot
            gate = (layout_rev[pair[0]], layout_rev[pair[1]])
            dag.apply_operation_front(CXGate(), qargs=[dag.qubits[gate[0]], dag.qubits[gate[1]]], cargs=[])
        elif isinstance(pair[0], int):  # t gate
            gate2 = layout_rev[pair]
            dag.apply_operation_front(TGate(), qargs=[dag.qubits[gate2]], cargs=[])
        else:
            msg = "other gates than t and cnot not implemented yet"  # type: ignore[unreachable]
            raise NotImplementedError(msg)

    # layers updated as list with layout dependency
    layers_updated = [extract_layer_from_dag(dag, layout, i) for i in range(len(list(dag.layers())))]

    return layers_updated, dag
