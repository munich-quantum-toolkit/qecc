from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library import TGate, CXGate
from qiskit.dagcircuit import DAGCircuit
import random


def count_cx_gates_per_layer(dag):
    """
    Count CX gates per DAG layer.
    """
    layer_counts = []
    for layer in dag.layers():
        graph = layer["graph"]
        ops = [op for op in graph.op_nodes() if op.name == "cx"]
        if len(ops) > 0:
            layer_counts.append(len(ops))
    return layer_counts


def pairs_into_dag_agnostic(pairs, q):
    """assumes tuples of int, and int values (i.e. agnostic of layout)"""
    circ = QuantumCircuit(q)
    for pair in pairs:
        if isinstance(pair, int):
            circ.t(pair)
        elif isinstance(pair, tuple):
            circ.cx(pair[0], pair[1])
    # dag
    dag = circuit_to_dag(circ)
    return dag


def terminal_pairs_into_dag(terminal_pairs, layout):
    """given (a subset of) terminal pairs used in LookaheadQuilting translate it back to integer qubit labels using layout and transform it into its dag"""
    # translate into integers
    layout_rev = {j: i for i, j in layout.items()}
    terminal_pairs_trans = []
    for pair in terminal_pairs:
        if isinstance(pair[0], int):
            terminal_pairs_trans.append(layout_rev[pair])
        elif isinstance(pair[0], tuple):
            terminal_pairs_trans.append((layout_rev[pair[0]], layout_rev[pair[1]]))
        else:
            raise NotImplementedError("other gate types not implemented yet")
    # create qiskit circuit instance
    circ = QuantumCircuit(len(layout))
    for pair in terminal_pairs_trans:
        if isinstance(pair, int):
            circ.t(pair)
        elif isinstance(pair, tuple):
            circ.cx(pair[0], pair[1])
    # dag
    dag = circuit_to_dag(circ)
    return dag


def extract_layer_from_dag_agnostic(dag, layer: int):
    """extracts a layer from the dag without assuming a layout. i.e. returns tuple[int,int] and int"""
    layers = dag.layers()
    chosen = list(layers)[layer]["graph"]
    circuit = dag_to_circuit(chosen)
    terminal_pairs_trans = []
    for ci in circuit.data:
        instr = ci.operation
        qargs = ci.qubits
        idx = [circuit.find_bit(q).index for q in qargs]
        if instr.name == "cx":
            terminal_pairs_trans.append((idx[0], idx[1]))
        elif instr.name == "t":
            terminal_pairs_trans.append(idx[0])
        else:
            raise NotImplementedError(
                "Gates other than T or CNOT are not implemented yet"
            )
    return terminal_pairs_trans


def extract_layer_from_dag(dag, layout, layer: int):
    """extracts a layer from the dag and translates it into a terminal pairs list depending on the given layout"""
    layers = dag.layers()
    chosen = list(layers)[layer]["graph"]
    circuit = dag_to_circuit(chosen)
    # translate the qiskit circuit into a terminal pairs trans structure
    terminal_pairs_trans = []
    for ci in circuit.data:
        instr = ci.operation
        qargs = ci.qubits
        idx = [circuit.find_bit(q).index for q in qargs]
        if instr.name == "cx":
            terminal_pairs_trans.append((idx[0], idx[1]))
        elif instr.name == "t":
            terminal_pairs_trans.append(idx[0])
        else:
            raise NotImplementedError(
                "Gates other than T or CNOT are not implemented yet"
            )
    # translate into the given layout
    terminal_pairs = []
    for pair in terminal_pairs_trans:
        if isinstance(pair, int):
            terminal_pairs.append(layout[pair])
        elif isinstance(pair, tuple):
            terminal_pairs.append((layout[pair[0]], layout[pair[1]]))
        else:
            raise NotImplementedError(
                "no other gates implemented yet despite T and CNOT"
            )

    return terminal_pairs


def push_remainder_into_layers_dag(
    dag, terminal_pairs_remainder, layout, current_layer
):
    """
    this step could be avoided in principle but i do not want to change the structure of utils too much
    take the dag, remove the very first layer and add terminal pairs remainder as operationis in front
    """
    # remove layer 0
    first_layer = list(dag.layers())[0]["graph"]
    # print(extract_layer_from_dag(dag, layout, 0))
    # print("current layer", current_layer)
    nodes_to_remove = list(first_layer.op_nodes())
    assert len(nodes_to_remove) == len(
        current_layer
    ), "mismatch between current_layer and the 0th layer extracted in helper"
    layers = list(dag.layers())
    layers.pop(0)
    new_dag = dag.copy_empty_like()
    for layer in list(dag.layers())[1:]:
        new_dag.compose(layer["graph"], inplace=True)
    dag = new_dag
    # translate terminal_pairs_remainder
    layout_rev = {j: i for i, j in layout.items()}
    # translate and add to dag in front
    for pair in terminal_pairs_remainder:
        if isinstance(pair[0], tuple):
            gate = (layout_rev[pair[0]], layout_rev[pair[1]])
            dag.apply_operation_front(
                CXGate(), qargs=[dag.qubits[gate[0]], dag.qubits[gate[1]]], cargs=[]
            )
        elif isinstance(pair[0], int):
            gate = layout_rev[pair]
            dag.apply_operation_front(TGate(), qargs=[dag.qubits[gate]], cargs=[])
        else:
            raise NotImplementedError("other gates than t and cnot not implemented yet")

    # layers updated as list with layout dependency
    layers_updated = []
    for i in range(len(list(dag.layers()))):
        layers_updated.append(extract_layer_from_dag(dag, layout, i))

    return layers_updated, dag
