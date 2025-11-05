# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Code Switching Compiler to find the minimum number of switches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from qiskit.converters import circuit_to_dag

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class CodeSwitchGraph:
    """A directed graph representation of a quantum circuit for code-switching analysis using min-cut / max-flow optimization.

    The graph is constructed such that:
      - Each quantum operation (T, H, CNOT) corresponds to one or more nodes.
      - Source (SRC) and sink (SNK) nodes represent two different codes:
          * Source-connected nodes (T, CNOT) → operations that can be done transversally in code A.
          * Sink-connected nodes (H, CNOT) → operations that can be done transversally in code B.
      - Infinite-capacity edges enforce code consistency between operations (e.g., CNOT links).
      - Finite-capacity edges (default 1.0) represent potential code transitions along qubit timelines.

    Attributes:
    ----------
    G : nx.DiGraph
        Directed graph storing the nodes and edges.
    source : str
        Identifier for the source node ("SRC").
    sink : str
        Identifier for the sink node ("SNK").
    """

    def __init__(self) -> None:
        """Initialize the CodeSwitchGraph with source and sink nodes."""
        self.G: nx.DiGraph = nx.DiGraph()
        self.source: str = "SRC"
        self.sink: str = "SNK"
        self.G.add_node(self.source)
        self.G.add_node(self.sink)

    def add_gate_node(self, gate_type: str, qubit: int, depth: int) -> str:
        """Add a node representing a quantum gate operation.

        Parameters
        ----------
        gate_type : str
            The gate type (e.g., "H", "T", "CNOTc", "CNOTt").
        qubit : int
            Index of the qubit the gate acts on.
        depth : int
            Depth (or layer index) of the operation in the circuit.

        Returns:
        -------
        str
            The unique node identifier created for this operation.
        """
        node_id = f"{gate_type}_q{qubit}_d{depth}"
        self.G.add_node(node_id, gate=gate_type, qubit=qubit, depth=depth)
        return node_id

    def add_edge_with_capacity(
        self, u: str, v: str, capacity: float, edge_type: str = "temporal", bidirectional: bool = True
    ) -> None:
        """Add a directed edge with specified capacity between two nodes.

        Parameters
        ----------
        u : str
            Source node identifier.
        v : str
            Target node identifier.
        capacity : float
            Edge capacity.
        """
        self.G.add_edge(u, v, capacity=capacity, edge_type=edge_type)
        if bidirectional:
            self.G.add_edge(v, u, capacity=capacity, edge_type=edge_type)

    def add_infinite_edge(self, u: str, v: str, bidirectional: bool = True) -> None:
        """Add an edge of infinite capacity between two nodes.

        Parameters
        ----------
        u : str
            Source node identifier.
        v : str
            Target node identifier.
        """
        self.add_edge_with_capacity(u, v, capacity=float("inf"), bidirectional=bidirectional)

    def add_regular_edge(self, u: str, v: str, capacity: float = 100.0, bidirectional: bool = True) -> None:
        """Add a regular (finite-capacity) directed edge.

        Parameters
        ----------
        u : str
            Source node identifier.
        v : str
            Target node identifier.
        capacity : float, optional
            Edge capacity (default is 1.0).
        """
        self.add_edge_with_capacity(u, v, capacity=capacity, edge_type="temporal", bidirectional=bidirectional)

    def add_bias_edges(self, node_id: str, biased_code: str = "SRC") -> None:
        """Add biased_code unary edges to the terminal nodes slightly preferring one code over the other.

        Parameters
        ----------
        biased_code : float
            Capacity of the biased_code edges to be added.
        """
        if biased_code == "SRC":
            self.add_edge_with_capacity(self.source, node_id, capacity=2.0, edge_type="unary")
            self.add_edge_with_capacity(self.sink, node_id, capacity=1.0, edge_type="unary")
        elif biased_code == "SNK":
            self.add_edge_with_capacity(self.source, node_id, capacity=1.0, edge_type="unary")
            self.add_edge_with_capacity(self.sink, node_id, capacity=2.0, edge_type="unary")

    def connect_to_code(self, node_id: str, gate_type: str) -> None:
        """Connect a gate node to the source and/or sink according to which code can perform the operation transversally.

        Note: Here we fix the convention that the 2D Color Code corresponds to the source (can perform H and CNOT)
        and the 3D Surface Code corresponds to the sink (can perform T and CNOT). This convention is arbitrary and can be swapped.
        However, swapping the convention requires to change the one-way transversal CNOT setting:
        2D Source + 3D Sink  <->  (infinite edge) Control -> Target
        3D Source + 2D Sink  <->  (infinite edge) Control <- Target

        Parameters
        ----------
        node_id : str
            Node identifier of the gate operation.
        gate_type : str
            Type of the gate (e.g., "H", "T", "CNOT").
        """
        # Source code can perform T and CNOT gates
        if gate_type == "T":
            self.add_infinite_edge(self.sink, node_id)
        # Sink code can perform H and CNOT gates
        if gate_type == "H":
            self.add_infinite_edge(node_id, self.source)

    def add_cnot_links(self, control_node: str, target_node: str, one_way_transversal_cnot: bool = False) -> None:
        """Add bidirectional infinite-capacity edges between two CNOT-related nodes to enforce that both qubits remain in the same code.

        Parameters
        ----------
        control_node : str
            Node representing the control qubit's CNOT operation.
        target_node : str
            Node representing the target qubit's CNOT operation.
        """
        self.add_infinite_edge(control_node, target_node, bidirectional=not (one_way_transversal_cnot))

    def build_from_qiskit(
        self, circuit: QuantumCircuit, one_way_transversal_cnot: bool = False, code_bias: bool = False
    ) -> None:
        """Construct the code-switch graph from a Qiskit QuantumCircuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The input quantum circuit containing H, T, and CX (CNOT) gates.

        Notes:
        -----
        - For each gate, a node is created per qubit.
        - Temporal ordering along qubit lines is maintained via regular edges.
        - CNOT gates create two linked nodes (control, target) with infinite capacity.
        """
        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())
        qubit_activity: dict[int, list[int]] = {q: [] for q in range(circuit.num_qubits)}
        qubit_last_node: list[str | None] = [None] * circuit.num_qubits

        for depth, layer in enumerate(layers):
            for node in layer["graph"].op_nodes():
                qubits = [circuit.find_bit(q).index for q in node.qargs]
                gate = node.name.upper()
                for qubit_index in qubits:
                    qubit_activity[qubit_index].append(depth)

                if gate in {"H", "T"}:
                    q = qubits[0]
                    gate_node = self.add_gate_node(gate, q, depth)
                    self.connect_to_code(gate_node, gate)
                    if qubit_last_node[q]:
                        self.add_regular_edge(qubit_last_node[q], gate_node)
                    qubit_last_node[q] = gate_node

                elif gate == "CX":
                    ctrl, tgt = qubits
                    node_ctrl = self.add_gate_node("CNOTc", ctrl, depth)
                    node_tgt = self.add_gate_node("CNOTt", tgt, depth)
                    self.add_cnot_links(node_ctrl, node_tgt, one_way_transversal_cnot=one_way_transversal_cnot)
                    if code_bias:
                        self.add_bias_edges(node_ctrl)
                        self.add_bias_edges(node_tgt)
                    for q, gate_node in [(ctrl, node_ctrl), (tgt, node_tgt)]:
                        if qubit_last_node[q]:
                            self.add_regular_edge(qubit_last_node[q], gate_node)
                        qubit_last_node[q] = gate_node

    def compute_min_cut(self, return_raw_data: bool = False) -> tuple[float, set[str], set[str]]:
        """Compute the minimum s-t cut between the source and sink.

        Returns:
        -------
        Tuple[float, Set[str], Set[str]]
            A tuple (cut_value, S, T) where:
              - cut_value is the total capacity of the minimum cut,
              - S is the set of nodes reachable from the source,
              - T is the complementary set of nodes.
        """
        cut_value, (S, T) = nx.minimum_cut(self.G, self.source, self.sink, capacity="capacity")  # noqa: N806
        num_switches = 0
        switch_cost = 0
        seen = set()
        for u, v, data in self.G.edges(data=True):
            if data["edge_type"] in {"temporal", "entangling"}:
                # needed to avoid double counting edges in undirected sense
                # BUG: must be repaired for one-way CNOTs (currently counts a valid edge as switch because we only check here if u and v are in different sets)
                key = tuple(sorted((u, v)))
                if key not in seen:
                    if (u in S and v in T) or (v in S and u in T):
                        num_switches += 1
                        switch_cost += data["capacity"]
                    seen.add(key)
        if return_raw_data:
            return cut_value, S, T
        return num_switches, S, T
