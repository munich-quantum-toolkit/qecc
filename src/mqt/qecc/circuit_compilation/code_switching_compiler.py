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

    def add_infinite_edge(self, u: str, v: str) -> None:
        """Add an edge of infinite capacity between two nodes.

        Parameters
        ----------
        u : str
            Source node identifier.
        v : str
            Target node identifier.
        """
        self.G.add_edge(u, v, capacity=float("inf"))
        self.G.add_edge(v, u, capacity=float("inf"))

    def add_regular_edge(self, u: str, v: str, capacity: float = 1.0) -> None:
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
        self.G.add_edge(u, v, capacity=capacity)
        self.G.add_edge(v, u, capacity=capacity)

    def connect_to_code(self, node_id: str, gate_type: str) -> None:
        """Connect a gate node to the source and/or sink according to which code can perform the operation transversally.

        Parameters
        ----------
        node_id : str
            Node identifier of the gate operation.
        gate_type : str
            Type of the gate (e.g., "H", "T", "CNOT").
        """
        # Source code can perform T and CNOT gates
        if gate_type == "T":
            self.add_infinite_edge(self.source, node_id)
        # Sink code can perform H and CNOT gates
        if gate_type == "H":
            self.add_infinite_edge(node_id, self.sink)

    def add_cnot_links(self, control_node: str, target_node: str) -> None:
        """Add bidirectional infinite-capacity edges between two CNOT-related nodes to enforce that both qubits remain in the same code.

        Parameters
        ----------
        control_node : str
            Node representing the control qubit's CNOT operation.
        target_node : str
            Node representing the target qubit's CNOT operation.
        """
        self.add_infinite_edge(control_node, target_node)
        self.add_infinite_edge(target_node, control_node)

    def build_from_qiskit(self, circuit: QuantumCircuit) -> None:
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
        qubit_last_node: list[str | None] = [None] * circuit.num_qubits
        depth = 0

        for depth, instr in enumerate(circuit.data):
            gate = instr.operation.name.upper()
            qubits = [circuit.find_bit(q).index for q in instr.qubits]

            if gate in {"H", "T"}:
                q = qubits[0]
                node = self.add_gate_node(gate, q, depth)
                self.connect_to_code(node, gate)
                if qubit_last_node[q]:
                    self.add_regular_edge(qubit_last_node[q], node)
                qubit_last_node[q] = node

            elif gate == "CX":
                ctrl, tgt = qubits
                node_ctrl = self.add_gate_node("CNOTc", ctrl, depth)
                node_tgt = self.add_gate_node("CNOTt", tgt, depth)
                self.add_cnot_links(node_ctrl, node_tgt)
                for q, node in [(ctrl, node_ctrl), (tgt, node_tgt)]:
                    if qubit_last_node[q]:
                        self.add_regular_edge(qubit_last_node[q], node)
                    qubit_last_node[q] = node

    def compute_min_cut(self) -> tuple[float, set[str], set[str]]:
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
        return cut_value, S, T
