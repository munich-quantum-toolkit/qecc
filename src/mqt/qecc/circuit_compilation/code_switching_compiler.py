# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Code Switching Compiler to find the minimum number of switches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
from qiskit.converters import circuit_to_dag

from mqt.qecc.circuit_compilation.compilation_utils import parse_node_id

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@dataclass
class CompilerConfig:
    """Holds all configuration parameters for the CodeSwitchGraph."""

    edge_capacity_ratio: float = 0.001
    default_temporal_edge_capacity: float = 1.0
    switching_time: int = 2
    biased_code: str = "SRC"


class CodeSwitchGraph:
    """A directed graph representation of a quantum circuit for code-switching analysis using min-cut / max-flow optimization.

    The graph is constructed such that:
      - Each quantum operation (T, H, CNOT) corresponds to one or more nodes.
      - Source (SRC) and sink (SNK) nodes represent two different codes:
          * Source-connected nodes (H, CNOT) → operations that can be done transversally in a 2D Color Code.
          * Sink-connected nodes (T, CNOT) → operations that can be done transversally in a 3D Surface Code.
      - Infinite-capacity edges enforce code consistency between operations (e.g., CNOT links).
      - Finite-capacity (temporal) edges represent potential code transitions along qubit timelines.

    Attributes:
    ----------
    G : nx.DiGraph
        Directed graph storing the nodes and edges.
    source : str
        Identifier for the source node ("SRC").
    sink : str
        Identifier for the sink node ("SNK").
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        """Initialize the CodeSwitchGraph with source and sink nodes."""
        if config is None:
            self.config = CompilerConfig()
        else:
            self.config = config

        self.G: nx.DiGraph = nx.DiGraph()
        self.source: str = "SRC"
        self.sink: str = "SNK"
        self.G.add_node(self.source)
        self.G.add_node(self.sink)
        self.base_unary_capacity: float = self.config.default_temporal_edge_capacity * self.config.edge_capacity_ratio

    def _add_gate_node(self, gate_type: str, qubit: int, depth: int) -> str:
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

    def _add_edge_with_capacity(
        self, u: str, v: str, capacity: float, edge_type: str = "temporal", *, bidirectional: bool = True
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
        edge_type : str, optional
            Type of the edge (default is "temporal").
        bidirectional : bool, optional
            If True, add the reverse edge as well (default is True).
        """
        self.G.add_edge(u, v, capacity=capacity, edge_type=edge_type)
        if bidirectional:
            self.G.add_edge(v, u, capacity=capacity, edge_type=edge_type)

    def _add_infinite_edge(self, u: str, v: str, *, bidirectional: bool = True) -> None:
        """Add an edge of infinite capacity between two nodes. Possibly bidirectional.

        Parameters
        ----------
        u : str
            Source node identifier.
        v : str
            Target node identifier.
        bidirectional : bool, optional
            If True, add the reverse edge as well (default is True).
        """
        self._add_edge_with_capacity(u, v, capacity=float("inf"), edge_type="fixed", bidirectional=bidirectional)

    def _add_regular_edge(self, u: str, v: str, capacity: float | None = None, *, bidirectional: bool = True) -> None:
        """Add a regular (finite-capacity) directed edge.

        Parameters
        ----------
        u : str
            Source node identifier.
        v : str
            Target node identifier.
        capacity : float, optional
            Edge capacity.
        bidirectional : bool, optional
            If True, add the reverse edge as well (default is True).
        """
        if capacity is None:
            capacity = self.config.default_temporal_edge_capacity
        self._add_edge_with_capacity(u, v, capacity=capacity, edge_type="temporal", bidirectional=bidirectional)

    def _add_bias_edges(self, node_id: str, biased_code: str | None = None) -> None:
        """Add biased_code unary edges to the terminal nodes slightly preferring one code over the other.

        Parameters
        ----------
        biased_code : float
            Capacity of the biased_code edges to be added.
        """
        if biased_code is None:
            biased_code = self.config.biased_code
        if biased_code == "SRC":
            self._add_edge_with_capacity(
                self.source, node_id, capacity=2.0 * self.base_unary_capacity, edge_type="unary"
            )
            self._add_edge_with_capacity(self.sink, node_id, capacity=self.base_unary_capacity, edge_type="unary")
        elif biased_code == "SNK":
            self._add_edge_with_capacity(self.source, node_id, capacity=self.base_unary_capacity, edge_type="unary")
            self._add_edge_with_capacity(self.sink, node_id, capacity=2.0 * self.base_unary_capacity, edge_type="unary")

    def _connect_to_code(self, node_id: str, gate_type: str) -> None:
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
        # Sink code can perform T and CNOT gates
        if gate_type == "T":
            self._add_infinite_edge(self.sink, node_id)
        # Source code can perform H and CNOT gates
        if gate_type == "H":
            self._add_infinite_edge(node_id, self.source)

    def _add_cnot_links(self, control_node: str, target_node: str, *, one_way_transversal_cnot: bool = False) -> None:
        """Add bidirectional infinite-capacity edges between two CNOT-related nodes to enforce that both qubits remain in the same code.

        Parameters
        ----------
        control_node : str
            Node representing the control qubit's CNOT operation.
        target_node : str
            Node representing the target qubit's CNOT operation.
        one_way_transversal_cnot : bool, optional
            If True, allow transversal CNOTs from 3D (control) to 2D (target) besides the usual requirement that both qubits remain in the same code.
        """
        self._add_infinite_edge(control_node, target_node, bidirectional=not (one_way_transversal_cnot))

    def compute_idle_bonus(self, previous_depth: int, current_depth: int) -> float:
        """Compute a bonus (capacity reduction) for idling qubits.

        The idea: if a qubit has been idle for several depth layers,
        we reduce the capacity of the temporal edge between its last
        and next gate nodes to encourage reuse of that qubit.

        Parameters
        ----------
        previous_depth : int
            The depth index of the previous active gate on the qubit.
        current_depth : int
            The depth index of the current active gate on the qubit.

        Returns:
        -------
        float
            The capacity reduction (bonus) to apply. Can be tuned by formula.
        """
        idle_length = max(0, current_depth - previous_depth - 1)

        if idle_length <= self.config.switching_time:
            idle_length = 0

        max_bonus = 5
        return min(max_bonus, idle_length) * self.base_unary_capacity

    def _edge_capacity_with_idle_bonus(self, depths: list[int], base_capacity: float | None = None) -> float:
        """Compute the effective temporal edge capacity.

        Optionally reduced by an idle bonus if the qubit has been inactive for several layers.

        Parameters
        ----------
        depths : list[int]
            The ordered list of depth indices for a given qubit's gates.
        base_capacity : float, optional
            The default temporal edge capacity.

        Returns:
        -------
        float
            The adjusted edge capacity.
        """
        if base_capacity is None:
            base_capacity = self.config.default_temporal_edge_capacity
        if len(depths) < 2:
            return base_capacity

        prev_depth, curr_depth = depths[-2], depths[-1]
        bonus = self.compute_idle_bonus(prev_depth, curr_depth)
        return base_capacity - bonus

    def build_from_qiskit(
        self,
        circuit: QuantumCircuit,
        *,
        one_way_transversal_cnot: bool = False,
        code_bias: bool = False,
        idle_bonus: bool = False,
    ) -> None:
        """Construct the code-switch graph from a Qiskit QuantumCircuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The input quantum circuit containing H, T, and CX (CNOT) gates.
        one_way_transversal_cnot : bool, optional
            If True, restrict transversal CNOTs to one direction.
        code_bias : bool, optional
            If True, add bias edges for CNOT nodes.
        idle_bonus : bool, optional
            If True, reduce temporal edge capacities based on idle durations via
            `_edge_capacity_with_idle_bonus`. Default is False.

        Notes:
        -----
        - For each gate, a node is created per qubit.
        - Temporal ordering along qubit lines is maintained via regular edges.
        - CNOT gates create two linked nodes (control, target) with infinite capacity.
        - Optionally adds code bias edges or one-way transversal CNOT constraints.
        - Idle bonuses reduce temporal edge capacities for qubits idle over multiple layers.
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
                    gate_node = self._add_gate_node(gate, q, depth)
                    self._connect_to_code(gate_node, gate)
                    if qubit_last_node[q]:
                        self._add_regular_edge(qubit_last_node[q], gate_node)
                    qubit_last_node[q] = gate_node

                elif gate == "CX":
                    ctrl, tgt = qubits
                    node_ctrl = self._add_gate_node("CNOTc", ctrl, depth)
                    node_tgt = self._add_gate_node("CNOTt", tgt, depth)
                    self._add_cnot_links(node_ctrl, node_tgt, one_way_transversal_cnot=one_way_transversal_cnot)
                    if code_bias:
                        self._add_bias_edges(node_ctrl)
                        self._add_bias_edges(node_tgt)
                    for q, gate_node in [(ctrl, node_ctrl), (tgt, node_tgt)]:
                        if qubit_last_node[q]:
                            capacity = (
                                self._edge_capacity_with_idle_bonus(qubit_activity[q])
                                if idle_bonus
                                else self.config.default_temporal_edge_capacity
                            )
                            self._add_regular_edge(qubit_last_node[q], gate_node, capacity=capacity)
                        qubit_last_node[q] = gate_node

    def compute_min_cut(self) -> tuple[int, list[tuple[int, int]], set[str], set[str]]:
        """Compute the minimum s-t cut between the source and sink.

        Returns:
        -------
        tuple[int, list[tuple[int, int]], set[str], set[str]]
            A tuple (num_switches, switch_positions, S, T) where:
              - num_switches is the count of temporal edges crossing the cut (number of code switches),
              - switch_positions is a list of (qubit, depth) pairs where switches occur,
              - S is the set of nodes reachable from the source,
              - T is the complementary set of nodes.
        """
        _, (S, T) = nx.minimum_cut(self.G, self.source, self.sink, capacity="capacity")  # noqa: N806
        num_switches, switch_positions = self._extract_switch_locations(S, T)
        return num_switches, switch_positions, S, T

    def _extract_switch_locations(self, S: set[str], T: set[str]) -> tuple[int, list[tuple[int, int]]]:  # noqa: N803
        """Return a list of (qubit, depth) pairs where switches should be inserted.

        Parameters:
        ----------
        S : set[str]
            Set of nodes reachable from the source after min-cut.
        T : set[str]
            Complementary set of nodes after min-cut.

        Returns:
        -------
        Tuple[int, List[Tuple[int, int]]]
            A tuple (num_switches, switch_positions) where:
              - num_switches is the total number of switches detected,
              - switch_positions is a list of (qubit, depth) pairs indicating where switches occur.
                Here, 'qubit' is the qubit index and 'depth' is the temporal position of the gate in terms of number of gates per qubit.
                So, a depth of 3 means that this is the 3rd single qubit gate position on that qubit line.
                That means a switch should be inserted just after that depth layer on that qubit.
        """
        switch_positions = []
        seen = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("edge_type") == "temporal":
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                if (u in S and v in T) or (v in S and u in T):
                    # We can take e.g. the 'earlier' node in time as the insertion point
                    qubit, depth = parse_node_id(u)
                    switch_positions.append((qubit, depth))
        return len(switch_positions), switch_positions
