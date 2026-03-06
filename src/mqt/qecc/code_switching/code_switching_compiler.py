# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Code Switching Compiler to find the minimum number of switches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
from qiskit.converters import circuit_to_dag

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.dagcircuit import DAGOpNode

pattern = re.compile(r".*_q(\d+)_d(\d+)")


@dataclass
class CompilerConfig:
    """Holds all configuration parameters for the MinimalCodeSwitchingCompiler."""

    edge_capacity_ratio: float = 0.001
    default_temporal_edge_capacity: float = 1.0
    switching_time: int = 2
    biased_code: str = "SRC"


class MinimalCodeSwitchingCompiler:
    """A directed graph representation of a quantum circuit for code-switching analysis using min-cut / max-flow optimization.

    The graph is constructed such that:
      - Each quantum operation (e.g. T, H, CNOT) corresponds to one or more nodes.
      - Source (SRC) and sink (SNK) nodes represent two different codes:
          * Source-connected nodes (e.g. H, CNOT) → operations that can be done transversally in a 2D Color Code.
          * Sink-connected nodes (e.g. T, CNOT) → operations that can be done transversally in a 3D Surface Code.
      - Infinite-capacity edges enforce code consistency between operations (e.g., CNOT links).
      - Finite-capacity (temporal) edges represent potential code transitions along qubit timelines.

    Attributes:
    ----------
    gate_set_source : set[str]
        Set of gate types supported by code A (e.g., 2D Color Code).
        This gate set is associated with the source node.
        Should be provided as a set of strings, e.g., {"H", "CNOT"}.
    gate_set_sink : set[str]
        Set of gate types supported by code B (e.g., 3D Surface Code).
        This gate set is associated with the sink node.
        Should be provided as a set of strings, e.g., {"T", "CNOT"}.
    common_gates : set[str]
        Set of gate types that can be performed transversally in both codes.
    config : CompilerConfig
        Configuration parameters for the compiler.
    G : nx.DiGraph
        Directed graph storing the nodes and edges.
    source : str
        Identifier for the source node ("SRC").
    sink : str
        Identifier for the sink node ("SNK").
    """

    def __init__(
        self, gate_set_code_source: set[str], gate_set_code_sink: set[str], config: CompilerConfig | None = None
    ) -> None:
        """Initialize the MinimalCodeSwitchingCompiler with source and sink nodes."""
        if config is None:
            self.config = CompilerConfig()
        else:
            self.config = config

        self.gate_set_source = gate_set_code_source
        self.gate_set_sink = gate_set_code_sink
        self.common_gates = self._get_common_gates()
        self.G: nx.DiGraph = nx.DiGraph()
        self.source: str = "SRC"
        self.sink: str = "SNK"
        self.G.add_node(self.source)
        self.G.add_node(self.sink)
        self.base_unary_capacity: float = self.config.default_temporal_edge_capacity * self.config.edge_capacity_ratio

    def _get_common_gates(self) -> set[str]:
        """Return the set of gates that can be performed transversally in both codes."""
        return self.gate_set_source.intersection(self.gate_set_sink)

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
        self,
        u: str,
        v: str,
        capacity: float,
        edge_type: str = "temporal",
        *,
        bidirectional: bool = True,
        one_way_is_snk2src: bool = True,
    ) -> None:
        """Add a directed edge with specified capacity between two nodes.

        Parameters
        ----------
        u : str
            Control node identifier.
        v : str
            Target node identifier.
        capacity : float
            Edge capacity.
        edge_type : str, optional
            Type of the edge (default is "temporal").
        bidirectional : bool, optional
            If True, add the reverse edge as well (default is True).
        """
        if not one_way_is_snk2src:
            u, v = v, u
        self.G.add_edge(u, v, capacity=capacity, edge_type=edge_type)
        if bidirectional:
            self.G.add_edge(v, u, capacity=capacity, edge_type=edge_type)

    def _add_infinite_edge(
        self, u: str, v: str, *, bidirectional: bool = True, one_way_is_snk2src: bool | None = None
    ) -> None:
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
        if one_way_is_snk2src is None:
            one_way_is_snk2src = True
        self._add_edge_with_capacity(
            u,
            v,
            capacity=float("inf"),
            edge_type="fixed",
            bidirectional=bidirectional,
            one_way_is_snk2src=one_way_is_snk2src,
        )

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
        """Add biased_code bias edges to the terminal nodes slightly preferring one code over the other.

        Parameters
        ----------
        biased_code : float
            Capacity of the biased_code edges to be added.
        """
        if biased_code is None:
            biased_code = self.config.biased_code
        if biased_code == "SRC":
            self._add_edge_with_capacity(
                self.source, node_id, capacity=2.0 * self.base_unary_capacity, edge_type="bias"
            )
            self._add_edge_with_capacity(self.sink, node_id, capacity=self.base_unary_capacity, edge_type="bias")
        elif biased_code == "SNK":
            self._add_edge_with_capacity(self.source, node_id, capacity=self.base_unary_capacity, edge_type="bias")
            self._add_edge_with_capacity(self.sink, node_id, capacity=2.0 * self.base_unary_capacity, edge_type="bias")

    def compute_idle_bonus(self, previous_depth: int, current_depth: int, total_edges: int) -> float:
        """Compute a normalized bonus (capacity reduction) for idling qubits.

        Formula: idle_time / (total_edges * (idle_time + 1))

        This creates a 'micro-bias'. It ensures that the bonus is always
        significantly smaller than the base capacity, acting only as a
        tie-breaker for the min-cut algorithm rather than forcing a cut.

        Parameters
        ----------
        previous_depth : int
            The depth index of the previous active gate.
        current_depth : int
            The depth index of the current active gate.
        total_edges : int
            The total count of temporal edges (qubit-operations) in the circuit.

        Returns:
        -------
        float
            The capacity reduction to apply.
        """
        idle_length = max(0, current_depth - previous_depth - 1)

        if idle_length <= self.config.switching_time:
            return 0.0

        if total_edges == 0:
            return 0.0

        return idle_length / (total_edges * (idle_length + 1))

    def _edge_capacity_with_idle_bonus(
        self, depths: list[int], total_edges: int, base_capacity: float | None = None
    ) -> float:
        """Compute the effective temporal edge capacity.

        Optionally reduced by an idle bonus if the qubit has been inactive for several layers.

        Parameters
        ----------
        depths : list[int]
            The ordered list of depth indices for a given qubit's gates.
        total_edges : int
            The total count of temporal edges in the circuit, used for normalization.
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
        bonus = self.compute_idle_bonus(prev_depth, curr_depth, total_edges)
        return max(0.0, base_capacity - bonus)

    def build_from_qiskit(
        self,
        circuit: QuantumCircuit,
        *,
        one_way_gates: dict[str, tuple[str, str]] | None = None,
        code_bias: bool = False,
        idle_bonus: bool = False,
    ) -> None:
        """Construct the code-switch graph generically.

        Parameters
        ----------
        circuit : QuantumCircuit
            The input quantum circuit.
        code_bias : bool, optional
            If True, add bias edges for nodes. Default is False.
        idle_bonus : bool, optional
            If True, reduce temporal edge capacities based on idle durations via
            `_edge_capacity_with_idle_bonus`. Default is False.
        one_way_gates : dict[str, tuple[str, str]], optional
            A dict mapping multi-qubit gate names (e.g., "CX") to direction tuples
            (e.g., ("SNK", "SRC")) specifying which code the control and target
            qubits should be in for one-way transversal execution.
            If a common multi-qubit gate is NOT in this dict, it is assumed
            both qubits must be in the same code (bidirectional infinite edges).
        """
        if one_way_gates is None:
            one_way_gates = {}

        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())

        total_temporal_edges = 0
        for node in dag.op_nodes():
            total_temporal_edges += len(node.qargs)

        qubit_activity: dict[int, list[int]] = {q: [] for q in range(circuit.num_qubits)}
        qubit_last_node: list[str | None] = [None] * circuit.num_qubits

        for depth, layer in enumerate(layers):
            for node in layer["graph"].op_nodes():
                self._process_gate_operation(
                    node,
                    depth,
                    circuit,
                    qubit_activity,
                    qubit_last_node,
                    one_way_gates,
                    code_bias,
                    idle_bonus,
                    total_temporal_edges,
                )

    def _process_gate_operation(
        self,
        node: DAGOpNode,
        depth: int,
        circuit: QuantumCircuit,
        qubit_activity: dict[int, list[int]],
        qubit_last_node: list[str | None],
        one_way_gates: dict[str, tuple[str, str]],
        code_bias: bool,
        idle_bonus: bool,
        total_temporal_edges: int,
    ) -> None:
        """Handle node creation, temporal edges, and code constraints for a single gate."""
        qubits_indices = [circuit.find_bit(q).index for q in node.qargs]
        gate_type = node.name.upper()

        current_step_nodes = []

        for q_idx in qubits_indices:
            qubit_activity[q_idx].append(depth)

            node_id = self._add_gate_node(gate_type, q_idx, depth)
            current_step_nodes.append((q_idx, node_id))

            prev_node = qubit_last_node[q_idx]
            if prev_node:
                capacity = self.config.default_temporal_edge_capacity
                if idle_bonus:
                    capacity = self._edge_capacity_with_idle_bonus(qubit_activity[q_idx], total_temporal_edges)
                self._add_regular_edge(prev_node, node_id, capacity=capacity)

            qubit_last_node[q_idx] = node_id

        self._apply_code_constraints(gate_type, current_step_nodes, one_way_gates, code_bias)

    def _apply_code_constraints(
        self,
        gate_type: str,
        current_step_nodes: list[tuple[int, str]],
        one_way_gates: dict[str, tuple[str, str]],
        code_bias: bool,
    ) -> None:
        """Apply infinite edges or bias edges based on gate sets."""
        is_source_unique = gate_type in self.gate_set_source and gate_type not in self.gate_set_sink
        is_sink_unique = gate_type in self.gate_set_sink and gate_type not in self.gate_set_source
        is_common = gate_type in self.common_gates

        if is_source_unique:
            for _, node_id in current_step_nodes:
                self._add_infinite_edge(self.source, node_id)

        elif is_sink_unique:
            for _, node_id in current_step_nodes:
                self._add_infinite_edge(self.sink, node_id)

        elif is_common:
            is_multi_qubit = len(current_step_nodes) > 1

            # Single qubit common gates are ignored (they float freely)
            if is_multi_qubit:
                is_one_way = gate_type in one_way_gates
                direction_is_snk2src = None
                if is_one_way:
                    direction = one_way_gates[gate_type]
                    direction_is_snk2src = self._parse_one_way_direction(direction)

                _, target_node = current_step_nodes[-1]
                controls = current_step_nodes[:-1]

                for _, ctrl_node in controls:
                    self._add_infinite_edge(
                        ctrl_node, target_node, bidirectional=not is_one_way, one_way_is_snk2src=direction_is_snk2src
                    )

                if code_bias:
                    for _, node_id in current_step_nodes:
                        self._add_bias_edges(node_id)

    @staticmethod
    def _parse_one_way_direction(direction: tuple[str, str]) -> bool:
        """Parse the one-way direction tuple into a boolean for control."""
        if direction == ("SRC", "SNK"):
            return False
        if direction == ("SNK", "SRC"):
            return True
        msg = f"Invalid one-way direction: {direction}. Must be ('SRC', 'SNK') or ('SNK', 'SRC')."
        raise ValueError(msg)

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
                    qubit, depth = self.parse_node_id(u)
                    switch_positions.append((qubit, depth))
        return len(switch_positions), switch_positions

    @staticmethod
    def parse_node_id(node_id: str) -> tuple[int, int]:
        """Extract (qubit, depth) from a node_id like 'H_q0_d3'."""
        match = pattern.match(node_id)
        if not match:
            msg = f"Invalid node_id format: {node_id}"
            raise ValueError(msg)
        qubit, depth = map(int, match.groups())
        return qubit, depth
