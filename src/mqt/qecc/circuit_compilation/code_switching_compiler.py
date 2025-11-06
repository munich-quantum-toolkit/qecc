# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Code Switching Compiler to find the minimum number of switches."""

from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

DEFAULT_TEMPORAL_EDGE_CAPACITY = 100.0


class CodeSwitchGraph:
    """A directed graph representation of a quantum circuit for code-switching analysis using min-cut / max-flow optimization.

    The graph is constructed such that:
      - Each quantum operation (T, H, CNOT) corresponds to one or more nodes.
      - Source (SRC) and sink (SNK) nodes represent two different codes:
          * Source-connected nodes (H, CNOT) → operations that can be done transversally in code a 2D Color Code.
          * Sink-connected nodes (T, CNOT) → operations that can be done transversally in code a 3D Color Code.
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
        edge_type : str, optional
            Type of the edge (default is "temporal").
        bidirectional : bool, optional
            If True, add the reverse edge as well (default is True).
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
        bidirectional : bool, optional
            If True, add the reverse edge as well (default is True).
        """
        self.add_edge_with_capacity(u, v, capacity=float("inf"), edge_type="fixed", bidirectional=bidirectional)

    def add_regular_edge(
        self, u: str, v: str, capacity: float = DEFAULT_TEMPORAL_EDGE_CAPACITY, bidirectional: bool = True
    ) -> None:
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
        # Sink code can perform T and CNOT gates
        if gate_type == "T":
            self.add_infinite_edge(self.sink, node_id)
        # Source code can perform H and CNOT gates
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
        one_way_transversal_cnot : bool, optional
            If True, allow transversal CNOTs from 3D (control) to 2D (target) besides the usual requirement that both qubits remain in the same code.
        """
        self.add_infinite_edge(control_node, target_node, bidirectional=not (one_way_transversal_cnot))

    @staticmethod
    def compute_idle_bonus(previous_depth: int, current_depth: int) -> float:
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

        max_bonus = 5
        proportional_factor = 1
        return min(max_bonus, proportional_factor * idle_length)

    def _edge_capacity_with_idle_bonus(
        self, depths: list[int], base_capacity: float = DEFAULT_TEMPORAL_EDGE_CAPACITY
    ) -> float:
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
            The adjusted edge capacity, ensuring a lower bound of 1.0.
        """
        if len(depths) < 2:
            return base_capacity

        prev_depth, curr_depth = depths[-2], depths[-1]
        bonus = self.compute_idle_bonus(prev_depth, curr_depth)
        return max(1.0, base_capacity - bonus)

    def build_from_qiskit(
        self,
        circuit: QuantumCircuit,
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
            If True, apply idle bonuses to temporal edges.

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
                            capacity = (
                                self._edge_capacity_with_idle_bonus(qubit_activity[q])
                                if idle_bonus
                                else DEFAULT_TEMPORAL_EDGE_CAPACITY
                            )
                            self.add_regular_edge(qubit_last_node[q], gate_node, capacity=capacity)
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
            if data["edge_type"] == "temporal":
                # needed to avoid double counting edges in undirected sense
                key = tuple(sorted((u, v)))
                if key not in seen:
                    if (u in S and v in T) or (v in S and u in T):
                        num_switches += 1
                        switch_cost += data["capacity"]
                    seen.add(key)
        if return_raw_data:
            return cut_value, S, T
        return num_switches, S, T


def inspect_serial_layers(circuit: QuantumCircuit) -> list[dict[str, object]]:
    """Return a lightweight description of `dag.layers()` for debugging.

    Each list element corresponds to one serial layer and contains:
      - 'ops': list of (op.name, [qubit indices touched]) tuples for the layer.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to inspect.

    Returns:
    -------
    List[Dict]
        A list describing each layer. Use this to confirm layer indices and which
        qubits are active in each layer.
    """
    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())
    layers_descr = []

    for layer_idx, layer in enumerate(layers):
        layer_graph = layer["graph"]
        ops_in_layer = []
        for node in layer_graph.op_nodes():
            assert isinstance(node, DAGOpNode)
            # map Qubit objects to their indices in the original circuit
            q_indices = [circuit.find_bit(q).index for q in node.qargs] if node.qargs else []
            ops_in_layer.append((getattr(node.op, "name", repr(node.op)), q_indices))
        layers_descr.append({"layer_index": layer_idx, "ops": ops_in_layer})

    return layers_descr


def insert_switch_placeholders(
    circuit: QuantumCircuit,
    switch_positions: list[tuple[int, int]],
    placeholder_depth: int = 1,
) -> QuantumCircuit:
    """Return a new circuit with 'switch' placeholders inserted between global DAG layers.

    This function inserts placeholders *after* the entire global layer with index
    `layer_index` (i.e., between layer `layer_index` and the next). Placeholders are
    placed on the requested qubit regardless of whether that qubit was active in that layer.

    Parameters
    ----------
    circuit : QuantumCircuit
        The original circuit to augment.
    switch_positions : List[Tuple[int, int]]
        List of (qubit_index, layer_index). `layer_index` refers to the index
        from `list(circuit_to_dag(circuit).layers())`. A placeholder for
        `(q, k)` will be inserted after global layer `k`.
    placeholder_depth : int, optional
        Virtual depth (single-qubit layers) the placeholder should represent.
    expand_placeholder : bool, optional
        If True, expand each placeholder into `placeholder_depth` calls to
        `QuantumCircuit.id(qubit)` so that `QuantumCircuit.depth()` increases.
        If False, append a single `SwitchGate` marker (informational only).

    Returns:
    -------
    QuantumCircuit
        New circuit with placeholders inserted.
    """
    # Build DAG and layers
    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())

    # Normalize and group requested placeholders by qubit
    placeholders_by_qubit: dict[int, list[int]] = {}
    for qidx, layer_idx in switch_positions:
        if layer_idx < 0:
            # ignore negative indices (could alternatively raise)
            continue
        placeholders_by_qubit.setdefault(qidx, []).append(layer_idx)

    # Sort layer indices per qubit for deterministic behavior
    for depths in placeholders_by_qubit.values():
        depths.sort()

    # Prepare output circuit with same registers
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name + "_with_switches")

    # Track which placeholders we already inserted
    inserted_placeholders: dict[int, set[int]] = {q: set() for q in placeholders_by_qubit}

    def _append_placeholder_on_qubit(q_index: int, depth_equiv: int) -> None:
        """Append a placeholder (either expanded ids or a SwitchGate) on the qubit."""
        qubit = circuit.qubits[q_index]
        # append id several times so `.depth()` counts them
        for _ in range(max(1, int(depth_equiv))):
            new_qc.id(qubit)

    # Iterate layers in order; copy each op, then after the whole layer insert placeholders for that layer
    for depth_idx, layer in enumerate(layers):
        layer_graph = layer["graph"]
        # append ops of the layer to new_qc (deterministic order: iterate nodes)
        for node in layer_graph.op_nodes():
            assert isinstance(node, DAGOpNode)
            # Map node.qargs (Qubit objects) to the corresponding qubit objects of the original circuit
            q_indices = [circuit.find_bit(q).index for q in node.qargs] if node.qargs else []
            qbit_objs = [circuit.qubits[i] for i in q_indices]
            c_indices = [circuit.find_bit(c).index for c in node.cargs] if getattr(node, "cargs", None) else []
            cbit_objs = [circuit.clbits[i] for i in c_indices] if c_indices else []

            # Append the operation to the new circuit on the same physical qubits/bits
            new_qc.append(node.op, qbit_objs, cbit_objs)

        # --- AFTER FINISHING THIS GLOBAL LAYER: insert placeholders targeted at this layer ---
        for q_index, depths in placeholders_by_qubit.items():
            for target_depth in depths:
                if target_depth == depth_idx and target_depth not in inserted_placeholders[q_index]:
                    _append_placeholder_on_qubit(q_index, placeholder_depth)
                    inserted_placeholders[q_index].add(target_depth)

    # Append any placeholders whose requested layer index was beyond the number of layers
    for q_index, depths in placeholders_by_qubit.items():
        for target_depth in depths:
            if target_depth not in inserted_placeholders[q_index]:
                # target was never inserted (layer out of range) -> append at end
                _append_placeholder_on_qubit(q_index, placeholder_depth)
                inserted_placeholders[q_index].add(target_depth)

    return new_qc
