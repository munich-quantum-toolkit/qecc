"""Construct Example Layouts for Numerical Evaluation."""

from __future__ import annotations

import random

import networkx as nx
import itertools

random.seed(45)


def translate_layout_circuit(
    pairs: list[tuple[int, int] | int],
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]],
) -> list[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]]:
    """Translates a `pairs` circuit (with int labels) into the lattice's labels for a given layout.

    However, pairs does not only include tuple[int,int] but can include int as well for T gates. Then, layout will also include
    a list of factory positions in the key="factory_positions". but this will be ignored for this
    """
    # return [(layout[pair[0]], layout[pair[1]]) for pair in pairs]
    # terminal_pairs = [(layout[pair[0]], layout[pair[1]]) if isinstance(pair, tuple) else layout[pair] for pair in pairs]
    terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]] = []
    for pair in pairs:
        if isinstance(pair, tuple):
            pos1_raw = layout[pair[0]]
            pos2_raw = layout[pair[1]]
            if not isinstance(pos1_raw, tuple) or not isinstance(pos2_raw, tuple):
                msg = "Expected tuple[int, int] in layout mapping."
                raise TypeError(msg)
            pos1 = (int(pos1_raw[0]), int(pos1_raw[1]))
            pos2 = (int(pos2_raw[0]), int(pos2_raw[1]))
            terminal_pairs.append((pos1, pos2))
        else:
            pos_raw = layout[pair]
            if not isinstance(pos_raw, tuple):
                msg = "Expected tuple[int, int] in layout mapping."
                raise TypeError(msg)
            pos = (int(pos_raw[0]), int(pos_raw[1]))
            terminal_pairs.append(pos)

    return terminal_pairs


def gen_layout_scalable(
    layout_type: str,
    m: int,
    n: int,
    factories: list[tuple[int, int]],
    remove_edges: bool = True,
) -> tuple[nx.Graph, list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Generates layouts automatically, similar to those manual layouts in  `gen_layouts`.

    remove_edges: bool = whether or not to remove edges of adjacent data qubit locs

    the method builds up the lattice from small units defining the layout type.
    Then, it connects the units with edges such that a hexagonal lattice emerges.
    afterwards, vertices and edges are added such that we have one corridor around the whole thing

    we consider the hex layout as a brickwall lattice on cartesian coordinates

    m = number of row units
    n = number of col units

    Be aware that the number of data qubits depens on m,n and the unit of the lattice type!

    smallest unit for pair layout:
    x--x--x
    |  |  |
    o--o--x--x
             |
             x
    x = ancilla, o = data

    smallest unit for row layout:
    x
    |
    o
    but the edge is not necessary here

    smallest unit for hex layout:
       o--o--o
       |     |
    x--o--o--o--x
    |     |     |
    x--x--x--x--x
       |
       x

    smallest unit for triple layout:
    o--o--o--x--x--o--o--o--x--x
    |     |     |     |     |
    x--x--x--x--x--x--x--x--x--x

    smallest unit for single layout:
    x--x--o--x
    |     |
    x--x--x--x

    ATTENTION: It's the user's responsibility to place factories in a reasonable way on the factory_ring!
    """

    def add_hex_edges(g: nx.Graph):
        """
        private fct for adding edges at right spot
        add all possible horizontal edges, i.e. between (x,y) and (x+1,y)
        add vertical edges s.t. brickwall layout, i.e. for two nodes (x1,y1) and (x2, y2):
        A) x1 = x2 = even, y1 = even, y2 = y1 + 1= odd
        B) x1 = x2 = odd, y1 = odd, y2 = y1 + 1 = even
        """
        for x1, y1 in g.nodes():
            if (x1 + 1, y1) in g.nodes():
                g.add_edge((x1, y1), (x1 + 1, y1))
            if y1 % 2 == 0 and x1 % 2 == 0 and (x1, y1 + 1) in g.nodes():  # A
                g.add_edge((x1, y1), (x1, y1 + 1))
            if y1 % 2 != 0 and x1 % 2 != 0 and (x1, y1 + 1) in g.nodes():  # B
                g.add_edge((x1, y1), (x1, y1 + 1))

    def add_outer_corridor(g: nx.Graph, data_qubit_locs):
        """
        this should add vertices and correct edges, such that there are no elements of data_qubit_locs at the boundary
        """
        xs = [x for x, _ in data_qubit_locs]
        ys = [y for _, y in data_qubit_locs]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # expand bounding box
        xmin -= 1
        xmax += 1
        ymin -= 1
        ymax += 1

        # collect outer ring nodes
        corridor_nodes = []
        for x in range(xmin, xmax + 1):
            corridor_nodes.append((x, ymin))
            corridor_nodes.append((x, ymax))
        for y in range(ymin, ymax + 1):
            corridor_nodes.append((xmin, y))
            corridor_nodes.append((xmax, y))
        for y in range(ymin, ymax + 1):
            corridor_nodes.append(
                (xmin - 1, y)
            )  # one further col in order to be sure that corridor
            corridor_nodes.append((xmax + 1, y))

        corridor_nodes = list(set(corridor_nodes))

        g.add_nodes_from(corridor_nodes)
        return corridor_nodes

    def hex_cycles_from_position(node: tuple[int, int]) -> list[list[tuple[int, int]]]:
        x, y = node
        gtemp = nx.Graph()
        for dx, dy in itertools.product(range(-2, 3), range(-1, 2)):
            gtemp.add_node((x + dx, y + dy))
        gtemp.add_node(node)
        add_hex_edges(gtemp)
        cycles = nx.minimum_cycle_basis(gtemp, 6)

        return cycles

    def add_outer_corridor_diag(g: nx.Graph, data_qubit_locs):
        """for hex layout there is not standard rectangular layout. instead you need diagonal boundaries of the layout"""
        # find min and max per row and column

        corridor_nodes = set()
        xs = set([x for x, _ in data_qubit_locs])
        for xlabel in xs:
            ys = [y for x, y in data_qubit_locs if x == xlabel]
            ymin, ymax = min(ys), max(ys)
            cyc = hex_cycles_from_position((xlabel, ymin)) + hex_cycles_from_position(
                (xlabel, ymax)
            )
            for b in cyc:
                corridor_nodes.update(b)

        g.add_nodes_from(list(corridor_nodes))
        return corridor_nodes

    def add_factory_ring(g: nx.Graph, corridor_nodes):
        """add the factory ring on which factories are placed"""
        factory_ring = set()
        for node in corridor_nodes:
            cycles = hex_cycles_from_position(node)
            for cycle in cycles:
                for el in cycle:
                    if el not in g.nodes():
                        factory_ring.add(el)

        g.add_nodes_from(list(factory_ring))

        return list(factory_ring)

    g = nx.Graph()
    if layout_type not in {"row", "pair", "hex", "triple", "single"}:
        raise ValueError("Wrong Layout type!")

    data_qubit_locs = []  # fill this up

    if layout_type == "pair":
        unit_data = [(0, 1), (1, 1)]
        unit_ancilla = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2)]
        x_offset = 4
        y_offset = 2

    elif layout_type == "row":
        unit_data = [(0, 0)]
        unit_ancilla = [(0, 1)]
        x_offset = 1
        y_offset = 2

    elif layout_type == "hex":
        unit_data = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        unit_ancilla = [
            (-1, 1),
            (3, 1),
            (-1, 2),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (0, 3),
        ]
        x_offset = 5
        y_offset = 3

    elif layout_type == "triple":
        unit_data = [(0, 0), (1, 0), (2, 0), (5, 0), (6, 0), (7, 0)]
        unit_ancilla = [
            (3, 0),
            (4, 0),
            (8, 0),
            (9, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
        ]
        x_offset = 10
        y_offset = 2

    elif layout_type == "single":
        unit_data = [(2, 0)]
        unit_ancilla = [(0, 0), (1, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
        x_offset = 4
        y_offset = 2

    for row in range(m):
        for col in range(n):
            if layout_type in {"row", "pair"}:
                temp_data_qubit_locs = [
                    (x + x_offset * col, y + y_offset * row) for (x, y) in unit_data
                ]
                temp_vertices = [
                    (x + x_offset * col, y + y_offset * row)
                    for (x, y) in unit_data + unit_ancilla
                ]
            elif layout_type in {"hex"}:
                temp_data_qubit_locs = [
                    (x + x_offset * col + row, y + y_offset * row + col)
                    for (x, y) in unit_data
                ]
                temp_vertices = [
                    (x + x_offset * col + row, y + y_offset * row + col)
                    for (x, y) in unit_data + unit_ancilla
                ]
            elif layout_type in {"triple", "single"}:
                temp_data_qubit_locs = [
                    (x + col * x_offset + row, y + y_offset * row)
                    for (x, y) in unit_data
                ]
                temp_vertices = [
                    (x + col * x_offset + row, y + y_offset * row)
                    for (x, y) in unit_data + unit_ancilla
                ]
            g.add_nodes_from(temp_vertices)
            data_qubit_locs += temp_data_qubit_locs

    if layout_type in {"row", "pair"}:
        corridor_nodes = add_outer_corridor(
            g, data_qubit_locs
        )  # inplace addition of corridor
    elif layout_type in {"hex", "triple", "single"}:
        corridor_nodes = add_outer_corridor_diag(g, data_qubit_locs)
    add_hex_edges(g)  # inplace addition edges

    # remove leave ancilla nodes because they are not really useful for routing and only increase overhead
    leaves = [n for n, d in g.degree() if d == 1 or d == 0]
    g.remove_nodes_from(leaves)

    for leaf in leaves:
        if leaf in corridor_nodes:
            corridor_nodes.remove(leaf)

    # add factory ring
    factory_ring = add_factory_ring(g, corridor_nodes)
    add_hex_edges(g)  # inplace addition edges
    # filter factory nodes
    g, _, _ = filter_factory_nodes(g, factory_ring, factories)

    if remove_edges:
        for data_qubit in data_qubit_locs:
            neighbours: list[tuple[int, int]] = list(g.neighbors(data_qubit))
            for neigh in neighbours:
                if neigh in data_qubit_locs:  # if a neighbor is also a logical qubit
                    g.remove_edges_from([(data_qubit, neigh), (neigh, data_qubit)])

    nx.set_node_attributes(g, {n: n for n in g.nodes()}, "pos")
    return g, data_qubit_locs, factory_ring


def filter_factory_nodes(
    g: nx.Graph, factory_ring: list[tuple[int, int]], factories: list[tuple[int, int]]
) -> tuple[nx.Graph, set[tuple[int, int]], set[tuple[int, int]]]:
    """Filter ancillas from factory_ring which should be kept in the graph considering the included factories.

    Note that one should place the factories in a suitable way such that they are not weirdly connected.

    Args:
        g (nx.Graph): Graph of the Layout
        factory_ring (list[tuple[int,int]]): Locations at the boundary where factories are allowed in principle.
        factories (list[tuple[int,int]]): factory positions.

    Returns:
        tuple[nx.Graph, set, set]: graph with reduced ancillas, kept nodes, nodes_to_dsicard
    """
    kept_nodes = set()

    for el in factories:
        assert (
            el in factory_ring
        ), "A factory is placed outside the factory ring which is not allowed"

    cycles = list(nx.simple_cycles(g, length_bound=6))  # only hexagons
    for node in factories:
        # add the least amount of qubits to get a full cycle
        possible_cycles = [
            set(cycle) & set(factory_ring)
            for cycle in cycles
            if node in list(cycle) and len(list(cycle)) == 6
        ]
        if possible_cycles:
            best_cycle_part = min(possible_cycles, key=len)
            kept_nodes.update(best_cycle_part)

    nodes_to_discard = set(factory_ring) - kept_nodes - set(factories)
    g.remove_nodes_from(nodes_to_discard)

    return g, kept_nodes, nodes_to_discard
