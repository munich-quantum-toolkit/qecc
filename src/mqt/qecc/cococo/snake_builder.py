"""Generates Stabilizers for n-snakes."""

from __future__ import annotations

import itertools
import warnings
from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

if TYPE_CHECKING:
    import numpy.typing as npt

from mqt.qecc import CSSCode

NodePos = tuple[
    int, int
]  # Position of the graphs is determined by 2 integer coordinates. Depending of the snake type, qubits are on the nodes, or on the edge between two nodes.


class SnakeBuilderSC:
    """Constructs a n-snake of distance d with a surface code in snake shape on the square lattice substate (leading to brickwall routing graph aka hex graph)."""

    def __init__(
        self,
        g: nx.Graph,
        positions_rough: list[list[NodePos]],
        positions_smooth: list[list[NodePos]],
        d: int,
    ) -> None:
        """Initializes a SC n-snake.

        Args:
            g (nx.Graph): quadratic graph
            positions_rough (list[list[NodePos]]): Node positions constituting the rough boundaries (a list per boundary)
            positions_smooth (list[list[NodePos]]): Node positions constituting the smooth boundaries (a list per boundary)
            d (int): distance
        """
        self.g = g
        self.positions_rough = positions_rough
        self.positions_smooth = positions_smooth
        self.d = d
        if len(positions_smooth) != 2:
            msg = "To encode 1 logical qubit, there must be 2 smooth boundaries."
            raise ValueError(msg)
        if len(positions_rough) != 2:
            msg = "To encode 1 logical qubit, there must be 2 rough boundaries."
            raise ValueError(msg)
        dist = (
            min(
                [len(el) for el in positions_rough]
                + [len(el) for el in positions_smooth]
            )
            - 1
        )  # -1 because we count edges not nodes
        if dist != d:
            msg = f"Distance d={dist} does not coincide with the geometry of the rough and smooth positions."
            raise ValueError(msg)

        # must ensure that given G is big enough to place the stabilizers on it (maybe remove G from input and generate it within the class)
        flat_smooth = {pos for group in positions_smooth for pos in group}
        flat_rough = {pos for group in positions_rough for pos in group}
        joint_positions = list(flat_smooth | flat_rough)
        for pos in joint_positions:
            if pos not in self.g.nodes():
                msg = "The input graph is chosen too small and cannot accommodate all qubits for the stabs."
                raise ValueError(msg)

    def fill_snake(self) -> list[list[NodePos]]:
        """Finds the true interior nodes by marking exterior nodes from all four edges.

        Returns:
            list[list[NodePos]]: Positions of ALL nodes in the snakes.
        """
        positions_smooth_flattened = [
            pos for sublist in self.positions_smooth for pos in sublist
        ]
        positions_rough_flattened = [
            pos for sublist in self.positions_rough for pos in sublist
        ]
        boundary_nodes = set(
            positions_smooth_flattened + positions_rough_flattened
        )  # Boundary nodes

        all_nodes = set(self.g.nodes())  # All nodes in the lattice
        exterior_nodes = set()  # Nodes confirmed to be exterior

        x_values = sorted({x for x, _ in all_nodes})
        y_values = sorted({y for _, y in all_nodes})

        def mark_exterior(start_x: int, start_y: int, dx: int, dy: int) -> None:
            """Mark exterior nodes by moving from a starting edge."""
            x, y = start_x, start_y
            while (x, y) in all_nodes:
                if (x, y) in boundary_nodes:
                    return  # Stop when hitting a boundary
                exterior_nodes.add((x, y))
                x += dx
                y += dy

        # Sweep from left
        for y in y_values:
            mark_exterior(x_values[0], y, 1, 0)

        # Sweep from right
        for y in y_values:
            mark_exterior(x_values[-1], y, -1, 0)

        # Sweep from bottom
        for x in x_values:
            mark_exterior(x, y_values[-1], 0, -1)

        # Sweep from right
        for x in x_values:
            mark_exterior(x, y_values[0], 0, 1)

        # Interior nodes = all nodes minus exterior and boundary
        interior_nodes = all_nodes - exterior_nodes - boundary_nodes

        self.inner_nodes = list(interior_nodes)
        self.boundary_nodes = list(boundary_nodes)

        return list(boundary_nodes) + list(interior_nodes)

    @staticmethod
    def neighbors_ver_hor(node1: NodePos, node2: NodePos) -> bool:
        """Checks whether two nodes are neighbors.

        Vertical and horizontal neighbors. Not based on initial graph structure because no diagonal edges present.

        Args:
            node1 (NodePos): node1
            node2 (NodePos): node2

        Returns:
            bool: whether both are neighbors.
        """
        return (abs(node1[0] - node2[0]) == 1 and node1[1] == node2[1]) or (
            abs(node1[1] - node2[1]) == 1 and node1[0] == node2[0]
        )

    @staticmethod
    def neighbors_diag(node1: NodePos, node2: NodePos) -> bool:
        """Checks whether two nodes are neighbors.

        Diagonal. Not based on initial graph structure because no diagonal edges present.

        Args:
            node1 (NodePos): node1
            node2 (NodePos): node2

        Returns:
            bool: whether both are neighbors.
        """
        return bool(
            abs(node1[1] - node2[1]) == 1 and abs(node1[0] - node2[0]) == 1
        )  # diagonal neighbors or no neighbors at all

    def collect_qubit_positions(self) -> list[tuple[NodePos, NodePos]]:
        """Collects the edges on which qubit are placed defining the merged object (snake+logical patches), depending on the given rough/smooth edges/boundaries.

        These positions can be used to construct the stabilizers (`gen_stars`, `gen_plaquettes`) given the rough/smooth boundaries.
        Only qubits on horizontal/vertical edges, no new diagonal edges added.

        Returns:
            list[tuple[NodePos, NodePos]]: List of edges where qubits are placed.
        """
        qubit_edges = []

        nodes = self.boundary_nodes + self.inner_nodes  # all nodes
        # add qubits on each horizontal/vertical edge which is contained in the snake
        for edge in itertools.combinations(nodes, 2):
            neighborhood_bool = self.neighbors_ver_hor(edge[0], edge[1])
            if neighborhood_bool:
                qubit_edges.append(edge)

        # remove the qubits on rough boundary (only remove if horizontal or vertical, because diagonal rough boundaries has no additional qubits)
        for rough_b in self.positions_rough:
            for edge in itertools.combinations(rough_b, 2):
                if edge in qubit_edges:
                    qubit_edges.remove(edge)
                elif (edge[1], edge[0]) in qubit_edges:
                    qubit_edges.remove((edge[1], edge[0]))

        self.qubit_edges = qubit_edges
        return qubit_edges

    def gen_stars(self) -> list[list[tuple[NodePos, NodePos]]]:
        """Generates star operators.

        Returns:
            list[list[tuple[NodePos, NodePos]]]: List of lists of edges, where each list determines a star operator.
        """
        stars = []
        nodes = self.boundary_nodes + self.inner_nodes  # all nodes
        for node in nodes:
            # collect all qubits which are connected to this node
            temp_qubits = [
                edge for edge in self.qubit_edges if edge[0] == node or edge[1] == node
            ]
            if len(temp_qubits) > 1:
                stars.append(temp_qubits)
        # if star is placed on rough boundary, it must be removed
        stars_new = []
        for star in stars:
            {
                tup for pair in star for tup in pair
            }  # flatten edges such that nodes can be compared
            lst_on_rough = []
            for rough_b in self.positions_rough:  # through both rough boundaries
                # check whether the central node of the star is on the rough boundary
                all_nodes = [node for edge in star for node in edge]
                node_counts = Counter(all_nodes)
                central_node = max(node_counts, key=lambda x: node_counts.get(x, 0))
                common_el = central_node in rough_b
                lst_on_rough.append(common_el)
            assert (
                sum(1 for x in lst_on_rough if x != 0) <= 1
            ), "The star has overlaps with both rough boundaries, this cannot be."
            # horizontal rough remove: weigh3 star with 2 overlapping nodes on rough b
            if sum(lst_on_rough) != 1 or len(star) != 2:
                stars_new.append(star)

        self.stars = stars_new
        return stars_new

    def gen_plaquettes(self) -> list[list[tuple[NodePos, NodePos]]]:
        """Generates Plaquette Operators.

        Returns:
            list[list[tuple[NodePos, NodePos]]]: A List of Plaquette Operators. Each Plaquette Operator is a list of tuple[NodePos, NodePos] since qubits are placed on edges.
        """
        plaquettes = []
        nodes = self.boundary_nodes + self.inner_nodes  # all nodes
        # since we check the nodes for being in the upper left corner, you can loose some plaquettes. hence add more nodes to minimum x and y, even though there will be useless checks
        min_x = min(t[0] for t in nodes)
        min_y = min(t[1] for t in nodes)
        max_x = max(t[0] for t in nodes)
        max_y = max(t[1] for t in nodes)
        collected = []
        i, j = 0, 0

        while True:
            point = (min_x + i, min_y + j)
            if point in set(nodes):
                j = 0
                i += 1
                continue
            collected.append(point)

            j += 1
            if j > max_y:
                j = 0
                i += 1
            if i > max_x:
                break
        nodes += collected

        # find all squares in the snake
        # for each square, check where qubits are and add to plaquettes (number of qubits can be smaller than 4)
        for node in nodes:
            # create square of nodes with node in upper left corner.
            square = [
                node,
                (node[0], node[1] + 1),
                (node[0] + 1, node[1] + 1),
                (node[0] + 1, node[1]),
            ]  # cyclically aligned
            edges_square = list(itertools.combinations(square, 2))  # also diagonals
            plaquette = [
                edge
                for edge in edges_square
                if (edge[0], edge[1]) in self.qubit_edges
                or (edge[1], edge[0]) in self.qubit_edges
            ]
            # add to plaquettes if more than 1 qubit contained
            if len(plaquette) > 1:
                plaquettes.append(plaquette)

        plaquettes_new = []
        # remove weight-2 plaquettes on the smooth boundary if there are any (on diagonal)
        for plaquette in plaquettes:
            if len(plaquette) == 2:  # 2edges= 2 qubits
                plaquette_flat = [node for edge in plaquette for node in edge]
                # only check for those lements which are NOT the central qubit
                node_counts = Counter(plaquette_flat)
                central_node = max(node_counts, key=lambda x: node_counts.get(x, 0))
                plaquette_flat = [
                    node for node in plaquette_flat if node != central_node
                ]
                plaquette_set = set(plaquette_flat)
                lst_on_smooth = []
                for smooth_b in self.positions_smooth:
                    common_el = sum(1 for t in plaquette_set if t in smooth_b)
                    lst_on_smooth.append(common_el)
                assert (
                    sum(1 for x in lst_on_smooth if x != 0) <= 1
                ), "The plaquette has overlaps with both rough boundaries, this cannot be."
                if sum(lst_on_smooth) != 2:
                    plaquettes_new.append(plaquette)
            else:
                plaquettes_new.append(plaquette)

        self.plaquettes = plaquettes_new
        return plaquettes_new

    def create_stabs(
        self,
    ) -> tuple[
        list[list[tuple[NodePos, NodePos]]], list[list[tuple[NodePos, NodePos]]]
    ]:
        """Summarizes all previously defined methods and returns the plaquette stabilizers and star stabilizers of the merged snake object.

        Returns:
            tuple[ list[list[tuple[NodePos, NodePos]]], list[list[tuple[NodePos, NodePos]]] ]: Returns two lists of stabilizers. The first are the plaquettes, second are the stars.
            Each list contains the plaquette/star operators respectively.
            Each Plaquette Operator is a list of tuple[NodePos, NodePos] since qubits are placed on edges.
        """
        _ = self.fill_snake()
        _ = self.collect_qubit_positions()
        _ = self.gen_stars()
        _ = self.gen_plaquettes()

        q = len(self.qubit_edges)
        lenstars = len(self.stars)
        lenplaq = len(self.plaquettes)

        if q - lenstars - lenplaq != 1:
            warnings.warn(
                f"Your stabilizers are wrong. They should create 1 logical qubit but they yield {q - lenstars - lenplaq} instead.",
                UserWarning,
                stacklevel=2,
            )

        return self.plaquettes, self.stars

    def gen_checks(
        self,
    ) -> tuple[
        npt.NDArray[np.int_], npt.NDArray[np.int_], dict[tuple[NodePos, NodePos], int]
    ]:
        """Return checks and translation dict.

        Returns:
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], dict[tuple[NodePos, NodePos], int]]: X check matrix, Z check matrix and the translation dict which translates between posiitons on the graph and positions in the check matrix.
        """
        trans_dict = {
            edge: i for i, edge in enumerate(self.qubit_edges)
        }  # both edge orderings should be included to make sure that we get no key errors
        trans_dict2 = {(edge[1], edge[0]): i for i, edge in enumerate(self.qubit_edges)}
        trans_dict |= trans_dict2
        q = len(self.qubit_edges)
        hz = np.zeros((len(self.plaquettes), q), dtype=int)
        for row, plaquette in enumerate(self.plaquettes):
            translated_plaquette = [int(trans_dict[edge]) for edge in plaquette]
            for el in translated_plaquette:
                hz[row, el] = 1
        hx = np.zeros((len(self.stars), q), dtype=int)
        for row, star in enumerate(self.stars):
            translated_star = [int(trans_dict[edge]) for edge in star]
            for el in translated_star:
                hx[row, el] = 1
        self.trans_dict = trans_dict
        return hx, hz, trans_dict

    def plot_stabs(
        self,
        opz: list[tuple[NodePos, NodePos]] | None = None,
        opx: list[tuple[NodePos, NodePos]] | None = None,
        size: tuple[int, int] = (8, 8),
    ) -> None:
        """Plots plaquettes and star operators as well as the snake itself.

        opz and opx are the logical operators retrieved via mqt.qecc.CSSCode which are already translated as edges on the graph.

        Args:
            opz (list[tuple[NodePos, NodePos]] | None, optional): Logical Operator Z_L as list of qubit positions. Each qubit position is an edge on the graph. Defaults to None.
            opx (list[tuple[NodePos, NodePos]] | None, optional): Logical Operator Z_L as list of qubit positions. Each qubit position is an edge on the graph. Defaults to None.
            size (tuple[int, int], optional): Size of the plot. Defaults to (8, 8).
        """
        # pos = {node: (node[0], -node[1]) for node in self.g.nodes()}  # Adjust for proper display
        pos = {(x, y): (x, y) for x, y in self.g.nodes()}  # Keep y as positive

        midpoints = [
            ((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1), (x2, y2) in self.qubit_edges
        ]

        plt.figure(figsize=size)
        nx.draw(
            self.g,
            pos,
            with_labels=True,
            node_size=100,
            edge_color="lightgray",
            font_size=8,
        )

        for star in self.stars:
            all_nodes = [node for edge in star for node in edge]
            node_counts = Counter(all_nodes)
            # central_node = max(node_counts, key=node_counts.get)
            central_node = max(node_counts, key=lambda x: node_counts.get(x, 0))
            for u, v in star:
                if u == central_node:
                    start, end = u, v
                elif v == central_node:
                    start, end = v, u
                else:
                    continue

                midpoint = (
                    (pos[start][0] + pos[end][0]) / 2,
                    (pos[start][1] + pos[end][1]) / 2,
                )

                # Plot the half-edge from the central node to the midpoint
                plt.plot(
                    [pos[start][0], midpoint[0]],
                    [pos[start][1], midpoint[1]],
                    color="orange",
                    linewidth=3,
                    label="X Star",
                )

        ax = plt.gca()
        for plaquette in self.plaquettes:
            square = {node for edge in plaquette for node in edge}
            square_pos = [pos[node] for node in square]
            square_pos = convex_hull(square_pos)
            # shrink the polygon a little
            square_pos_arr = np.array(square_pos)
            centroid = square_pos_arr.mean(axis=0)
            factor = 0.6
            square_pos_arr = centroid + factor * (square_pos_arr - centroid)
            polygon = Polygon(
                square_pos_arr, closed=True, color="green", alpha=0.3, label="Z Face"
            )
            ax.add_patch(polygon)

        nodes = self.boundary_nodes + self.inner_nodes  # all nodes
        x_mid, y_mid = zip(*midpoints)  # Extract x and y coordinates
        plt.scatter(x_mid, y_mid, color="red", s=20, zorder=3)  # Small blue dots
        nx.draw_networkx_nodes(
            self.g, pos, nodelist=nodes, node_color="blue", node_size=100
        )

        # integer labels
        for original_label, new_label in self.trans_dict.items():
            if (
                original_label[0] in pos and original_label[1] in pos
            ):  # Ensure both nodes exist
                x1, y1 = pos[original_label[0]]
                x2, y2 = pos[original_label[1]]

                # Compute the midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # Plot the text at the midpoint
                plt.text(
                    mid_x,
                    mid_y,
                    str(new_label),
                    fontsize=8,
                    color="green",
                    ha="center",
                    va="center",
                )

        if opx is not None:
            nx.draw_networkx_edges(
                self.g,
                pos,
                edgelist=opx,
                edge_color="teal",
                width=5,
                alpha=0.7,
                label="$X_L$",
            )
        if opz is not None:
            nx.draw_networkx_edges(
                self.g,
                pos,
                edgelist=opz,
                edge_color="blueviolet",
                width=5,
                alpha=0.7,
                label="$Z_L$",
            )

        # no duplicates in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_legend = dict.fromkeys(
            labels, None
        )  # Removes duplicates while keeping order
        unique_handles = [handles[labels.index(label)] for label in unique_legend]
        plt.legend(unique_handles, unique_legend.keys())
        plt.show()

    def get_logical_operator_basis(self) -> npt.NDArray[np.int8]:
        """Generates Logical Operators of the snake.

        Returns:
            npt.NDArray[np.int8]: logical operators.
        """
        hx, hz, _ = self.gen_checks()
        return CSSCode._compute_logical(np.array(hx), np.array(hz))  # noqa: SLF001


class SnakeBuilderSTDW:
    """Constructs a n-snake of distance d color codes with semi transparent domain wall."""

    def __init__(self, g: nx.Graph, positions: list[list[NodePos]], d: int) -> None:
        """Initializes a n snake with ancillas on the interface and semi transparent domain wall.

        Args:
            g (nx.Graph): Hexagonal graph on which the data qubits are placed
            positions (list[list[nodePos]]): vertex positions on the nx graph (tuples) for each triangular color code patch. One list per triangle.
                The order of the triangles is important, as subsequent triangle lists should belong to neighboring triangles.
            d (int): distance of the triangular color code patches
        """
        self.g = g
        self.positions = positions
        self.d = d

        # determine global labeling for all vertices in the n-snake
        all_pos = []
        for tile in self.positions:
            all_pos += tile
        labels = {}
        for i, pos in enumerate(all_pos):
            labels.update({pos: i})
        self.labels = labels

        # assertions for q(d) per triangle and p(d)=(q(d)-1)/2 per triangle (number of patches)
        t = (d - 1) / 2
        q = int(3 * t**2 + 3 * t + 1)
        for i, triangle in enumerate(self.positions):
            assert (
                len(triangle) == q
            ), f"Your set of vertices for triangle {i} does not fit the expected number of qubits for distance d={d}."
        p = int((q - 1) / 2)
        self.q = q
        self.p = p
        self.n = len(positions)
        self.q_tilde = self.n * self.q + (self.n - 1) * (self.d - 1)
        self.trans_dict: dict[NodePos, int] | None = None

    def find_triangle_edges_corners(self, n_triangle: int) -> list[list[NodePos]]:
        """Searches for the graph labels of the vertices in the 3 edges of the given triangle.

        Args:
            n_triangle (int): label of triangle in self.positions

        Returns:
            list[list[NodePos]]: list of positions in the three boundaries/edges of the triangle.
        """
        triangle = self.positions[n_triangle]
        lst_boundary = (
            []
        )  # gather all vertices with a single neighbor outside the set (i.e. on edge)
        lst_corner = (
            []
        )  # gather all 3 vertices with 2 neighbors outside the set (i.e. corners of triangle)
        for vertex in triangle:
            neighbors_temp = list(self.g.neighbors(vertex))
            outside_neighbors = [v for v in neighbors_temp if v not in triangle]
            if len(outside_neighbors) == 2:
                lst_corner.append(vertex)
            elif len(outside_neighbors) == 1:
                lst_boundary.append(vertex)
            elif len(outside_neighbors) == 0:
                continue
            elif len(outside_neighbors) == 3:
                msg = f"There is an isolated qubit in your input triangle {n_triangle}."
                raise ValueError(msg)
        assert (
            len(lst_corner) == 3
        ), f"Something weird happened. lst_corner has {len(lst_corner)} elements instead of 3."
        assert (
            len(lst_boundary) == (self.d - 2) * 3
        ), f"Something weird happened. lst_boundary has {len(lst_boundary)} elements instead of {(self.d - 2) * 3}."

        return [lst_corner, lst_boundary]

    def find_interface_ancillas(
        self, triangle_0: int, triangle_1: int
    ) -> list[tuple[NodePos, NodePos]]:
        """Finds ancilla vertices on the interface between triangle_0 and triangle_1.

        Args:
            triangle_0 (int): label of first triangle in self.positions[triangle_0]
            triangle_1 (int): label of second triangle in self.positions[triangle_1] -> must be adjacent to the first!

        Returns:
            list[tuple[NodePos, NodePos]]: ancilla vertices on the interface between triangle_0 and triangle_1
        """
        [lst_corner_0, lst_boundary_0] = self.find_triangle_edges_corners(triangle_0)
        [lst_corner_1, lst_boundary_1] = self.find_triangle_edges_corners(triangle_1)

        # find pairs of elements which are next nearest neighbors.
        next_nearest_neighbors = []
        for el0 in lst_corner_0:
            for el1 in lst_corner_1:
                path = nx.dijkstra_path(self.g, el0, el1)
                if len(path) - 1 == 2:
                    next_nearest_neighbors.append(
                        {"el0": el0, "el1": el1, "path": path}
                    )
        for el0 in lst_boundary_0:
            for el1 in lst_boundary_1:
                path = nx.dijkstra_path(self.g, el0, el1)
                if len(path) - 1 == 2:
                    next_nearest_neighbors.append(
                        {"el0": el0, "el1": el1, "path": path}
                    )

        # filter ancillas
        ancillas = [el["path"][1] for el in next_nearest_neighbors]

        # only use those ancillas which do indeed have a nearest neighbor in the ancilla set (single ancillas not wanted)
        ancilla_pairs: list[tuple[NodePos, NodePos]] = []
        for node in ancillas:
            neighbors = set(self.g.neighbors(node))
            valid_neighbors = neighbors.intersection(ancillas)
            if valid_neighbors:
                # Ensure unique pairs, avoid (n1, n2) and (n2, n1)
                ancilla_pairs.extend(
                    (node, neighbor) for neighbor in valid_neighbors if node < neighbor
                )

        return ancilla_pairs

    def hex_plaquettes(self) -> list[list[NodePos]]:
        """Find all hexagonal plaquettes on original g.

        Returns:
            list[list[NodePos]]: all possible hexagonal plaquettes as vertices on g.
        """
        cycles = list(nx.simple_cycles(self.g, length_bound=6))
        return [list(set(cycle)) for cycle in cycles if len(cycle) == 6]

    def find_stabilizers(self) -> tuple[list[list[NodePos]], list[list[NodePos]]]:
        """Find stabilizers on self.positions.

        Returns:
            tuple[list[list[NodePos]], list[list[NodePos]]]: Two lists, first the Z stabilizrs, and second the X stabilizers. There are more Z than X stabilizers
            because we assume a Z merge by default. but can be interchanged of course for a X merge.
        """
        total_nodes = []  # find all relevant nodes first
        z_plaquettes: list[list[NodePos]] = []

        for triangle in self.positions:  # all nodes in the triangles
            total_nodes += triangle

        for i in range(len(self.positions) - 1):  # everything in the interface
            ancilla_pairs = self.find_interface_ancillas(i, i + 1)
            z_plaquettes += [list(el) for el in ancilla_pairs]  # pair stabs
            ancillas_flattened = [item for sublist in ancilla_pairs for item in sublist]
            total_nodes += ancillas_flattened

        # structure the nodes as intersection to the underlying hexagonal plaquettes (automatically other shape in interface)
        hexagonal_plaquettes = self.hex_plaquettes()
        for plaquette in hexagonal_plaquettes:
            overlap = set(plaquette) & set(total_nodes)
            if len(overlap) >= 3:  # Ensure a meaningful plaquette (full or partial)
                z_plaquettes.append(list(overlap))

        # filter out interface only plaquettes to distinguish x_plaquettes and z_plaquettes
        x_plaquettes: list[list[NodePos]] = []
        for plaquette in z_plaquettes:
            if (
                len(plaquette) == 6
            ):  # pairs, weight-3, weight-5 in the interface NOT wanted for X stabs
                # also remove the hex plaquettes within the interface (touching vertices of two triangles)
                is_x_plaquette = True
                for i in range(len(self.positions) - 1):
                    if set(plaquette) & set(self.positions[i]) and set(plaquette) & set(
                        self.positions[i + 1]
                    ):
                        is_x_plaquette = False
                        break
                if is_x_plaquette:
                    x_plaquettes.append(list(plaquette))  # only if above NOT fulfilled
            elif len(plaquette) == 4:
                x_plaquettes.append(list(plaquette))

        self.total_nodes = total_nodes
        assert (
            len(x_plaquettes) == self.n * self.p
        ), "Your number of final x_plaquettes is wrong, maybe weird input?"
        assert len(z_plaquettes) == self.n * self.p + self.d * (
            self.n - 1
        ), "Your number of final z_plaquettes is wrong, maybe weird input?"

        unique_tuples = set()
        for item in z_plaquettes + x_plaquettes:
            unique_tuples.update(item)
        assert self.q_tilde == len(
            list(unique_tuples)
        ), "Q tilde is not the same number as the qubit support of the stabilizers. something is odd"

        # check whether only 1 qubit is encoded
        assert (
            self.q_tilde - (len(z_plaquettes) + len(x_plaquettes)) == 1
        ), "The snake does not encode one logical qubit!!!!"

        return z_plaquettes, x_plaquettes

    def find_separate_stabilizers(self, n_triangle: int) -> list[list[NodePos]]:
        """Generates the stabilizers of a plain triangular color code. Not including the interface.

        Args:
            n_triangle (int): index of triangular patch of interest.

        Returns:
            list[list[NodePos]]: stabilizers of the code (both x and z stabilizers because self-dual)
        """
        plaquettes = []
        hexagonal_plaquettes = self.hex_plaquettes()
        for plaquette in hexagonal_plaquettes:
            overlap = set(plaquette) & set(self.positions[n_triangle])
            if len(overlap) >= 3:  # Ensure a meaningful plaquette (full or partial)
                plaquettes.append(list(overlap))

        assert (
            len(plaquettes) == self.p
        ), "Your number of final triangular color code plaquettes is wrong."
        return plaquettes

    def integer_labeling(self) -> None:
        """Finds a random integer labeling. Only works after having run find_stabilizers."""
        trans_dict = {}
        for i, node in enumerate(self.total_nodes):
            trans_dict.update({node: i})
        self.trans_dict = trans_dict

    def plot_stabilizers(
        self, plaquettes: list[list[NodePos]], size: tuple[int, int] = (7, 7)
    ) -> None:
        """Plots the stabilizers, either z_plaquettes or x_plaquettes.

        Args:
            plaquettes (list[list[NodePos]]): List of stabilizers to plot.
            size (tuple[int, int], optional): size of the plot. Defaults to (7, 7).
        """
        pos = nx.get_node_attributes(self.g, "pos")
        plt.figure(figsize=size)
        nx.draw(
            self.g,
            pos,
            with_labels=True,
            font_size=8,
            node_color="lightgray",
            edge_color="lightblue",
        )

        # integer labels
        if self.trans_dict is None:
            self.integer_labeling()
        if self.trans_dict is not None:
            for original_label, new_label in self.trans_dict.items():
                if original_label in pos:  # Ensure the node exists in the graph
                    x, y = pos[original_label]
                    plt.text(
                        x,
                        y + 0.2,
                        str(new_label),
                        fontsize=8,
                        color="blue",
                        ha="center",
                        va="center",
                    )

        # outline of the triangles
        for i in range(len(self.positions)):
            [lst_corner, _] = self.find_triangle_edges_corners(i)
            # plot three connection lines
            triangle_pos = [pos[node] for node in lst_corner]
            x_coords, y_coords = zip(*triangle_pos)
            plt.plot(
                (*x_coords, x_coords[0]),  # Close the triangle
                (*y_coords, y_coords[0]),
                linewidth=3,
                color="black",
                alpha=0.5,  # Thick and semi-transparent
            )

        colors = plt.cm.rainbow(np.linspace(0, 1, len(plaquettes)))
        for idx, face in enumerate(plaquettes):
            # Get the positions for the vertices in the face
            face_positions = [pos[node] for node in face]

            if len(face_positions) == 2:
                v1, v2 = face_positions[0], face_positions[1]
                line = Line2D(
                    [v1[0], v2[0]], [v1[1], v2[1]], color=colors[idx], lw=4
                )  # 'lw' is line width
                plt.gca().add_line(line)
            else:
                face_positions = convex_hull(face_positions)
                polygon = Polygon(
                    face_positions,
                    closed=True,
                    edgecolor="blue",
                    facecolor=colors[idx],
                    alpha=0.6,
                )
                plt.gca().add_patch(polygon)

        plt.show()
        # !todo store also the pdf of the figure in a given path.

    def gen_check_matrix(self, plaquettes: list[list[NodePos]]) -> npt.NDArray[np.int_]:
        """Takes plaquettes and translates with self.integer_labeling.

        Args:
            plaquettes (list[list[NodePos]]): Stabilizers as list of qubit positions.

        Returns:
            npt.NDArray[np.int_]: check matrix.
        """
        self.integer_labeling()
        h = np.zeros((len(plaquettes), self.q_tilde), dtype=int)
        for row, plaquette in enumerate(plaquettes):
            assert self.trans_dict is not None, "`trans_dict` is not defined yet."
            translated_plaquette = [int(self.trans_dict[node]) for node in plaquette]
            h[row, translated_plaquette] = 1
        return h

    # ------------methods for ZLZL stabilizer subset---------------
    def find_outer_bdry(self) -> list[list[NodePos]]:
        """Finds the set of outer boundaries of the triangles (not connected to stdw) for the inner triangles (start and end triangle not included).

        Returns:
            list[list[NodePos]]: list of the outer boundaries. each boundary is a list of qubit positions NodePos.
        """
        triangles_to_check = list(range(1, len(self.positions) - 1))
        outer_nodes_total = []
        for triangle in triangles_to_check:
            prior_triangle = triangle - 1
            next_triangle = triangle + 1
            [lst_corner, lst_boundary] = self.find_triangle_edges_corners(triangle)
            all_boundary = lst_corner + lst_boundary
            prior_ancilla_pairs = self.find_interface_ancillas(prior_triangle, triangle)
            next_ancilla_pairs = self.find_interface_ancillas(triangle, next_triangle)
            # flatten to nodes
            nodes_stdw = [node for plaq in prior_ancilla_pairs for node in plaq] + [
                node for plaq in next_ancilla_pairs for node in plaq
            ]  # all nodes within the adjacent stdw
            # find those nodes in the triangle boundary which are next to stdw and complement
            inner_nodes = []
            for node in nodes_stdw:
                neighbors = list(self.g.neighbors(node))
                boundary_neighbors = [n for n in neighbors if n in all_boundary]
                inner_nodes += boundary_neighbors
            inner_nodes = list(set(inner_nodes))  # remove duplicates
            outer_nodes = [n for n in all_boundary if n not in inner_nodes]
            # outer_nodes does not include the corners yet, since they may be neighbor to stdw ancilla
            # find the two elements of lst_corner which are closest to the elements in outer_nodes
            closest_corners: list[NodePos] = []
            for node in outer_nodes:
                shortest_paths = nx.single_source_shortest_path_length(
                    self.g, node, cutoff=3
                )  # this would not work for d=3
                closest_corners.extend(
                    corner for corner in lst_corner if corner in shortest_paths
                )
            outer_nodes += list(set(closest_corners))
            outer_nodes = list(set(outer_nodes))  # remove duplicates
            outer_nodes_total.append(outer_nodes)
        self.outer_nodes_total = outer_nodes_total
        return outer_nodes_total

    def fill_triangle(self, triangle_idx: int) -> list[list[NodePos]]:
        """Selects the subset of z stabilizers within the triangle (possibly also including stdw nodes) s.t. each node is maximally touched twice by a plaquette.

        Args:
            triangle_idx (int): The label of the triangle following the numbering of the input of this class.

        Returns:
            list[list[NodePos]]: subset of stabilizers (each a list[NodePos]).
        """
        assert (
            triangle_idx != 0
        ), "filling of triangles only possible if not at the ends of the snake"
        assert (
            triangle_idx != len(self.positions) - 1
        ), "filling of triangles only possible if not at the ends of the snake"
        z_plaquettes, _ = self.find_stabilizers()
        subset_stabs = []

        positions_triangle = self.positions[triangle_idx]
        prior_ancilla_pairs = self.find_interface_ancillas(
            triangle_idx - 1, triangle_idx
        )
        next_ancilla_pairs = self.find_interface_ancillas(
            triangle_idx, triangle_idx + 1
        )
        nodes_stdw = [node for plaq in prior_ancilla_pairs for node in plaq] + [
            node for plaq in next_ancilla_pairs for node in plaq
        ]
        relevant_positions = positions_triangle + nodes_stdw
        filtered_z_plaquettes: list[list[NodePos]] = [
            plaquette
            for plaquette in z_plaquettes
            if any(node in relevant_positions for node in plaquette)
        ]

        [lst_corner, lst_boundary] = self.find_triangle_edges_corners(triangle_idx + 1)
        filtered_z_plaquettes += [lst_corner, lst_boundary]
        [lst_corner, lst_boundary] = self.find_triangle_edges_corners(triangle_idx - 1)
        filtered_z_plaquettes += [lst_corner, lst_boundary]

        triangle_idx -= 1
        for tup in itertools.combinations(self.outer_nodes_total[triangle_idx], 2):
            for plaquette in filtered_z_plaquettes:
                if tup[0] in plaquette and tup[1] in plaquette:
                    subset_stabs.append(plaquette)  # noqa: PERF401

        flag_filler = True
        while flag_filler:
            temp_plaquettes = []
            single_nodes = self._get_single_nodes(subset_stabs)

            assert all(
                value % 2 == 0 for value in Counter(single_nodes).values() if value != 1
            ), "Something went wrong, there are 3,5,... plaquettes touching a node in the chosen stab subset."

            for plaq in subset_stabs:
                single_nodes_in_plaq = [node for node in plaq if node in single_nodes]
                if not single_nodes_in_plaq:
                    continue

                all_neighboring_pairs = self.get_neighboring_pairs(single_nodes_in_plaq)
                all_neighboring_pairs_disjoint = self.filter_disjoint_pairs(
                    all_neighboring_pairs
                )

                for pair in all_neighboring_pairs_disjoint:
                    matching_plaq = self.find_matching_plaquette(
                        pair, filtered_z_plaquettes, subset_stabs
                    )

                    if matching_plaq is not None and len(matching_plaq) != 2:
                        temp_plaquettes.append(matching_plaq)
                        single_nodes = self._get_single_nodes(
                            subset_stabs + temp_plaquettes
                        )

            subset_stabs += temp_plaquettes
            if not temp_plaquettes:
                flag_filler = False

        triangle_idx += 1
        subset_stabs_temp = []
        for plaq in subset_stabs:
            overlap_left = sum(
                1 for node in plaq if node in self.positions[triangle_idx - 1]
            )
            overlap_right = sum(
                1 for node in plaq if node in self.positions[triangle_idx + 1]
            )
            if overlap_left <= 2 and overlap_right <= 2:
                subset_stabs_temp.append(plaq)

        return subset_stabs_temp

    # helper functions for fill_triangle
    @staticmethod
    def _get_single_nodes(subset_stabs: list[list[NodePos]]) -> list[NodePos]:
        """Flatten a list of stabilizers."""
        flattened_nodes = [item for sublist in subset_stabs for item in sublist]
        counts = Counter(flattened_nodes)
        return [key for key, value in counts.items() if value == 1]

    def get_neighboring_pairs(
        self, single_nodes_in_plaq: list[NodePos]
    ) -> list[tuple[NodePos, NodePos]]:
        """Find pairs of single nodes which are neighbors on the graph."""
        return [
            pair
            for pair in itertools.combinations(single_nodes_in_plaq, 2)
            if pair in self.g.edges() and (pair[1], pair[0]) in self.g.edges()
        ]

    @staticmethod
    def filter_disjoint_pairs(
        all_neighboring_pairs: list[tuple[NodePos, NodePos]],
    ) -> list[tuple[NodePos, NodePos]]:
        """Find disjoint pairs of neighboring nodes on a plaquette."""
        disjoint_pairs = []
        for pair in all_neighboring_pairs:
            other_nodes = [
                item
                for sublist in all_neighboring_pairs
                if sublist != pair
                for item in sublist
            ]
            if pair[0] not in other_nodes or pair[1] not in other_nodes:
                disjoint_pairs.append(pair)
        return disjoint_pairs

    @staticmethod
    def find_matching_plaquette(
        pair: tuple[NodePos, NodePos],
        filtered_z_plaquettes: list[list[NodePos]],
        subset_stabs: list[list[NodePos]],
    ) -> list[NodePos] | None:
        """Find plaquette which shares pair but is not in subset_stabs."""
        for plaquette in [
            plaq for plaq in filtered_z_plaquettes if plaq not in subset_stabs
        ]:
            if all(node in plaquette for node in pair):
                return plaquette
        return None

    def find_stabilizers_zz(self) -> list[list[NodePos]]:
        """Summarizes the methods above and joins the subsets per triangle on the STDW."""
        n = len(self.positions)
        subset_z_stabs = []
        self.find_outer_bdry()
        for triangle_idx in range(1, n - 1):
            subset_stabs = self.fill_triangle(triangle_idx)
            subset_z_stabs += subset_stabs
        subset_z_stabs = [list(set(s)) for s in {frozenset(s) for s in subset_z_stabs}]

        # adapt the gaps connecting the snake with the logical patches
        # throw away plaquettes if they go into the logical patch (beyond the boundary nodes), i.e. if a plaquette overlaps with more than 2 nodes in the logical patch
        left_logical = self.positions[0]
        right_logical = self.positions[n - 1]
        subset_z_stabs_temp = []
        for plaq in subset_z_stabs:
            overlap_left_logical = 0
            overlap_right_logical = 0
            for node in plaq:
                if node in left_logical:
                    overlap_left_logical += 1
                if node in right_logical:
                    overlap_right_logical += 1
            if overlap_left_logical > 2 or overlap_right_logical > 2:
                pass
            else:
                subset_z_stabs_temp.append(plaq)
        subset_z_stabs = subset_z_stabs_temp

        # all stdw ancilla positions:
        interface_ancillas = []
        for i in range(n - 1):
            interface_ancillas += self.find_interface_ancillas(i, i + 1)
        interface_ancillas_flat = [
            item for sublist in interface_ancillas for item in sublist
        ]

        # count which nodes have odd number of touches with a stabilizer, if yes add a weight-2
        flattened_nodes = [item for sublist in subset_z_stabs for item in sublist]
        counts = Counter(flattened_nodes)
        for node, count in counts.items():
            if (
                count % 2 != 0
                and node not in left_logical
                and node not in right_logical
            ):  # exclude the logical patches' boundaries, because they should of course not vanish as they constitute the ZL ZL
                # check whether part of interface ancillas, b.c. we only can do corrections there
                assert (
                    node in interface_ancillas_flat
                ), "There is a correction to be done which you cannot do with the current stabilizer construction..."
                # find stdw ancilla neighbor
                neighbors: list[NodePos] = self.g.neighbors(node)
                pair = None
                for neigh in neighbors:
                    if neigh in interface_ancillas_flat:
                        pair = [node, neigh]
                if pair is None:
                    msg = "the zlzl operator cannot be constructed with the given stabilizers"
                    raise RuntimeError(msg)
                # add pair of nodes to subset_z_stabs
                subset_z_stabs.append(pair)

        # final removal of duplicates
        return [list(set(s)) for s in {frozenset(s) for s in subset_z_stabs}]

    def test_zz_stabs(self, subset_z_stabs: list[list[NodePos]]) -> bool:
        """Checks whether all nodes are touched by stabilizers even number of times besides the logical operators."""
        # determine nodes on which the logical operators act.
        n = len(self.positions)
        [lst_corner, lst_boundary] = self.find_triangle_edges_corners(0)
        ancilla_pairs = self.find_interface_ancillas(0, 1)
        ancillas = [item for sublist in ancilla_pairs for item in sublist]
        z_left_nodes: list[NodePos] = []
        for anc in ancillas:
            neighbors = self.g.neighbors(anc)
            z_left_nodes.extend(
                neigh for neigh in neighbors if neigh in lst_corner + lst_boundary
            )

        closest_corners: list[NodePos] = []
        for node in z_left_nodes:
            shortest_paths = nx.single_source_shortest_path_length(
                self.g, node, cutoff=3
            )  # this would not work for d=3
            closest_corners.extend(
                corner for corner in lst_corner if corner in shortest_paths
            )
        z_left_nodes += closest_corners
        z_left_nodes = list(set(z_left_nodes))

        [lst_corner, lst_boundary] = self.find_triangle_edges_corners(n - 1)
        ancilla_pairs = self.find_interface_ancillas(n - 2, n - 1)
        ancillas = [item for sublist in ancilla_pairs for item in sublist]
        z_right_nodes: list[NodePos] = []
        for anc in ancillas:
            neighbors = self.g.neighbors(anc)
            z_right_nodes.extend(
                neigh for neigh in neighbors if neigh in lst_corner + lst_boundary
            )

        closest_corners = []
        for node in z_right_nodes:
            shortest_paths = nx.single_source_shortest_path_length(
                self.g, node, cutoff=3
            )  # this would not work for d=3
            for corner in lst_corner:
                if corner in shortest_paths:
                    closest_corners.append(corner)
        z_right_nodes += closest_corners
        z_right_nodes = list(set(z_right_nodes))

        flattened_nodes = [item for sublist in subset_z_stabs for item in sublist]
        counts = Counter(flattened_nodes)
        for node, count in counts.items():
            if (
                count % 2 != 0
                and node not in z_left_nodes
                and node not in z_right_nodes
            ):
                return False
        return True


class SnakeBuilderSteane:
    """Constructs a snake with n Steane patches on specified vertices in G. Without ancillas in the interface."""

    def __init__(self, g: nx.Graph, positions: list[dict[NodePos, int]]) -> None:
        """Initializes n snake.

        Args:
            g (nx.Graph): Hexagonal graph on which the data qubits are placed
            positions (list[dict[NodePos, int]]): The positions must have key = networkx label, value = 0,..6. The labeling from 0-6 for each
                steane patch (each patch has one dictionary). This follows a strict convention, the order of the overall list is important
                since consecutive dictionaries must have patches neighboring on the lattice. Each patch has three edges: (0,2,1), (3,5,1), (0,4,3).
                Your current patch has to be connected with the next patch via a (3,5,1) -  (0,2,1) or (0,4,3) - (0,2,1) connection. This means
                you always have to `dock` your new patch with its (0,2,1) patch to the previous patch. note that (0,2,1) -(0,2,1) connections are NOT allowed.
                the ordering of 0-6 per patch must follow the convention such that self.standard_steane is consistent.
                     3
                   / | \
                  5--6--4
                 /   |   \
                2----1----0
        """
        self.g = g
        self.positions = positions
        self.n = len(positions)
        for tile in self.positions:
            assert sorted(tile.values()) == list(
                range(7)
            ), "Your 0-6 labeling of each Steane Tile is wrong!"

        # determine global labeling for all vertices in the n-snake
        all_pos = []
        for tile in self.positions:
            all_pos += list(tile.keys())
        labels = {}
        for i, pos in enumerate(all_pos):
            labels.update({pos: i})
        self.labels = labels

    @staticmethod
    def compatible_x_stabs() -> list[dict[str, list[int]]]:
        """Returns the allowed Weight 8 X Stabilizers crossing Steane patches."""
        return [
            {"i": [0, 2, 4, 6], "i+1": [0, 2, 4, 6]},
            {"i": [3, 4, 5, 6], "i+1": [1, 2, 5, 6]},
            {"i": [3, 4, 5, 6], "i+1": [0, 2, 4, 6]},
            {"i": [1, 2, 5, 6], "i+1": [1, 2, 5, 6]},
        ]

    @staticmethod
    def compatible_z_stabs_weight_two() -> list[dict[str, int]]:
        """Returns allowed weight-2 z stabilizer connections."""
        return [
            {"i": 1, "i+1": 1},
            {"i": 0, "i+1": 0},
            {"i": 3, "i+1": 1},
            {"i": 1, "i+1": 3},
            {"i": 3, "i+1": 0},
            {"i": 0, "i+1": 3},
            {"i": 1, "i+1": 0},
            {"i": 0, "i+1": 1},
        ]

    @staticmethod
    def compatible_z_stabs_weight_four() -> list[dict[str, list[int]]]:
        """Returns allowed weight-4 z stabilizers connections."""
        return [
            {"i": [0, 4], "i+1": [0, 2]},
            {"i": [3, 4], "i+1": [1, 2]},
            {"i": [1, 5], "i+1": [1, 2]},
            {"i": [3, 5], "i+1": [0, 2]},
        ]

    @staticmethod
    def standard_steane() -> list[list[int]]:
        """Returns the standard separate steane stabilizer plaquettes."""
        return [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]

    def neighboring_vertex(self, vertex_0: NodePos, vertex_1: NodePos) -> bool:
        """Checks whether two vertices are adjacent."""
        return (vertex_0, vertex_1) in self.g.edges() or (
            vertex_1,
            vertex_0,
        ) in self.g.edges()

    def check_interface(self, i: int) -> dict[NodePos, int]:
        """Checks which edge of the ith steane tile is connected to the next (i+1) 0,2,1 edge."""
        next_edge = [
            key for key, value in self.positions[i + 1].items() if value in {0, 1, 2}
        ]
        # find adjacent edge of ith steane to `next_edge`'s 0,1,2 edge
        adjacent_vertices = (
            set()
        )  # whole set of adjacent vertices to vertices in `next_edge`
        for vertex in next_edge:
            adjacent_vertices.update(self.g.neighbors(vertex))
        # check which nodes from ith steane are in adjacent_vertices
        adjacent_edge = {
            vertex: self.positions[i][vertex]
            for vertex in adjacent_vertices
            if vertex in self.positions[i]
        }
        assert (
            len(adjacent_edge) == 3
        ), "Something with the input steane tiles must be wrong (incorrect number of adjacent vertices. should be 3)"
        return adjacent_edge

    def check_paired_neighbor(
        self, pos_i_new: dict[NodePos, int], pos_i1_new: dict[NodePos, int]
    ) -> bool:
        """Checks whether we can find a weight-8 x plaquette which actually connects neighbored plaqeuttes between i and i+1."""
        neighboring_pairs = []
        # neighboring_two is a bool which determines whether the weight 8 stab would connect neighboring patches
        for key_i in pos_i_new:
            for key_i1 in pos_i1_new:
                # Check if the pair of keys are neighbors using the neighboring_vertex function
                if self.neighboring_vertex(key_i, key_i1):
                    neighboring_pairs.append((key_i, key_i1))  # Store the pair
                    # If we already found 2 pairs, return true
                    if len(neighboring_pairs) == 2:
                        return True
        return False

    def generate_x_stabilizers(self) -> list[dict[NodePos, int]]:
        """Subsequently builds the X stabilizers. Focus on the big `trillerpfeifen` weight-8 stabilizers."""
        x_stabilizers: list[dict[NodePos, int]] = []
        compatible_x_stabs = self.compatible_x_stabs()

        # check at which interface the next steane tile is placed
        adjacent_edge = self.check_interface(0)

        if sorted(adjacent_edge.values()) == [0, 3, 4]:
            k = 0  # make a choice for the two possibilities
            x_stab = {}
            x_stab.update(
                {
                    key: value
                    for key, value in self.positions[0].items()
                    if value in compatible_x_stabs[k]["i"]
                }
            )
            x_stab.update(
                {
                    key: value
                    for key, value in self.positions[1].items()
                    if value in compatible_x_stabs[k]["i+1"]
                }
            )
            x_stabilizers.extend(
                (
                    x_stab,
                    {
                        key: value
                        for key, value in self.positions[0].items()
                        if value in {1, 2, 5, 6}
                    },
                    {
                        key: value
                        for key, value in self.positions[0].items()
                        if value in {3, 4, 5, 6}
                    },
                    {
                        key: value
                        for key, value in self.positions[1].items()
                        if value in {1, 2, 5, 6}
                    },
                    {
                        key: value
                        for key, value in self.positions[1].items()
                        if value in {3, 4, 5, 6}
                    },
                )
            )
        elif sorted(adjacent_edge.values()) == [1, 3, 5]:
            k = 2
            x_stab = {}
            x_stab.update(
                {
                    key: value
                    for key, value in self.positions[0].items()
                    if value in compatible_x_stabs[k]["i"]
                }
            )
            x_stab.update(
                {
                    key: value
                    for key, value in self.positions[1].items()
                    if value in compatible_x_stabs[k]["i+1"]
                }
            )
            x_stabilizers.extend(
                (
                    x_stab,
                    {
                        key: value
                        for key, value in self.positions[0].items()
                        if value in {1, 2, 5, 6}
                    },
                    {
                        key: value
                        for key, value in self.positions[0].items()
                        if value in {0, 2, 4, 6}
                    },
                    {
                        key: value
                        for key, value in self.positions[1].items()
                        if value in {1, 2, 5, 6}
                    },
                    {
                        key: value
                        for key, value in self.positions[1].items()
                        if value in {3, 4, 5, 6}
                    },
                )
            )
        else:
            msg = "Wrong edge connected between Steane patches."
            raise RuntimeError(msg)

        standard_steane_plaquettes = self.standard_steane()
        # remaining steane patches
        for i in range(1, len(self.positions) - 1):
            current_patch = self.positions[i]
            next_patch = self.positions[i + 1]
            for el in compatible_x_stabs:
                # find the present stab which includes el["i"]
                stab = self.find_matching_dict(x_stabilizers, el["i"], i)
                if len(stab) == 4:
                    pos_i_new = {
                        key: value
                        for key, value in current_patch.items()
                        if value in el["i"]
                    }
                    pos_i1_new = {
                        key: value
                        for key, value in next_patch.items()
                        if value in el["i+1"]
                    }
                    neighboring_two = self.check_paired_neighbor(pos_i_new, pos_i1_new)
                    if neighboring_two:
                        x_stabilizers.remove(stab)
                        # add the weight 8 stabilizer
                        x_stab_new = {}
                        x_stab_new.update(
                            {
                                key: value
                                for key, value in current_patch.items()
                                if value in el["i"]
                            }
                        )
                        x_stab_new.update(
                            {
                                key: value
                                for key, value in next_patch.items()
                                if value in el["i+1"]
                            }
                        )
                        temp_occupied_i1 = el[
                            "i+1"
                        ]  # already occupied plaquette on i+1 steane patch
                        x_stabilizers.append(x_stab_new)
                        break
            # add remaining weight 4 stabilizers on i+1
            remainder = [
                element
                for element in standard_steane_plaquettes
                if sorted(element) != sorted(temp_occupied_i1)
            ]
            filtered_result = [
                {key: value for key, value in next_patch.items() if value in plaquette}
                for plaquette in remainder
            ]
            x_stabilizers.extend(filtered_result)

        self.x_stabilizers = x_stabilizers
        return x_stabilizers

    def find_matching_dict(
        self, x_stabilizers: list[dict[NodePos, int]], target_values: list[int], i: int
    ) -> dict[NodePos, int]:
        """Finds the set of stabilizers in the ith patch which have the desired target_values."""
        target_set = set(
            target_values
        )  # Convert target_values to a set for fast lookup
        temp_keys = set(self.positions[i].keys())
        x_stabilizers_temp = [
            stabilizer
            for stabilizer in x_stabilizers
            if not temp_keys.isdisjoint(
                stabilizer.keys()
            )  # Check if there is an intersection
        ]
        candidates = []
        for stabilizer_dict in x_stabilizers_temp:
            dict_values = set(stabilizer_dict.values())  # Convert dict values to set
            # Check if all target values are present in the dict's values
            if target_set.issubset(dict_values):
                candidates.append(stabilizer_dict)
        return min(candidates, key=len)  # the shortest suitable candidate

    def generate_z_stabilizers(self) -> list[dict[NodePos, int]]:
        """Builds Z stabilizers based on X stabilizers."""
        z_stabilizers: list[dict[NodePos, int]] = []

        # first, add all standard stabilizers
        standard_steane_plaquettes = self.standard_steane()
        for tile in self.positions:
            z_stabilizers.extend(
                {key: value for key, value in tile.items() if value in std_stab}
                for std_stab in standard_steane_plaquettes
            )

        # add the weight-4, weight-2 stabilizers depending on the weight-8 X stabilizer's positions
        compatible_weight_four = self.compatible_z_stabs_weight_four()
        compatible_weight_two = self.compatible_z_stabs_weight_two()
        x_stabs_weight_eight = [stab for stab in self.x_stabilizers if len(stab) == 8]
        for i in range(len(self.positions) - 1):  # noqa: PLR1702
            current_patch = self.positions[i]
            next_patch = self.positions[i + 1]
            # find weight-8 stabilizer which connects both patches
            x_stab_connect = self.find_matching_dict_z(
                x_stabs_weight_eight, current_patch, next_patch
            )
            # find the compatible weight two stab sharing qubits with the x_stab_connect
            for weight_two in compatible_weight_two:
                for key, val in current_patch.items():
                    if val == weight_two["i"]:
                        pos_current = key
                for key, val in next_patch.items():
                    if val == weight_two["i+1"]:
                        pos_next = key
                neighbors = self.neighboring_vertex(pos_current, pos_next)
                if x_stab_connect is not None:  # noqa: SIM102
                    if (
                        all(
                            value in x_stab_connect.values()
                            for value in weight_two.values()
                        )
                        and neighbors
                    ):
                        # add weight_two to stabs
                        stab_temp = {}
                        for key, val in current_patch.items():
                            if val == weight_two["i"]:
                                stab_temp.update({key: val})
                        for key, val in next_patch.items():
                            if val == weight_two["i+1"]:
                                stab_temp.update({key: val})
                        z_stabilizers.append(stab_temp)
                        break
            # find the weight4 which must be disjoint to the weight 2 and pairwise neighbors on the lattice
            weight_four = self.find_disjoint_dict(
                current_patch,
                next_patch,
                compatible_weight_four,
                list(weight_two.values()),
            )
            # add to stabs
            stab_temp = {}
            if weight_four is not None:  # mypy
                for label in weight_four["i"]:
                    for key, val in current_patch.items():
                        if label == val:
                            stab_temp.update({key: val})
            if weight_four is not None:  # mypy
                for label in weight_four["i+1"]:
                    for key, val in next_patch.items():
                        if label == val:
                            stab_temp.update({key: val})
            z_stabilizers.append(stab_temp)

        self.z_stabilizers = z_stabilizers
        return z_stabilizers

    @staticmethod
    def find_matching_dict_z(
        x_stabs: list[dict[NodePos, int]],
        current_patch: dict[NodePos, int],
        next_patch: dict[NodePos, int],
    ) -> dict[NodePos, int] | None:
        """Finds the x stabilizer which connects current_patch and next_patch."""
        current_keys = set(current_patch.keys())
        next_keys = set(next_patch.keys())

        for x_stab in x_stabs:
            x_keys = set(x_stab.keys())
            if (
                x_keys & current_keys and x_keys & next_keys
            ):  # Check intersection with both
                return x_stab
        return None  # Return None if no match is found

    def find_disjoint_dict(
        self,
        current_patch: dict[NodePos, int],
        next_patch: dict[NodePos, int],
        dict_list: list[dict[str, list[int]]],
        pair: list[int],
    ) -> dict[str, list[int]] | None:
        """Finds a dictionary in which the total of integers in the values is disjoint to the given pair. AND the weight4 must bepairwise neighbors."""
        pair_set = set(pair)  # Convert the pair to a set for easy comparison

        def neighboring_pair(
            current_patch: dict[NodePos, int],
            next_patch: dict[NodePos, int],
            dictionary: dict[str, list[int]],
        ) -> bool | None:
            """Checks whether at least one neighboring pair in chosen 4-weight stab."""
            for label_i in dictionary["i"]:
                for label_i1 in dictionary["i+1"]:
                    # Find positions in the dictionaries
                    positions_i = [
                        pos for pos, label in current_patch.items() if label == label_i
                    ]
                    positions_i1 = [
                        pos for pos, label in next_patch.items() if label == label_i1
                    ]

                    # Check if any position from i is a neighbor of a position from i+1
                    for pos_i in positions_i:
                        for pos_i1 in positions_i1:
                            if self.g.has_edge(pos_i, pos_i1):  # Check for an edge
                                return True
            return False

        for dictionary in dict_list:
            # Get all integers in the dictionary's values
            value_set = {val for values in dictionary.values() for val in values}
            # Check if the two sets are disjoint
            if pair_set.isdisjoint(value_set) and neighboring_pair(
                current_patch, next_patch, dictionary
            ):
                return dictionary

        return None

    def translate_checks(self) -> tuple[list[list[int]], list[list[int]]]:
        """Translates the x/z_stabilizers into check matrices."""
        # translate stabilizers in lists of global labels
        x_stabs_temp: list[list[int]] = []
        z_stabs_temp: list[list[int]] = []
        for stab in self.x_stabilizers:
            temp = [self.labels[el] for el in stab]
            x_stabs_temp.append(temp)
        for stab in self.z_stabilizers:
            temp = [self.labels[el] for el in stab]
            z_stabs_temp.append(temp)

        checks_x = []
        checks_z = []
        for stabt in x_stabs_temp:
            check_temp = [0] * len(self.labels)
            for el in stabt:
                check_temp[el] = 1
            checks_x.append(check_temp.copy())
        for stabt in z_stabs_temp:
            check_temp = [0] * len(self.labels)
            for el in stabt:
                check_temp[el] = 1
            checks_z.append(check_temp.copy())

        return checks_z, checks_x

    def plot_stabilizers(
        self, stabilizers: list[dict[NodePos, int]], size: tuple[int, int] = (7, 7)
    ) -> None:
        """Plots the faces of the stabilizers for a given list of stabilizers (either x or z)."""
        pos = nx.get_node_attributes(self.g, "pos")
        num_faces = len(stabilizers)
        colors = cm.rainbow(
            np.linspace(0, 1, num_faces)
        )  # Generate colors from the rainbow palette

        plt.figure(figsize=size)
        nx.draw(
            self.g,
            pos,
            with_labels=True,
            node_color="lightgray",
            edge_color="lightblue",
            font_size=8,
        )

        for original_label, new_label in self.labels.items():
            if original_label in pos:  # Ensure the node exists in the graph
                x, y = pos[original_label]
                plt.text(
                    x,
                    y + 0.15,
                    str(new_label),
                    fontsize=8,
                    color="blue",
                    ha="center",
                    va="center",
                )

        for face_vertices, color in zip(
            [list(el.keys()) for el in stabilizers], colors
        ):
            if len(face_vertices) == 2:  # Check if it's a digon
                v1, v2 = [pos[label] for label in face_vertices]
                line = Line2D(
                    [v1[0], v2[0]], [v1[1], v2[1]], color=color, lw=4
                )  # 'lw' is line width
                plt.gca().add_line(line)
            else:
                # Order vertices and compute the convex hull for other faces
                face_coords = convex_hull([pos[label] for label in face_vertices])
                polygon = Polygon(
                    face_coords,
                    closed=True,
                    edgecolor="black",
                    facecolor=color,
                    alpha=0.7,
                )
                plt.gca().add_patch(polygon)

        plt.gca().set_aspect(
            "equal"
        )  # Ensure the aspect ratio is equal for proper visualization
        plt.show()


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Find the convex hull of a set of 2D points."""
    points = sorted(points)

    def cross(
        o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]
