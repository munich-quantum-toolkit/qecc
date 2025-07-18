# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Routing for Hexagonal Lattices."""

from __future__ import annotations

import collections
import copy
import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class HexagonalLattice:
    """Hexagonal Lattice with Distance Metric."""

    def __init__(self, m: int, n: int) -> None:
        """Generates the connectivity lattice of the logical qubits.

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
        """
        self.m = m
        self.n = n
        self.G = nx.hexagonal_lattice_graph(m=m, n=n, periodic=False, with_positions=True, create_using=None)
        self.G_copy = copy.deepcopy(self.G)

    def map_hex_to_triangular(self) -> dict[tuple[int, int], tuple[int, int, int]]:
        """Maps positions of hex lattice to dual triangular lattice.

        maps the positions of the G lattice to a triangular lattice such that
        distances can be measured. follows along
        https://github.com/mhwombat/grid/wiki/Implementation:-Triangular-tiles

        Returns:
            dict: dictionary which maps between networkx labels key (x,y) to
                    triangular labels (x,y,z)
        """
        # generate the triangular lattice of the correct size
        max_x_tilde = max(el[0] for el in nx.get_node_attributes(self.G, "pos"))
        max_y_tilde = max(el[1] for el in nx.get_node_attributes(self.G, "pos"))
        full_max = max(max_x_tilde, max_y_tilde)

        # initialize large enough (double proof) such that only positive
        # integers can appear in triangular map
        dct = {(0, 0): (2 * full_max, 0, -2 * full_max)}
        start_x = copy.deepcopy(dct[0, 0][0])
        start_y = copy.deepcopy(dct[0, 0][1])
        for i, x_t in enumerate(range(max_x_tilde + 1)):
            y_range = range(1, max_y_tilde + 1) if i == 0 else range(max_y_tilde + 1)
            for y_t in y_range:
                if y_t != 0:
                    start_x -= 1
                    start_y += 1
                z = -start_x - start_y if start_y % 2 == 0 else -start_x - start_y + 1
                if self.G_copy.has_node((x_t, y_t)):
                    dct.update({
                        (x_t, y_t): (
                            copy.deepcopy(start_x),
                            copy.deepcopy(start_y),
                            copy.deepcopy(z),
                        )
                    })

            if x_t != max_x_tilde:
                start_x = copy.deepcopy(dct[x_t, 0][0]) + 1
                start_y = copy.deepcopy(dct[x_t, 0][1]) + 1

        return dct

    def distance_triangular(self, pos_1: tuple[int, int], pos_2: tuple[int, int]) -> int:
        """Determines distance considering the triangular dual lattice.

        Args:
            pos_1 (tuple): position 1 on networkx graph G
            pos_2 (tuple): position 2 on networkx graph G

        Returns:
            int: distance between pos_1 and pos_2
        """
        assert all(isinstance(x, int) for x in pos_1), "Each entry in pos_1 must be an integer!"

        assert all(isinstance(x, int) for x in pos_2), "Each entry in pos_2 must be an integer!"

        dct = self.map_hex_to_triangular()
        mapped_1 = dct[pos_1]
        mapped_2 = dct[pos_2]
        lst = [abs(mapped_1[i] - mapped_2[i]) for i in range(len(mapped_1))]
        assert len(mapped_1) == len(mapped_2), "Something went wrong in the triangular mapping"
        return int(max(lst))

    def gen_layout_sparse(self) -> list[tuple[int, int]]:
        """Generates Sparse Layout (data qubit locations without qubit labels).

        Returns:
            list[tuple[int, int]]: Locations on the graph for data qubits (no qubit labels assigned yet)
        """
        data_qubit_locs = []  # start with (x,y) = 1,2
        for x in range(1, self.n + 1):  # no data qubits on x=0 to ensure free boundary
            if (x + 1, 0) not in list(self.G.nodes) and (x + 1, 1) not in list(self.G.nodes):
                break
            for y in np.arange(2, self.m * 2 + 1, 6):
                y_temp = y
                if x % 2 == 0:
                    y_temp += 3
                if (x, y_temp + 1) in list(self.G.nodes) and (x, y_temp + 2) in list(self.G.nodes):
                    data_qubit_locs.append((x, y_temp))
                else:
                    break
        return data_qubit_locs

    def gen_layout_pair(self) -> list[tuple[int, int]]:
        """Generates Pair Layout (data qubit locations without qubit labels).

        Returns:
            list[tuple[int, int]]: Locations on the graph for data qubits (no qubit labels assigned yet)
        """
        data_qubit_locs = []  # start with (x,y) = 1,2
        for x in np.arange(1, self.n + 1, 2):  # no data qubits on x=0 to ensure free boundary
            if (x + 1, 0) not in list(self.G.nodes) and (x + 1, 1) not in list(self.G.nodes):
                break
            for y in np.arange(2, self.m * 2 + 1, 4):
                if (x, y + 1) in list(self.G.nodes) and (x, y + 2) in list(self.G.nodes):
                    data_qubit_locs.append((x, y))
                else:
                    break
                if (x, y + 2) in list(self.G.nodes) and (x, y + 3) in list(self.G.nodes):
                    data_qubit_locs.append((x, y + 1))
                else:
                    break
        return data_qubit_locs

    def gen_layout_row(self) -> list[tuple[int, int]]:
        """Generates Row Layout (data qubit locations without qubit labels).

        Returns:
            list[tuple[int, int]]: Locations on the graph for data qubits (no qubit labels assigned yet)
        """
        min_x = 1
        max_x = self.n + 1
        min_y = 2
        max_y = self.m * 2

        data_qubit_locs = []  # start with (x,y) = 1,2
        for y in np.arange(min_y, max_y, 4):
            flag_x1 = True
            flag_x2 = True
            for x in range(min_x, max_x):
                if (x, y + 1) not in list(self.G.nodes) or (x, y + 2) not in list(self.G.nodes):
                    break
                if (x + 1, y) in list(self.G.nodes):
                    data_qubit_locs.append((x, y))
                else:
                    flag_x1 = False
                if (x + 1, y + 1) in list(self.G.nodes):
                    data_qubit_locs.append((x, y + 1))
                else:
                    flag_x2 = False
                if not flag_x1 and not flag_x2:
                    break
        return data_qubit_locs

    def gen_layout_hex(self) -> list[tuple[int, int]]:
        """Generates Hexagon Layout (data qubit locations without qubit labels).

        This may not be fully general for arbitrarily sized lattices. but for the sizes we consider it suffices.

        Returns:
            list[tuple[int, int]]: Locations on the graph for data qubits (no qubit labels assigned yet)
        """
        start_hex = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

        start_hex_lst = [start_hex.copy()]
        # vertical shift y by 14 until and add if hex still inside the grid
        in_lat = True
        while in_lat:
            temp_hex = [(el[0], el[1] + 14) for el in start_hex]
            # check whether nodes in lattice
            missing_nodes = [node for node in temp_hex if node not in self.G]
            if len(missing_nodes) != 0:
                in_lat = False
                break
            start_hex_lst.append(temp_hex)
            start_hex = temp_hex

        # horizontal shift
        in_lat = True
        start_hex = start_hex_lst[0].copy()
        while in_lat:
            temp_hex = [(el[0] + 3, el[1] + 1) for el in start_hex]
            missing_nodes = [node for node in temp_hex if node not in self.G]
            if missing_nodes:
                in_lat = False
                break
            # check whether lower hexagon also available
            temp_hex_lower = [(el[0] - 1, el[1] - 5) for el in temp_hex]
            missing_nodes = [node for node in temp_hex_lower if node not in self.G]
            if missing_nodes:
                start_hex_lst.append(temp_hex)
                start_hex = temp_hex
            else:
                start_hex_lst.append(temp_hex_lower)
                start_hex = temp_hex_lower

        # go from each element in start_hex_lst to the right until the lattice ends
        final_hex_lst = start_hex_lst.copy()
        for start_hex in start_hex_lst:
            in_lat = True
            start_hex_t = start_hex
            while in_lat:
                temp_hex = [(el[0] + 1, el[1] + 5) for el in start_hex_t]
                # check whether nodes in lattice
                missing_nodes = [node for node in temp_hex if node not in self.G]
                if missing_nodes:
                    in_lat = False
                    break
                final_hex_lst.append(temp_hex)
                start_hex_t = temp_hex

        filtered_hex = []
        for hexagon in final_hex_lst:
            min_neighbors = [len(list(self.G.neighbors(node))) >= 3 for node in hexagon]
            if all(min_neighbors):
                filtered_hex.append(hexagon)

        # flatten
        return [el for sublist in filtered_hex for el in sublist]

    def plot_lattice(
        self,
        size: tuple[float, float] = (3.5, 3.5),
        data_qubit_locs: list[tuple[int, int]] | None = None,
        factory_locs: list[tuple[int, int]] | None = None,
    ) -> None:
        """Plots the lattice G with networkx labels."""
        if data_qubit_locs is None:
            data_qubit_locs = []
        if factory_locs is None:
            factory_locs = []
        pos = nx.get_node_attributes(self.G, "pos")

        plt.figure(figsize=size)
        nx.draw(self.G, pos, with_labels=True, font_size=8, node_color="lightgray", edge_color="lightblue")

        if len(data_qubit_locs) != 0:
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=data_qubit_locs,
                node_color="orange",
            )

        if len(factory_locs) != 0:
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=factory_locs,
                node_color="violet",
            )


class ShortestFirstRouter(HexagonalLattice):
    """Shortest First Routing for VDP on Hexagonal Lattice."""

    # ! possibly remove this class because we only care about the ones with T gates and here some basic things are not fully correct.

    def __init__(self, m: int, n: int, terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> None:
        """Routing for Hexagonal Lattice.

        Start with graph $G$ and an empty solution.
        While $G$ contains any path connecting any demand pair,
        choose the shortest such path $P$, add $P$ to the solution,
        and delete all vertices of $P$ from $G$

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
            terminal_pairs (list[tuple[tuple[int, int], tuple[int, int]]]): pairs of vertices to be connected (networkx labeling),

        """
        super().__init__(m, n)
        self.terminal_pairs_orig = terminal_pairs.copy()
        self.terminal_pairs = terminal_pairs
        self.layers_cnots = self.split_layer_terminal_pairs()
        self.layers_cnots_orig = self.layers_cnots.copy()
        # self.vdp_layers = self.find_total_vdp_layers()
        self.vdp_layers: list[dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]] | None = None

    def split_layer_terminal_pairs(self) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
        """Split Terminal Pairs into layers initially.

        split up the terminal pairs into layers which can be
        compiled in parallel in principle because no qubits overlap
        """
        layers: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
        current_layer: list[tuple[tuple[int, int], tuple[int, int]]] = []
        used_qubits: set[tuple[int, int]] = set()

        for pair in self.terminal_pairs:
            if pair[0] in used_qubits or pair[1] in used_qubits:
                layers.append(current_layer)
                current_layer = [pair]
                used_qubits = set(pair)
            else:
                current_layer.append(pair)
                used_qubits.update(pair)

        if current_layer:
            layers.append(current_layer)

        return layers

    def measure_terminal_pair_distances(self) -> list[int]:
        """Compute the plain distance between all the terminal pairs.

        Returns:
            list[int]: Distances between terminal pairs (same order as self.terminal_pairs)
        """
        lst_distances = []
        for t_p in self.terminal_pairs_orig:
            p1, b1 = tuple(int(i) for i in t_p[0])
            p2, b2 = tuple(int(i) for i in t_p[1])
            tp1 = (p1, b1)
            tp2 = (p2, b2)
            d = self.distance_triangular(tp1, tp2)
            lst_distances.append(d)
        return lst_distances

    def count_crossings_per_layer(self) -> list[int]:
        """Counts the crossings of the simple paths (respecting terminals) per layer.

        Returns:
            list[int]: Number of crossings per initial layer. len is len(self.layers_cnots_orig)
        """
        lst_crossings = []
        for layer in self.layers_cnots_orig:
            paths = []
            for t_p in layer:
                g_temp = self.G.copy()  # this is duplicate code from `order_terminal_pairs`
                terminal_pairs_flattened = [pair for sublist in layer for pair in sublist]
                terminals_temp = [pair for pair in terminal_pairs_flattened if pair != t_p[0] and pair != t_p[1]]
                terminals_temp = list(set(terminals_temp))
                g_temp.remove_nodes_from(terminals_temp)
                try:
                    path = nx.dijkstra_path(g_temp, t_p[0], t_p[1])
                    paths.append(path)
                except nx.NetworkXNoPath as exc:
                    msg = (
                        "Your choice of terminal pairs locks in at least one terminal. "
                        "Reconsider your choice of terminal pairs."
                    )
                    raise ValueError(msg) from exc
            # check the paths for overlaps
            # Create a mapping of elements to the sublists they appear in
            element_to_sublists = collections.defaultdict(set)
            for i, sublist in enumerate(paths):
                for element in sublist:
                    element_to_sublists[element].add(i)
            # Count crossings (pairwise sublist overlaps for each element)
            crossing_count = 0
            for sublists in element_to_sublists.values():
                if len(sublists) > 1:
                    crossing_count += len(list(itertools.combinations(sublists, 2)))
            lst_crossings.append(crossing_count)
        return lst_crossings

    def order_terminal_pairs(self, layer: int) -> None:
        """Orders terminal pairs of a layer inplace.

        order the terminal pairs s.t. the pairs
        closest together are routed first
        adapts self.terminal_pairs in place
        """
        terminal_pair_dist = {}
        for t_p in self.layers_cnots_orig[layer]:  # self.terminal_pairs_orig:
            # paths must be found excluding other terminals
            g_temp = self.G.copy()
            terminal_pairs_flattened = [
                pair
                for sublist in self.layers_cnots_orig[layer]  # self.terminal_pairs_orig
                for pair in sublist
            ]
            terminals_temp = [pair for pair in terminal_pairs_flattened if pair != t_p[0] and pair != t_p[1]]
            terminals_temp = list(set(terminals_temp))
            g_temp.remove_nodes_from(terminals_temp)
            try:
                path = nx.dijkstra_path(g_temp, t_p[0], t_p[1])
            except nx.NetworkXNoPath as exc:
                msg = (
                    "Your choice of terminal pairs locks in at least one terminal. "
                    "Reconsider your choice of terminal pairs."
                )
                raise ValueError(msg) from exc
            terminal_pair_dist.update({
                t_p: len(path) - 1
            })  # -1 because we want to count only what is between the terminals
        sorted_terminal_pairs = sorted(terminal_pair_dist.keys(), key=lambda tp: terminal_pair_dist[tp])
        # self.terminal_pairs = sorted_terminal_pairs
        self.layers_cnots_orig[layer] = sorted_terminal_pairs
        self.layers_cnots[layer] = sorted_terminal_pairs

    def find_max_vdp_set(
        self, layer: int
    ) -> tuple[
        dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]],
        list[tuple[tuple[int, int], tuple[int, int]]],
    ]:
        """Find largest VDP with shortest first.

        iteratively applies dijkstra and searches greedily the largest
        possible VDP set in this setting

        Returns:
            dict: path per terminal pair
            list[tuple[int,int]]: remaining terminal pairs which must be placed
                in a new layer
        """
        vdp_dict = {}
        terminal_pairs_remainder = []
        successful_terminals = []  # gather successful terminal pairs
        flag_problem = False
        g_temp = self.G.copy()
        # a dct which checks whether a qubit
        # was already used in the current layer
        dct_qubits = {}
        terminal_pairs_orig_current = self.layers_cnots_orig[layer].copy()
        terminal_pairs_current = self.layers_cnots[layer].copy()
        terminal_pairs_flattened = [pair for sublist in terminal_pairs_orig_current for pair in sublist]
        for t in terminal_pairs_flattened:
            dct_qubits.update({t: False})
        dct_qubits_copy = dct_qubits.copy()
        for t_p in terminal_pairs_current:
            # path must be found excluding other terminals
            g_temp_temp = g_temp.copy()
            if dct_qubits[t_p[0]] or dct_qubits[t_p[1]]:
                flag_problem = True
            else:
                terminals_temp = [pair for pair in terminal_pairs_flattened if pair != t_p[0] and pair != t_p[1]]
                terminals_temp = list(set(terminals_temp))
                g_temp_temp.remove_nodes_from(terminals_temp)
                # find shortest path of t_p
                try:
                    path = nx.dijkstra_path(g_temp_temp, t_p[0], t_p[1])
                except nx.NetworkXNoPath:
                    # if no path could be found: stop and return remaining,
                    # unallocated terminal pairs as well
                    """
                    terminal_pairs_remainder = [
                        s
                        for s in self.terminal_pairs
                        if s not in successful_terminals
                    ]
                    """
                    flag_problem = True
                    # break
            # update already used qubits
            dct_qubits[t_p[0]] = True
            dct_qubits[t_p[1]] = True
            if flag_problem:
                terminal_pairs_remainder = [s for s in terminal_pairs_current if s not in successful_terminals]
                dct_qubits = dct_qubits_copy.copy()
            else:  # if no problem
                # remove nodes and edges from G
                # but only remove the path vertices, NOT the terminals
                # because the terminals might be used multiple times
                for node in path[1:-1]:
                    g_temp.remove_node(node)
                successful_terminals.append(t_p)
                vdp_dict.update({t_p: path})

        return vdp_dict, terminal_pairs_remainder

    def find_all_vdp_layers(
        self, layer: int
    ) -> list[dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        """Find VDP layers within a given initial layer.

        if find_max_VDP_set returns nonzero terminal_pairs_remainder
        it is required to run the algorithm as long s.t. we find all VDP
        sets even if they are in multiple layers

        Returns:
            list[dict]: list of layers with simultaneous paths (VDP per layer)
        """
        flag_continue = True
        vdp_layers = []
        while flag_continue:
            vdp_dict, terminal_pairs_remainder = self.find_max_vdp_set(layer)
            vdp_layers.append(vdp_dict)
            if len(terminal_pairs_remainder) == 0:
                # flag_continue = False
                break
            self.layers_cnots[layer] = terminal_pairs_remainder

        return vdp_layers

    def find_total_vdp_layers(self) -> list[dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        """Find all routes for all initial and secondary layers.

        finds total VDP layers, i.e. more than `all` meaning that
        it also respects the initial layer structure of the cnots
        """
        vdp_layers = []
        for layer in range(len(self.layers_cnots_orig)):
            self.order_terminal_pairs(layer)
            vdp_layers_temp = self.find_all_vdp_layers(layer)
            vdp_layers += vdp_layers_temp
            # it might be possible that there are bugs. hence, check whether vdp layers really contains as main paths as there are gates.
        keys: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for lst in vdp_layers:
            keys += lst.keys()
        assert len(keys) == len(self.terminal_pairs), (
            f"The static routing has a bug. There are {len(self.terminal_pairs)} to be routed, but the final vdp_layers only has {len(keys)} paths."
        )
        return vdp_layers

    def plot_lattice_paths(
        self, layer: int, layout: dict[int, tuple[int, int]] | None = None, size: tuple[float, float] = (3.5, 3.5)
    ) -> None:
        """Plots the graph and the corresponding VDP of a layer.

        Args:
            layer (int): label of layer to plot
            layout (dict): potentially also display the qubit labels. keys = qubit label, value = node label
            size (tuple[float,float], optional): _description_. Size of the plot. Defaults to (3.5,3.5).
        """
        if layout is None:
            layout = {}
        pos = nx.get_node_attributes(self.G, "pos")

        if self.vdp_layers is not None:  # for mypy
            num_paths = len(self.vdp_layers[layer].keys())
            colormap = plt.cm.get_cmap("rainbow", num_paths)
            colors = [mcolors.to_hex(colormap(i)) for i in range(num_paths)]
        else:
            msg = "vdp_layers must be present, otherwise plotting makes no sense."
            raise RuntimeError(msg)

        plt.figure(figsize=size)
        nx.draw(self.G, pos, with_labels=True, node_color="gray", edge_color="lightblue", font_size=8)

        if self.vdp_layers is not None:
            for i, path in enumerate(self.vdp_layers[layer].values()):
                if path:
                    path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
                    nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, width=2, edge_color=colors[i])
                    nx.draw_networkx_nodes(self.G, pos, nodelist=path, node_color=colors[i], label=f"Path {i + 1}")

        if len(list(layout.keys())) != 0:
            for key, value in layout.items():
                node_pos = pos[value]
                plt.text(
                    node_pos[0], node_pos[1] - 0.1, str(key), fontsize=8, color="white", horizontalalignment="center"
                )
                # also highlight data qubits
                nx.draw_networkx_nodes(
                    self.G,
                    pos,
                    nodelist=layout.values(),  # Nodes to highlight
                    node_color="none",  # Unfilled circles
                    edgecolors="lime",  # Neon green outline
                    linewidths=1.5,  # Line width for the outline
                )

        plt.legend()
        plt.show()


class ShortestFirstRouterTGates(HexagonalLattice):
    """Shortest First Routing for VDP on Hexagonal Lattice with adaption to greedily include T gates."""

    def __init__(
        self,
        m: int,
        n: int,
        terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]],
        factory_positions: list[tuple[int, int]],
        t: int,
    ) -> None:
        """Routing for Hexagonal Lattice with adaption to greedily include T gates.

        Start with graph $G$ and an empty solution.
        While $G$ contains any path connecting any demand pair,
        choose the shortest such path $P$, add $P$ to the solution,
        and delete all vertices of $P$ from $G$

        T gates are included by including the shortest connection between any factory site
        and the respective qubit for the initial ordering. After the ordering, handle T gates
        similar to CNOTs just that we have to check which factories are available + it may be
        necessary to wait 1 or more layers until a T factory becomes available.

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
            terminal_pairs (list[tuple[tuple[int, int], tuple[int, int]]): pairs of vertices to be connected (networkx labeling)
            factory_positions (list[tuple[int,int]]): Positions were factories are placed (should be on the boundary of the data qubits), follows networkx labeling
            t (int): A factory needs t logical time steps of our scheme to generate a new T state
        """
        super().__init__(m, n)
        self.terminal_pairs = terminal_pairs
        self.t = t
        # check that factory_positions do not overlap with data qubit positions
        flattened_terminals = [
            pair for item in terminal_pairs for pair in (item if isinstance(item[0], tuple) else [item])
        ]
        self.flattened_terminals = flattened_terminals
        for factory in factory_positions:
            assert factory not in flattened_terminals, "Your factory positions overlap with data qubits!"
        self.factory_positions = factory_positions
        self.factory_times = {}
        for factory in factory_positions:
            self.factory_times.update({factory: t})
        self.layers_cnot_t = self.split_layer_terminal_pairs()
        self.layers_cnot_t_orig = self.layers_cnot_t.copy()
        self.layers_copy = self.layers_cnot_t.copy()

    def count_crossings_per_layer(self, t_crossings: bool = False) -> list[int]:
        """Counts the crossings of the simple paths between cnots and between shortest factory to qubit path (respecting terminals and factory positions) per layer.

        Args:
            t_crossings (bool): decides whether the crossings to the factory are included (true) or not (false).

        Returns:
            list[int]: Number of crossings per initial layer. len is len(self.layers_cnot_t_orig)
        """
        # ! TODO: remove redundancies (order_terminal_pairs very similar)
        lst_crossings = []
        flattened_terminals_and_factories = self.flattened_terminals.copy() + self.factory_positions.copy()
        for layer in self.layers_cnot_t_orig:
            paths = []
            for t_p in layer:
                g_temp = self.G.copy()
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                    terminals_temp = [
                        pair for pair in flattened_terminals_and_factories.copy() if pair != t_p[0] and pair != t_p[1]
                    ]
                    terminals_temp = list(set(terminals_temp))
                    g_temp.remove_nodes_from(terminals_temp)
                    try:
                        path = nx.dijkstra_path(g_temp, t_p[0], t_p[1])
                        paths.append(path)
                    except nx.NetworkXNoPath as exc:
                        msg = (
                            "Your choice of terminal pairs locks in at least one terminal. "
                            "Reconsider your choice of terminal pairs."
                        )
                        raise ValueError(msg) from exc

                elif t_crossings:
                    dist_factories = {}  # gather distances to each factory to greedily choose the shortest path
                    for factory in self.factory_positions:
                        g_temp = self.G.copy()
                        terminals_temp = [
                            pair for pair in flattened_terminals_and_factories.copy() if pair not in {t_p, factory}
                        ]
                        terminals_temp = list(set(terminals_temp))
                        g_temp.remove_nodes_from(terminals_temp)
                        try:
                            path = nx.dijkstra_path(g_temp, t_p, factory)
                        except nx.NetworkXNoPath as exc:
                            msg = (
                                "Your choice of terminal pairs locks in at least one terminal. "
                                "Reconsider your choice of terminal pairs."
                            )
                            raise ValueError(msg) from exc
                        dist_factories.update({factory: path})
                    # choose shortest factory path
                    nearest_factory = min(dist_factories, key=lambda k: len(dist_factories[k]))
                    paths.append(dist_factories[nearest_factory])
            # check the paths for overlaps
            # Create a mapping of elements to the sublists they appear in
            element_to_sublists = collections.defaultdict(set)
            for i, sublist in enumerate(paths):
                for element in sublist:
                    element_to_sublists[element].add(i)
            # Count crossings (pairwise sublist overlaps for each element)
            crossing_count = 0
            for sublists in element_to_sublists.values():
                if len(sublists) > 1:
                    crossing_count += len(list(itertools.combinations(sublists, 2)))
            lst_crossings.append(crossing_count)
        return lst_crossings

    def split_layer_terminal_pairs(self) -> list[list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]]]:
        """Split Terminal Pairs into layers initially.

        split up the terminal pairs into layers which can be
        compiled in parallel in principle because no qubits overlap
        (can handle both pairs of qubits and single qubits)
        """
        layers = []
        current_layer: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = []
        used_qubits = set()

        for pair in self.terminal_pairs:
            if isinstance(pair[0], tuple) and isinstance(pair[1], tuple):
                if pair[0] in used_qubits or pair[1] in used_qubits:
                    layers.append(current_layer)
                    current_layer = [pair]
                    used_qubits = set(pair)
                else:
                    current_layer.append(pair)
                    used_qubits.update(pair)
            elif isinstance(pair[1], int):
                if pair in used_qubits:
                    layers.append(current_layer)
                    current_layer = [pair]
                    used_qubits = {pair}
                else:
                    current_layer.append(pair)
                    used_qubits.update([pair])

        if current_layer:
            layers.append(current_layer)

        return layers

    def order_terminal_pairs(self, layer: int) -> None:
        """Orders terminal pairs of a layer inplace.

        order the terminal pairs s.t. the pairs
        closest together are routed first (looks for shortest route to the factories if T gate)
        adapts self.terminal_pairs in place
        """
        terminal_pair_dist: dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], int] = {}
        for t_p in self.layers_cnot_t_orig[layer]:
            g_temp = self.G.copy()
            flattened_terminals_and_factories = self.flattened_terminals.copy() + self.factory_positions.copy()
            if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                terminals_temp = [
                    pair for pair in flattened_terminals_and_factories.copy() if pair != t_p[0] and pair != t_p[1]
                ]
                terminals_temp = list(set(terminals_temp))
                g_temp.remove_nodes_from(terminals_temp)
                try:
                    path = nx.dijkstra_path(g_temp, t_p[0], t_p[1])
                except nx.NetworkXNoPath as exc:
                    msg = (
                        "Your choice of terminal pairs locks in at least one terminal. "
                        "Reconsider your choice of terminal pairs."
                    )
                    raise ValueError(msg) from exc
                terminal_pair_dist.update({
                    t_p: len(path) - 1
                })  # -1 because we want to count only what is between the terminals

            elif isinstance(t_p[1], int):
                dist_factories = {}  # gather distances to each factory to greedily choose the shortest path
                for factory in self.factory_positions:
                    g_temp = self.G.copy()
                    terminals_temp = [
                        pair for pair in flattened_terminals_and_factories.copy() if pair not in {t_p, factory}
                    ]
                    terminals_temp = list(set(terminals_temp))
                    g_temp.remove_nodes_from(terminals_temp)
                    path = nx.dijkstra_path(g_temp, t_p, factory)
                    dist_factories.update({factory: len(path) - 1})
                # choose shortest factory path
                nearest_factory = min(dist_factories, key=lambda x: dist_factories.get(x, 0))
                # add corresponding distance to terminal_pair_dist
                terminal_pair_dist.update({t_p: dist_factories[nearest_factory]})

            # order the layer according to terminal_pair_dist
            sorted_terminal_pairs = sorted(terminal_pair_dist.keys(), key=lambda tp: terminal_pair_dist[tp])
            self.layers_cnot_t_orig[layer] = sorted_terminal_pairs
            self.layers_cnot_t[layer] = sorted_terminal_pairs
            self.terminal_pair_dist = terminal_pair_dist

    def find_max_vdp_set(
        self, layer: int
    ) -> tuple[
        dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]],
        list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]],
    ]:
        """Find largest VDP with shortest first.

        iteratively applies dijkstra according to ordering from `order_terminal_pairs`.
        for T gates, paths to all available factories are computed, shortest path is taken
        factory is updated on waiting mode, according to self.t.

        change: 250307: for each gate the dijkstra paths are computed again and the shortest path is taken. no predetermined order of the gates as before

        Returns:
            dict: path per terminal pair
            list[tuple[int,int]]: remaining terminal pairs which must be placed
                in a new layer
        """
        vdp_dict: dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {}
        terminal_pairs_remainder = []
        successful_terminals = []  # gather successful terminal pairs
        flag_problem = False
        g_temp = self.G.copy()
        dct_qubits = {}  # a dct which checks whether a qubit was already used in the layer
        terminal_pairs_orig_current = self.layers_cnot_t_orig[layer].copy()
        terminal_pairs_current = self.layers_cnot_t[layer].copy()
        flattened_terminals = [
            pair for item in terminal_pairs_orig_current for pair in (item if isinstance(item[0], tuple) else [item])
        ]
        for t in flattened_terminals:
            dct_qubits.update({t: False})
        dct_qubits_copy = dct_qubits.copy()
        flattened_terminals_and_factories = self.flattened_terminals.copy() + self.factory_positions.copy()

        while len(terminal_pairs_current) > 0 and flag_problem is False:  # noqa: PLR1702
            paths_temp_lst = []  # gather all possible paths here, between all terminal pairs (cnots) and between all qubits for a tgate with all factories
            tp_list: list[
                tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ] = []  # same order, actually redundant but error otherwise
            for t_p in terminal_pairs_current:
                g_temp_temp = g_temp.copy()
                # cnot
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                    if dct_qubits[t_p[0]] or dct_qubits[t_p[1]]:
                        flag_problem = True
                        break
                    terminals_temp = [
                        pair for pair in flattened_terminals_and_factories.copy() if pair != t_p[0] and pair != t_p[1]
                    ]
                    terminals_temp = list(set(terminals_temp))
                    g_temp_temp.remove_nodes_from(terminals_temp)
                    # find shortest path of t_p
                    try:
                        path = nx.dijkstra_path(g_temp_temp, t_p[0], t_p[1])
                        paths_temp_lst.append(path)
                        tp_list.append(t_p)
                    except nx.NetworkXNoPath:
                        pass

                # t gate
                elif isinstance(t_p[1], int):
                    if dct_qubits[t_p]:
                        flag_problem = True
                        break
                    dist_factories = {}
                    for factory in self.factory_positions:
                        g_temp_temp = g_temp.copy()
                        if self.factory_times[factory] == 0:  # only include available factories
                            # remove other terminals
                            terminals_temp = [
                                pair for pair in flattened_terminals_and_factories.copy() if pair not in {t_p, factory}
                            ]
                            terminals_temp = list(set(terminals_temp))
                            g_temp_temp.remove_nodes_from(terminals_temp)
                            try:
                                path = nx.dijkstra_path(g_temp_temp, t_p, factory)
                            except nx.NetworkXNoPath:
                                continue
                            dist_factories.update({factory: path})
                    # choose shortest available path or if no elements in dist_factories, flag_problem = True
                    if len(dist_factories) == 0:
                        pass
                    else:
                        nearest_factory = min(dist_factories, key=lambda k: len(dist_factories[k]))
                        path = dist_factories[nearest_factory]
                        # self.factory_times[nearest_factory] = self.t  # reset time, here is the wrong place to update this
                        paths_temp_lst.append(path)
                        tp_list.append(t_p)

                if flag_problem:
                    break  # type: ignore[unreachable]

            # add case for flag problem to avoid infinite loop
            # if only t gates in terminal_pairs_current and empty paths_temp_lst, because then we are stuck because of reset time of factories
            all_t = []
            for _ in terminal_pairs_current:
                if isinstance(t_p[0], int) and isinstance(t_p[1], int):
                    all_t.append(True)
                else:
                    all_t.append(False)
            if all(all_t) and len(paths_temp_lst) == 0:
                flag_problem = True

            if len(paths_temp_lst) != 0 and not flag_problem:
                shortest_path = min(paths_temp_lst, key=len)
                shortest_idx = paths_temp_lst.index(shortest_path)  # index in current terminal_pairs_current
                t_p = tp_list[shortest_idx]  # terminal_pairs_current[shortest_idx]
                # update already used qubits based on chosen t_p path
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                    dct_qubits[t_p[0]] = True
                    dct_qubits[t_p[1]] = True
                elif isinstance(t_p[1], int):
                    dct_qubits[t_p] = True
                    # update the times of the factory patch (which is the position at one end of the path)
                    if shortest_path[0] == t_p:
                        self.factory_times[shortest_path[-1]] = self.t
                    elif shortest_path[-1] == t_p:
                        self.factory_times[shortest_path[0]] = self.t
                    else:
                        msg = "Factory not in path."
                        raise RuntimeError(msg)

                # remove nodes from g_temp from path
                for node in shortest_path[1:-1]:
                    g_temp.remove_node(node)
                successful_terminals.append(t_p)
                vdp_dict.update({t_p: shortest_path})

                # remove t_p from terminal_pairs_current
                terminal_pairs_current = [x for x in terminal_pairs_current if x != t_p]

            if len(paths_temp_lst) == 0 or flag_problem:
                terminal_pairs_remainder = [s for s in terminal_pairs_current if s not in successful_terminals]
                dct_qubits = dct_qubits_copy.copy()
                flag_problem = True  # in case the paths_temp_list is empty, you need to set the flag_problem to handle this case as well. otherwise endless loop.

        # check whether the keys in vdp_dict fit the start and end point of the path
        for pair, path in vdp_dict.items():
            start, end = path[0], path[-1]
            if isinstance(pair[1], tuple) and set(pair) != {start, end}:
                msg = f"The path does not coincide with the terminal pair. There is a bug. terminal_pair = {pair} but path = {path}"
                raise RuntimeError(msg)
            if isinstance(pair[1], int) and pair not in {start, end}:
                msg = f"The path does not coincide with the T gate location. There is a bug. terminal_pair = {pair} but path = {path}"
                raise RuntimeError(msg)

        return vdp_dict, terminal_pairs_remainder

    def find_all_vdp_layers(
        self, layer: int
    ) -> list[dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        """Find VDP layers within a given initial layer.

        if find_max_VDP_set returns nonzero terminal_pairs_remainder
        it is required to run the algorithm as long s.t. we find all VDP
        sets even if they are in multiple layers
        Important: Adapt time stampes of the factories.

        Returns:
            list[dict]: list of layers with simultaneous paths (VDP per layer)
        """
        # ! TODO One should actually check whether a remainder might be able to be merged with a new layer after having to split up. sometimes the remainder does not overlap with qubits in the next layer
        flag_continue = True
        vdp_layers = []
        while flag_continue:
            vdp_dict, terminal_pairs_remainder = self.find_max_vdp_set(layer)
            vdp_layers.append(vdp_dict)
            # adapt times
            for key in self.factory_times:
                if self.factory_times[key] != 0:
                    self.factory_times[key] -= 1
            if len(terminal_pairs_remainder) == 0:
                # flag_continue = False
                break
            self.layers_cnot_t[layer] = terminal_pairs_remainder

        return vdp_layers

    def find_total_vdp_layers(
        self,
    ) -> list[dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        """Find all routes for all initial and secondary layers.

        finds total VDP layers, i.e. more than `all` meaning that
        it also respects the initial layer structure of the cnots

        Important: Adapt time stampes of the factories.
        """
        vdp_layers = []
        for layer in range(len(self.layers_cnot_t_orig)):
            self.order_terminal_pairs(layer)
            vdp_layers_temp = self.find_all_vdp_layers(layer)
            vdp_layers += vdp_layers_temp
        return vdp_layers


class ShortestFirstRouterTGatesDyn(ShortestFirstRouterTGates):
    """Shortest First Routing for VDP on Hexagonal Lattice with adaption to greedily include T gates. Dynamically adapts the initial layers."""

    def __init__(
        self,
        m: int,
        n: int,
        terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]],
        factory_positions: list[tuple[int, int]],
        t: int,
    ) -> None:
        """Routing for Hexagonal Lattice with adaption to greedily include T gates.

        Start with graph $G$ and an empty solution.
        While $G$ contains any path connecting any demand pair,
        choose the shortest such path $P$, add $P$ to the solution,
        and delete all vertices of $P$ from $G$

        T gates are included by including the shortest connection between any factory site
        and the respective qubit for the initial ordering. After the ordering, handle T gates
        similar to CNOTs just that we have to check which factories are available + it may be
        necessary to wait 1 or more layers until a T factory becomes available.

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
            terminal_pairs (list[tuple[tuple[int, int], tuple[int, int]]): pairs of vertices to be connected (networkx labeling)
            factory_positions (list[tuple[int,int]]): Positions were factories are placed (should be on the boundary of the data qubits), follows networkx labeling
            t (int): A factory needs t logical time steps of our scheme to generate a new T state
        """
        super().__init__(m, n, terminal_pairs, factory_positions, t)

    def find_total_vdp_layers_dyn(
        self,
    ) -> list[dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        """Find all routes for all initial and secondary layers.

        Important: Adapt time stamps of the factories.
        """
        vdp_layers: list[dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]] = []
        # temp = 0
        layers_cnot_t_prev = None
        counter = 0
        while len(self.layers_cnot_t) > 0:
            layer = 0  # since we adapt the layers_cnot_t_orig inplace, always layer=0 needed
            vdp_dict, terminal_pairs_remainder = self.find_max_vdp_set(
                layer
            )  # layer is successively reordered within find_max_vdp_set

            keys: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = []
            for lst in vdp_layers:
                keys += list(lst.keys())
            # if layers_cnot_t_prev == self.layers_cnot_t_orig and len(terminal_pairs_remainder)==0 and len(keys) == len(self.terminal_pairs):
            if layers_cnot_t_prev == self.layers_cnot_t_orig and len(keys) == len(self.terminal_pairs):
                break
            layers_cnot_t_prev = self.layers_cnot_t_orig.copy()

            for key in self.factory_times:
                if self.factory_times[key] != 0:
                    self.factory_times[key] -= 1

            vdp_layers.append(vdp_dict)
            initial_layers_update = self.push_remainder_into_layers(terminal_pairs_remainder)
            self.layers_cnot_t_orig = initial_layers_update
            self.layers_cnot_t = initial_layers_update
            if len(self.layers_cnot_t) == 0:
                break
            counter += 1

            # avoid infinite loops
            if counter == len(self.terminal_pairs) * 10:
                break

        # it might be possible that there are bugs. hence, check whether vdp layers really contains as main paths as there are gates.
        keys = []
        for lst in vdp_layers:
            keys += lst.keys()
        assert len(keys) == len(self.terminal_pairs), (
            f"The dynamic routing has a bug. There are {len(self.terminal_pairs)} to be routed, but the final vdp_layers only has {len(keys)} paths."
        )
        # also check whether each key can be found in terminal pairs and vice versa
        assert set(keys) == set(self.terminal_pairs), (
            "The dynamic routing has a bug. The finally routed pairs do not coincide with the given terminal_pairs."
        )

        return vdp_layers

    @staticmethod
    def split_current_layer(
        single_initial_layer: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]],
    ) -> list[list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]]]:
        """Similar to split_layer_terminal_pairs. but does not inplace update."""
        layers: list[list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]]] = []
        current_layer: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = []
        used_qubits = set()

        for pair in single_initial_layer:
            if isinstance(pair[0], tuple) and isinstance(pair[1], tuple):
                if pair[0] in used_qubits or pair[1] in used_qubits:
                    layers.append(current_layer)
                    current_layer = [pair]
                    used_qubits = set(pair)
                else:
                    current_layer.append(pair)
                    used_qubits.update(pair)
            elif isinstance(pair[1], int):  # and isinstance(pair[1], int)
                if pair in used_qubits:
                    layers.append(current_layer)
                    current_layer = [pair]
                    used_qubits = {pair}
                else:
                    current_layer.append(pair)
                    used_qubits.update([pair])

        if current_layer:
            layers.append(current_layer)

        return layers

    def push_remainder_into_layers(
        self, remainder: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]]
    ) -> list[list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]]]:
        """Updates a copy of layers_cnot_t (removed used stuff and takes remainder of previous layer, pushes through).

        Args:
            remainder (list[tuple[int,int]]): remaining gates which could not be routed so far in current layer.

        Returns:
            list[list[tuple[int,int]]]: layered gates with remainder being pushed into next layer.
        """
        initial_layers = self.layers_cnot_t_orig.copy()
        if len(initial_layers) > 1:
            del initial_layers[0]  # delete already processed layer (remainder was part of this layer)
        elif len(initial_layers) == 1 and len(remainder) != 0:
            del initial_layers[0]
        i = 0
        flag = True
        while flag is True:
            try:
                initial_layers[i] = (
                    remainder + initial_layers[i]
                )  # push remainder in front of the new zeroth entry (previously entry 1)
            except IndexError:  # if no further initial_layer[i] available but still the previous layer was split
                initial_layers.append(remainder)
            layers = self.split_current_layer(initial_layers[i])
            if len(layers) == 1:
                # adding remainder to initial_layers[0] caused no conflict, so we are finished
                # flag = False
                break
            if len(layers) == 2:  # push further through
                initial_layers[i] = layers[0]
                remainder = layers[1]
                i += 1
            else:
                msg = (
                    f"Something weird happened during pushing remainders. len(layers)={len(layers)}, layers = {layers}"
                )
                raise RuntimeError(msg)

        return initial_layers


def plot_lattice_paths(
    g: nx.Graph,
    vdp_dict: dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]],
    layout: dict[tuple[int, int], int] | None = None,
    factory_locs: list[tuple[int, int]] | None = None,
    size: tuple[float, float] = (3.5, 3.5),
) -> None:
    """Plots the graph and the corresponding VDP of a layer.

    Args:
        g (nx.Graph): Graph on which we route
        vdp_dict (dict): Output of router.find_total_vdp_layers
        layer (int): label of layer to plot
        layout (dict): potentially also display the qubit labels. keys = qubit label, value = node label
        factory_locs (list[tuple[int,int]] | None): factory locations.
        size (tuple[float,float], optional): _description_. Size of the plot. Defaults to (3.5,3.5).
    """
    if layout is None:
        layout = {}
    if factory_locs is None:
        factory_locs = []
    pos = nx.get_node_attributes(g, "pos")

    num_paths = len(vdp_dict.keys())
    colormap = plt.cm.get_cmap("rainbow", num_paths)
    colors = [mcolors.to_hex(colormap(i)) for i in range(num_paths)]

    plt.figure(figsize=size)
    nx.draw(g, pos, with_labels=True, node_color="gray", edge_color="lightblue", font_size=8)

    for i, path in enumerate(vdp_dict.values()):
        if path:
            path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            nx.draw_networkx_edges(g, pos, edgelist=path_edges, width=2, edge_color=colors[i])
            nx.draw_networkx_nodes(g, pos, nodelist=path, node_color=colors[i], label=f"Path {i + 1}")

    if len(factory_locs) != 0:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=factory_locs,
            node_color="violet",
        )

    if len(list(layout.keys())) != 0:
        for key, value in layout.items():
            node_pos = pos[value]
            plt.text(node_pos[0], node_pos[1] - 0.1, str(key), fontsize=8, color="white", horizontalalignment="center")
            # also highlight data qubits
            nx.draw_networkx_nodes(
                g,
                pos,
                nodelist=layout.values(),  # Nodes to highlight
                node_color="none",  # Unfilled circles
                edgecolors="lime",  # Neon green outline
                linewidths=1.5,  # Line width for the outline
            )

    plt.legend()
    plt.show()
