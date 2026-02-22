# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Routing Routines for Lattice Surgery Compilation."""

from __future__ import annotations

import collections
import datetime
import itertools
import logging
import pathlib
import pickle  # noqa: S403
import random
import sys
import warnings
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np

import mqt.qecc.cococo.internal_testing as tst
from mqt.qecc.cococo import dag_helper

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


pos = tuple[int, int]
steiner_type = dict[tuple[pos, pos, pos] | tuple[pos, pos], list[list[pos]]]
lock_penalty = 200


class BasicRouter:
    """Basic Routing for CNOT + T gates based on shortest-first VDP solving."""

    def __init__(
        self,
        g: nx.Graph,
        logical_pos: list[pos],
        factory_pos: list[pos],
        valid_path: str,
        t: int,
        metric: str,
        use_dag: bool,
    ) -> None:
        """Class for shortest-first routing based compilation.

        Args:
            g (nx.Graph): Macroscopic Routing Graph. Created via mqt.cococo.layouts
            logical_pos (list[pos]): Logical positions on the graph. Also from mqt.cococo.layouts
            factory_pos (list[pos]): Positions of the factories. Also from mqt.cococo.layouts
            valid_path (str): Either "cc" or "sc" for color code and surface code. However, revisit usefulness of "sc".
            t (int): Reset time for the factories
            metric (str): Either "exact" or "crossing", but it is recommended to use "exact". "crossing" works only for pure cnot circuits.
            use_dag (bool, optional): Determines whether DAG structure from qiskit is used or naive sequential layering. It is recommended to use `True`.
        """
        self.g = g
        self.logical_pos = logical_pos
        self.factory_pos = factory_pos
        if valid_path not in {"cc", "sc"}:
            msg = "Other valid path setups are not implemented yet."  # pragma: no cover
            raise NotImplementedError(msg)
        self.valid_path = valid_path
        self.use_dag = use_dag
        self.t = t

        self.factory_times = {}
        for factory in factory_pos:
            self.factory_times.update({factory: t})

        self.logical_pos_temp = None

        self.metric = metric
        if metric not in {"crossing", "exact"}:
            msg = "Other metrics than crossing and exact not implemented yet."  # pragma: no cover
            raise NotImplementedError(msg)

    @staticmethod
    def path_sc(g: nx.Graph, control: pos, target: pos) -> list[pos]:
        """Find the shortest path with Dijkstra but with constraints for standard sc.

        This means, that control qubits are only entered horizontally, targets vertically.
        thus horizontal bdrys are Z_L and vertical X_L.

        This method is actually useless because the Lookahead optimization in this class is only valid for color code connectivity.
        """
        g_temp = g.copy()
        # for control, remove vertical edges
        vertical_neighbors = [
            (control[0], control[1] + 1),
            (control[0], control[1] - 1),
        ]
        for neigh in vertical_neighbors:
            if (neigh, control) in g_temp.edges() or (control, neigh) in g_temp.edges():
                g_temp.remove_edge(neigh, control)

        # for target, remove horizontal edges
        horizontal_neighbors = [(target[0] + 1, target[1]), (target[0] - 1, target[1])]
        for neigh in horizontal_neighbors:
            if (neigh, target) in g_temp.edges() or (target, neigh) in g_temp.edges():
                g_temp.remove_edge(neigh, target)
        # run dijkstra on the adapted graph
        return cast("list[pos]", nx.dijkstra_path(g_temp, control, target))

    @staticmethod
    def path_cc(g: nx.Graph, control: pos, target: pos) -> list[pos]:
        """Find the shortest path with Dijkstra for a color code architecture.

        There is only constraint: directly neighboring control/target cannot be directly connected.
        Hence, length of path must at least be 3.
        """
        g_temp = g.copy()
        if (control, target) in g_temp.edges():
            g_temp.remove_edge(control, target)
        return cast("list[pos]", nx.dijkstra_path(g_temp, control, target))

    def valid_path_method(self) -> Callable[[nx.Graph, pos, pos], list[pos]]:
        """Calls the correct version of Dijkstra depending on `self.valid_path`."""
        if self.valid_path == "cc":
            return self.path_cc
        if self.valid_path == "sc":
            for n in self.g.nodes():
                if "pos" not in self.g.nodes[n]:
                    msg = "Node does not have pos attribute."
                    raise RuntimeError(msg)  # pragma: no cover
                if self.g.nodes[n]["pos"] != n:
                    msg = """
                            Node pos attribute does not match node label.
                            Make sure you construct the pos of your initial graph like this:

                            g = nx.grid_2d_graph(m, n)
                            pos = {node: node for node in g.nodes()}
                            for node in g.nodes():
                                g.nodes[node]["pos"] = node
                        """
                    raise RuntimeError(msg)  # pragma: no cover
            return self.path_sc

        msg = "Other valid paths not implemented yet."  # pragma: no cover
        raise NotImplementedError(msg)

    @staticmethod
    def split_layer_terminal_pairs(terminal_pairs: list[pos | tuple[pos, pos]]) -> list[list[pos | tuple[pos, pos]]]:
        """Split terminal_pairs into layers of disjoint qubit support.

        Only really needed if self.use_dag=False. If true, it is often computed uselessly.
        """
        layers = []
        current_layer: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = []
        used_qubits = set()

        for pair in terminal_pairs:
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

    def find_max_vdp_set(
        self,
        layer: list[tuple[pos, pos] | pos],
        logical_pos: list[pos] | None,
        factory_times: dict[pos, int],
    ) -> tuple[dict[pos | tuple[pos, pos], list[pos]], list[pos | tuple[pos, pos]], dict[pos, int]]:
        """Finds the approximation for a vdp set for some given layer and returns the routing as well as a potential remainder.

        If logical_pos is given as input, you use the input instead of the self.logical_pos
        """
        factory_times_temp = factory_times.copy()
        vdp_dict: dict[
            pos | tuple[pos, pos],
            list[pos],
        ] = {}
        terminal_pairs_remainder = []
        successful_terminals = []  # gather successful terminal pairs
        flag_problem = False
        g_temp = self.g.copy()
        dct_qubits = {}  # a dct which checks whether a qubit was already used in the layer
        terminal_pairs_current = layer.copy()
        flattened_terminals = [
            pair for item in layer.copy() for pair in (item if isinstance(item[0], tuple) else [item])
        ]
        for t in flattened_terminals:
            dct_qubits.update({t: False})
        dct_qubits_copy = dct_qubits.copy()
        flattened_terminals_and_factories = (
            flattened_terminals.copy() + self.factory_pos.copy()
        )  # was using self.flattened before, which was problem because not updated after logical repositioning

        while len(terminal_pairs_current) > 0 and flag_problem is False:
            paths_temp_lst = []  # gather all possible paths here, between all terminal pairs (cnots) and between all qubits for a tgate with all factories
            tp_list: list[
                tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
            ] = []  # same order, actually redundant but error otherwise
            for t_p in terminal_pairs_current:
                # cnot
                if isinstance(t_p[0], tuple) and isinstance(t_p[1], tuple):
                    g_temp_temp = g_temp.copy()
                    # ADAPT THE GRAPH AND REMOVE ALL LOGICAL DATA PATCHES DESPITE THE t_p
                    if logical_pos is None:
                        nodes_to_remove = [x for x in self.logical_pos if x != t_p[0] and x != t_p[1]]
                    else:
                        nodes_to_remove = [x for x in logical_pos if x != t_p[0] and x != t_p[1]]
                    g_temp_temp.remove_nodes_from(nodes_to_remove)

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
                        path = self.valid_path_method()(g_temp_temp, t_p[0], t_p[1])
                        paths_temp_lst.append(path)
                        tp_list.append(t_p)
                    except nx.NetworkXNoPath:
                        # skip the t_p if no path exists
                        pass  # therefore just pass

                # t gate
                elif isinstance(t_p[1], int):
                    g_temp_temp = g_temp.copy()
                    # ADAPT THE GRAPH AND REMOVE ALL LOGICAL DATA PATCHES DESPITE THE t_p position where the t gate is applied
                    if logical_pos is None:
                        nodes_to_remove = [x for x in self.logical_pos if x != t_p]
                    else:
                        nodes_to_remove = [x for x in logical_pos if x != t_p]
                    g_temp_temp.remove_nodes_from(nodes_to_remove)

                    if dct_qubits[t_p]:
                        flag_problem = True
                        break
                    dist_factories = {}
                    for factory in self.factory_pos:
                        g_temp_temp2 = g_temp_temp.copy()
                        if factory_times_temp[factory] == 0:  # only include available factories
                            # remove other terminals
                            terminals_temp = [
                                pair for pair in flattened_terminals_and_factories.copy() if pair not in {t_p, factory}
                            ]
                            terminals_temp = list(set(terminals_temp))
                            g_temp_temp2.remove_nodes_from(terminals_temp)
                            try:
                                path = self.valid_path_method()(g_temp_temp2, t_p, factory)
                            except nx.NetworkXNoPath:
                                continue
                            dist_factories.update({factory: path})
                    # choose shortest available path or if no elements in dist_factories, flag_problem = True
                    if len(dist_factories) == 0:
                        pass
                    else:
                        nearest_factory = min(dist_factories, key=lambda k: len(dist_factories[k]))
                        path = dist_factories[nearest_factory]
                        paths_temp_lst.append(path)
                        tp_list.append(t_p)

                if flag_problem:
                    break  # type: ignore[unreachable]

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
                        factory_times_temp[shortest_path[-1]] = self.t + 1
                    elif shortest_path[-1] == t_p:
                        factory_times_temp[shortest_path[0]] = self.t + 1
                    else:
                        msg = "Factory not in path."  # pragma: no cover
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
                msg = f"The path does not coincide with the terminal pair. There is a bug. terminal_pair = {pair} but path = {path}"  # pragma: no cover
                raise RuntimeError(msg)
            if isinstance(pair[1], int) and pair not in {start, end}:
                msg = f"The path does not coincide with the T gate location. There is a bug. terminal_pair = {pair} but path = {path}"  # pragma: no cover
                raise RuntimeError(msg)

        return vdp_dict, terminal_pairs_remainder, factory_times_temp

    def push_remainder_into_layers(
        self,
        layers: list[list[tuple[pos, pos] | pos]],
        remainder: list[pos | tuple[pos, pos]],
        delete_layer_zero: bool = True,
    ) -> list[list[pos | tuple[pos, pos]]]:
        """Updates a copy of layers_cnot_t (removed used stuff and takes remainder of previous layer, pushes through).

        Only really needed if self.use_dag=False. If true, it is often computed uselessly.

        Args:
            layers (list[list[tuple[pos,pos]|pos]]): given layers
            remainder (list[tuple[int,int]]): remaining gates which could not be routed so far in current layer.
            delete_layer_zero (bool): whether 0th layer is deleted or not.

        Returns:
            list[list[pos | tuple[pos, pos]]]: layered gates with remainder being pushed into next layer.
        """
        initial_layers = layers.copy()
        if delete_layer_zero:
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
            layers = self.split_layer_terminal_pairs(initial_layers[i])
            if len(layers) == 1:
                # adding remainder to initial_layers[0] caused no conflict, so we are finished
                break
            if len(layers) >= 2:  # push further through
                initial_layers[i] = layers[0]
                remainder = [item for sublayer in layers[1:] for item in sublayer]
                i += 1
            else:
                msg = f"Something weird happened during pushing remainders. len(layers)={len(layers)}, layers = {layers}"  # pragma: no cover
                raise RuntimeError(msg)

        return initial_layers

    def find_total_vdp_layers_dyn(
        self,
        next_layers: list[list[pos | tuple[pos, pos]]],
        logical_pos: list[pos],
        factory_times: dict[pos, int],
        layout: dict[int, pos],
        testing: bool = False,
    ) -> tuple[
        list[
            dict[
                pos | tuple[pos, pos],
                list[pos],
            ]
        ]
        | None,
        dict[pos, int],
    ]:
        """Finds all routes for given logical layers called `next_layers`.

        This is only used if self.use_dag = False

        Dynamically pushes remaining gates into the next layers.
        """
        if self.use_dag and layout is None:
            msg = "If self.use_dag=True you need to enter a layout in find_total_vdp_layers_dyn"  # type: ignore[unreachable] # pragma: no cover
            raise ValueError(msg)

        factory_times_temp = factory_times.copy()
        terminal_pairs = []
        for layer in next_layers:
            terminal_pairs += layer
        layers_cnot_t_orig = next_layers.copy()
        vdp_layers: list[
            dict[
                tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
                list[tuple[int, int]],
            ]
        ] = []

        layers_cnot_t_prev = None

        stuck_counter = 0

        terminal_pairs_temp = []
        for layer_temp in layers_cnot_t_orig:
            terminal_pairs_temp += layer_temp
        dag = dag_helper.terminal_pairs_into_dag(terminal_pairs_temp, layout)

        while len(layers_cnot_t_orig) > 0:
            layer_idx: int = 0  # since we adapt the layers_cnot_t_orig inplace, always layer=0 needed

            if self.use_dag:
                current_layer = dag_helper.extract_layer_from_dag(dag, layout, layer_idx)  # layer=0
            else:
                current_layer = layers_cnot_t_orig[layer_idx]

            vdp_dict, terminal_pairs_remainder, factory_times_temp = self.find_max_vdp_set(
                current_layer,
                logical_pos=logical_pos,
                factory_times=factory_times_temp,  # !IMPORTANT: logical_pos is not taken from self.logical_pos here, because it would be overwritten too often in the annealing procedure. but we need to compute the metric correctly with this _dyn method
            )  # layer is successively reordered within find_max_vdp_set

            keys: list[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]] = []
            for lst in vdp_layers:
                keys += list(lst.keys())
            if layers_cnot_t_prev == layers_cnot_t_orig and len(keys) == len(terminal_pairs):
                break
            layers_cnot_t_prev = layers_cnot_t_orig.copy()

            if len(vdp_dict) == 0:
                stuck_counter += 1
            if stuck_counter > 10 * self.t:
                return None, factory_times_temp  # need a stuck counter!!!

            for key in factory_times_temp:
                if factory_times_temp[key] != 0:
                    factory_times_temp[key] -= 1

            vdp_layers.append(vdp_dict)
            if self.use_dag:
                initial_layers_update, dag = dag_helper.push_remainder_into_layers_dag(
                    dag, terminal_pairs_remainder, layout, current_layer
                )
            else:
                initial_layers_update = self.push_remainder_into_layers(layers_cnot_t_orig, terminal_pairs_remainder)
            layers_cnot_t_orig = initial_layers_update
            if len(layers_cnot_t_orig) == 0:
                break

        # it might be possible that there are bugs. hence, check whether vdp layers really contains as main paths as there are gates.
        keys = []
        for lst in vdp_layers:
            keys += lst.keys()
        assert len(keys) == len(terminal_pairs), (
            f"The dynamic routing has a bug. There are {len(terminal_pairs)} to be routed, but the final vdp_layers only has {len(keys)} paths."
        )

        if testing:
            if tst.check_order_dyn_gates_st(terminal_pairs, vdp_layers, layout=layout):
                logger.info("stim test succeeded for standard routing (:")
            else:
                logger.info("stim test failed - THERE IS A PROBLEM!")
            if tst.check_duplicate_nodes_per_layer_st(vdp_layers):
                logger.info("no duplicates found in standard routing (:")

            if logical_pos is None:
                if tst.check_path_on_logical_st(vdp_layers, self.logical_pos):  # type: ignore[unreachable]
                    logger.info("paths do not occupy logical pos (:")
            elif tst.check_path_on_logical_st(vdp_layers, logical_pos):
                logger.info("paths do not occupy logical pos (:")
            if tst.test_times_t_gates_st(vdp_layers, self.t, self.factory_pos):
                logger.info("Reset times make sense, all good(:")

        return vdp_layers, factory_times_temp

    def count_crossings(self, layers: list[list[tuple[pos, pos]]], logical_pos_temp: list[pos]) -> int:
        """Heuristic energy function to minimize.

        Computes number of crossings of Dijkstra paths without constraints.
        If there is a shortest path which is not possible at all (due to locking), then add high, hard coded penalty.
        This only works for CNOT circuits only.
        """
        total_penalty = 0
        lst_crossings = []
        for layer in layers:
            paths = []
            for t_p in layer:
                g_temp = self.g.copy()  # this is duplicate code from `order_terminal_pairs`
                terminal_pairs_flattened = [pair for sublist in layer for pair in sublist]
                terminals_temp = [pair for pair in terminal_pairs_flattened if pair != t_p[0] and pair != t_p[1]]
                terminals_temp = list(set(terminals_temp))
                g_temp.remove_nodes_from(terminals_temp)
                nodes_to_remove = [
                    x for x in logical_pos_temp if x != t_p[0] and x != t_p[1]
                ]  # these are also necessary since logical vertices which are not used in the layer are important nevertheless.
                g_temp.remove_nodes_from(nodes_to_remove)
                try:
                    path = self.valid_path_method()(g_temp, t_p[0], t_p[1])
                    paths.append(path)
                except nx.NetworkXNoPath:
                    total_penalty += lock_penalty
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

        return sum(lst_crossings) + total_penalty

    def count_crossings_per_layer(
        self, layers: list[list[pos | tuple[pos, pos]]], t_crossings: bool = False
    ) -> list[int]:
        """Counts the crossings of the simple paths between cnots and between shortest factory to qubit path (respecting terminals and factory positions) per layer.

        Args:
            layers (list[list[pos | tuple[pos,pos]]]): circuit layers.
            t_crossings (bool): decides whether the crossings to the factory are included (true) or not (false).

        Returns:
            list[int]: Number of crossings per initial layer. len is len(self.layers_cnot_t_orig)
        """
        # ! TODO: remove redundancies (order_terminal_pairs very similar)
        lst_crossings = []
        flattened_terminals_and_factories = self.factory_pos.copy()
        for layer in layers:
            temp = [pair for item in layer.copy() for pair in (item if isinstance(item[0], tuple) else [item])]
            flattened_terminals_and_factories += temp

        for layer in layers:
            paths = []
            for t_p in layer:
                g_temp = self.g.copy()
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
                        msg = (  # pragma: no cover
                            "Your choice of terminal pairs locks in at least one terminal. "
                            "Reconsider your choice of terminal pairs."
                        )
                        raise ValueError(msg) from exc

                elif t_crossings:
                    dist_factories = {}  # gather distances to each factory to greedily choose the shortest path
                    for factory in self.factory_pos:
                        g_temp = self.g.copy()
                        terminals_temp = [
                            pair for pair in flattened_terminals_and_factories.copy() if pair not in {t_p, factory}
                        ]
                        terminals_temp = list(set(terminals_temp))
                        g_temp.remove_nodes_from(terminals_temp)
                        try:
                            path = nx.dijkstra_path(g_temp, t_p, factory)
                        except nx.NetworkXNoPath as exc:
                            msg = (  # pragma: no cover
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


class TeleportationRouter(BasicRouter):
    """Compilation routine that exploits CNOT + teleportation steps."""

    def __init__(
        self,
        g: nx.Graph,
        logical_pos: list[pos],
        factory_pos: list[pos],
        valid_path: str,
        t: int,
        metric: str,
        use_dag: bool,
        seed: int,
    ) -> None:
        """Compilation routine that exploits CNOT + teleportation steps based on BasicRouter.

        Args:
            g (nx.Graph): Macroscopic Routing Graph. Created via mqt.cococo.layouts
            logical_pos (list[pos]): Logical positions on the graph. Also from mqt.cococo.layouts
            factory_pos (list[pos]): Positions of the factories. Also from mqt.cococo.layouts
            valid_path (str): Either "cc" or "sc" for color code and surface code. However, revisit usefulness of "sc".
            t (int): Reset time for the factories
            metric (str): Either "exact" or "crossing", but it is recommended to use "exact". Using "crossing" is not safe (as it is not tested and not fully implemented.)
            use_dag (bool, optional): Determines whether DAG structure from qiskit is used or naive sequential layering. It is recommended to use `True`.
            seed (int): Seed for the randomized parts in the optimization.
        """
        super().__init__(g, logical_pos, factory_pos, valid_path, t, metric, use_dag)
        self.seed = seed
        random.seed(seed)

    def initialize_steiner(
        self,
        vdp_dict: dict[str | pos | tuple[pos, pos], list[pos]],
        steiner_init_type: str,
        layers: list[list[pos | tuple[pos, pos]]] | None = None,
        k_lookahead: int | None = None,
    ) -> steiner_type:
        """Initialize a random steiner tree per path which are non-overlapping.

        This initialization depends on the vdp_dict you already have for your current layer. This is fixed, it only tries to find additional terminal of a 3-terminal steiner tree.

        If layers and k_lookahead are not None, only a limited number of trees is initialized; namely only for the qubits which are actually used in layers[:k_lookahead]. Other qubits are not moved.
        However, this turned out not to be really useful, because movements can be relevant even if this constraint does not hold.
        Therefore, layers and k_lookahead default to None.
        """
        # find out which qubits are actually moved in layers[:k_lookahead]
        if layers is not None and k_lookahead is not None:
            layers_temp = layers[:k_lookahead]
            terminals = []
            for layer in layers_temp:
                terminals += layer
            qubits_k_lookahead = [t for outer in terminals for t in outer]
            # remove those from vdp_cit which are not used, such that no tree is created for them
            vdp_dict_reduced = {}
            for key, val in vdp_dict.items():
                if key[0] in qubits_k_lookahead or key[1] in qubits_k_lookahead:
                    vdp_dict_reduced.update({key: val})

            vdp_dict = vdp_dict_reduced

        # remove the nodes from the graph which are already occupied by the magic/logical patches
        # we allow the 3-terminal to be placed on the path, thus the graph must be adapted per terminal choice
        # because for path a, you can place the terminal along path a, but not along path b
        g_temp = self.g.copy()
        g_temp.remove_nodes_from(self.factory_pos)
        g_temp.remove_nodes_from(self.logical_pos)

        if steiner_init_type not in {"full_random", "on_path_random"}:
            msg = "Make sure that `steiner_init_type` is either full_random or on_path_random. Other possibilities not implemented yet."  # pragma: no cover
            raise NotImplementedError(msg)

        steiner_dct = {}

        # for each already present path, choose one random ancilla terminal which is allowed to be on the same path, but not on another path
        for key, path in vdp_dict.items():
            if isinstance(key, str) and key.startswith("idle"):
                continue  # we do not want to create a tree at an idling path!
            other_paths = [
                pos for keyy, path in vdp_dict.items() if keyy != key for pos in path
            ]  # collect all terminals occupied by other paths which is not the present one, this also captures potential idling paths.
            g_temp_temp = g_temp.copy()
            g_temp_temp.remove_nodes_from(other_paths)
            # choose some node on the path randomly
            flag = False
            pathcopy = path.copy()
            path = path[  # noqa: PLW2901
                1:-1
            ]  # remove last and first node from the list because those are logical data patches
            random.shuffle(path)
            if steiner_init_type == "full_random":
                for node_on_path in path:  # loop in case a random node has  no other reachable nodes
                    # determine all reachable nodes from that chosen node
                    reachable_nodes = list(nx.single_source_shortest_path_length(g_temp_temp, node_on_path).keys())
                    if reachable_nodes:
                        flag = True
                        break  # if there is a reachable node found, take it, otherwise try another random node
                if not flag:
                    continue  # skip this path if no reachable node found
                # select a random reachable node
                terminal_node = random.choice(reachable_nodes)  # noqa: S311
                # determine the path between the node which is ensured on the path and the terminal
                path_steiner = nx.dijkstra_path(g_temp_temp, node_on_path, terminal_node)
                paths_lst_temp = []  # collect all paths from path1[1:-1] to new_terminal
                for node_on_path in path:
                    try:
                        path_temp = nx.dijkstra_path(g_temp_temp, node_on_path, terminal_node)
                        paths_lst_temp.append(path_temp)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):  # noqa: PERF203
                        pass
                if paths_lst_temp:
                    path_steiner = min(paths_lst_temp, key=len)
            elif steiner_init_type == "on_path_random":
                terminal_node = random.choice(path)  # just choose a random terminal ON the path  # noqa: S311
                path_steiner = [
                    terminal_node
                ]  # terminal on the path does not need an extended path, but list should not be empty, otherwise error.
            else:
                msg = "steiner_init_type must be on_path_random or full_random."  # pragma: no cover
                raise ValueError(msg)
            # add to list
            if isinstance(key[0], tuple):  # if CNOT
                tup = (*key, terminal_node)
            elif isinstance(key[0], int):  # if T gate
                tup = (key, terminal_node)
            steiner_dct.update({tup: [pathcopy, path_steiner]})
            # also remove the nodes from the graph such that no overlapping steiner trees can be generated
            g_temp.remove_nodes_from(path_steiner)
        return steiner_dct

    def perturbation(
        self, steiner_dct: steiner_type, radius: int, vdp_dict: dict[str | pos | tuple[pos, pos], list[pos]]
    ) -> tuple[steiner_type, nx.Graph]:
        """Computes a perturbation of a given collection of trees within a given radius of edges around the current terminal.

        For each tree in steiner_dct a new location of the 3rd terminal is updated randomly.
        """
        if self.logical_pos_temp is None:
            msg = "Need to initialize logical pos temp properly in a summarizing method."  # pragma: no cover
            raise RuntimeError(msg)
        g_temp = self.g.copy()  # type: ignore[unreachable]
        g_temp.remove_nodes_from(self.factory_pos)
        g_temp.remove_nodes_from(
            self.logical_pos_temp
        )  # logical pos temp must be successively updated in a loop where you apply multiple perturbations

        steiner_dct_update = steiner_dct.copy()
        # used_nodes = set()
        new_terminal = None

        for key_tree, (path1, path2) in steiner_dct.items():
            # determine whether tree corresponds to a CNOT or T gate
            if len(key_tree) == 3:
                (a, b, terminal) = key_tree
                other_paths = [
                    pos
                    for keyy, path in steiner_dct_update.items()
                    if keyy != (a, b, terminal)
                    for pos in path[0][1:-1]
                ]
                other_paths += [
                    pos for keyy, path in steiner_dct_update.items() if keyy != (a, b, terminal) for pos in path[1]
                ]  # not 1:-1 because 3rd terminal is not allowed to be part of other terminal path
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                other_paths = [
                    pos for keyy, path in steiner_dct_update.items() if keyy != (a, terminal) for pos in path[0][1:-1]
                ]
                other_paths += [
                    pos for keyy, path in steiner_dct_update.items() if keyy != (a, terminal) for pos in path[1]
                ]
            else:
                msg = "Something is wrong with the allocation of keys in the steiner_dict"  # pragma: no cover
                raise RuntimeError(msg)

            # a terminal can be placed on the path. in this case you are NOT allowed to remove it! the above somehow sometimes add terminal, hence remove it again
            if terminal in other_paths:
                other_paths.remove(terminal)
            if path2[0] in other_paths:
                other_paths.remove(path2[0])

            g_temp_temp = g_temp.copy()
            g_temp_temp.remove_nodes_from(other_paths)

            # remove nodes from vdp dict (the tree is not allowed to be on or cross another path)
            for path_label, path in vdp_dict.items():
                if isinstance(path_label, str):
                    nodes_to_delete = path[1:]  # for idle move you need to delete more
                elif isinstance(path_label[0], tuple):  # cnot
                    nodes_to_delete = path[1:-1]
                elif isinstance(path_label[0], int):  # t
                    nodes_to_delete = path[1:]
                for node in nodes_to_delete:
                    if node in g_temp_temp.nodes() and node not in path1 + path2:  # {terminal, path2[0]}
                        g_temp_temp.remove_node(node)
            # find "neighborhood" of the terminal
            neighborhood = set(nx.single_source_shortest_path_length(g_temp_temp, terminal, cutoff=radius).keys())
            # the single source shortest path,... ensures that only reachable nodes are included
            # choose one of them
            if len(neighborhood) == 1:  # if only one neighbor, i.e. the terminal itself
                # new_terminal = None #to skip the updating of the root node below.
                break
            while True:
                new_terminal = random.choice(list(neighborhood))  # noqa: S311
                if new_terminal == terminal:  # do not want same terminal again
                    continue
                try:
                    path_terminal = nx.dijkstra_path(
                        g_temp_temp, path2[0], new_terminal
                    )  # path2[0] is the connecting node on the path
                except nx.NetworkXNoPath:
                    warnings.warn("If this is called you need to check why this is happening.")  # noqa: B028
                if path_terminal:
                    break

            # !TODO should i skip this since we do it globally afterwards again?
            # (A) loop to possibly find shorter path_terminal
            paths_lst_temp = []  # collect all paths from path1[1:-1] to new_terminal
            for node_on_path in path1[1:-1]:
                try:
                    path_temp = nx.dijkstra_path(g_temp_temp, node_on_path, new_terminal)
                    paths_lst_temp.append(path_temp)
                except nx.NetworkXNoPath:  # noqa: PERF203
                    pass
            if paths_lst_temp:
                path_terminal = min(paths_lst_temp, key=len)

            # delete old entry and add new with updated key
            steiner_dct_update.pop(key_tree, None)
            if len(key_tree) == 3:
                (a, b, terminal) = key_tree
                new_key_tree = (a, b, new_terminal)
            elif len(key_tree) == 2:
                (a, terminal) = key_tree
                new_key_tree = (a, new_terminal)
            steiner_dct_update[new_key_tree] = (path1, path_terminal)
            # remove_branch_nodes += path_terminal

        # it is possible that (A) does not capture everything, as the terminal path may change in a later iteration and thus make even shorter paths possible.
        if (
            new_terminal is not None
        ):  # if the neighborhood has 1 item only, the above breaks. then we do not want to do this reduction.
            steiner_dct_update_second = steiner_dct_update.copy()
            for key_tree, (path1, path2) in steiner_dct_update.items():
                if len(key_tree) == 3:
                    (a, b, terminal) = key_tree
                elif len(key_tree) == 2:
                    (a, terminal) = key_tree
                else:
                    msg = "steiner dct keys are wrong."  # pragma: no cover
                    raise ValueError(msg)
                other_paths = [
                    pos for keyy, path in steiner_dct_update_second.items() if keyy != key_tree for pos in path[0][1:-1]
                ]
                other_paths += [
                    pos for keyy, path in steiner_dct_update_second.items() if keyy != key_tree for pos in path[1]
                ]
                if terminal in other_paths:
                    other_paths.remove(terminal)
                if path2[0] in other_paths:
                    other_paths.remove(path2[0])
                if terminal in other_paths:
                    other_paths.remove(terminal)
                g_temp_temp = g_temp.copy()
                g_temp_temp.remove_nodes_from(other_paths)
                for path_label, path in vdp_dict.items():
                    if isinstance(path_label, str):  # noqa: SIM108
                        nodes_to_delete = path[1:]  # for idle move you need to delete more
                    else:
                        nodes_to_delete = path[1:-1]
                    for node in nodes_to_delete:
                        if node in g_temp_temp.nodes() and node not in path1 + path2:  # {terminal, path2[0]}:
                            g_temp_temp.remove_node(node)
                paths_lst_temp = []  # collect all paths from path1[1:-1] to new_terminal
                for node_on_path in path1[1:-1]:
                    try:
                        path_temp = nx.dijkstra_path(g_temp_temp, node_on_path, terminal)
                        paths_lst_temp.append(path_temp)
                    except nx.NetworkXNoPath:  # noqa: PERF203
                        pass
                if paths_lst_temp:
                    path_terminal = min(paths_lst_temp, key=len)
                    if len(key_tree) == 3:
                        (a, b, terminal) = key_tree
                        new_key_tree = (a, b, terminal)
                    elif len(key_tree) == 2:
                        (a, terminal) = key_tree
                        new_key_tree = (a, terminal)
                    steiner_dct_update_second.pop(key_tree, None)
                    steiner_dct_update_second[new_key_tree] = (path1, path_terminal)
        if new_terminal is None:
            steiner_dct_update_second = steiner_dct_update
            g_temp_temp = g_temp.copy()
        return steiner_dct_update_second, g_temp_temp

    @staticmethod
    def replace_pos(lst: list[tuple[pos, pos] | pos], old: pos, new: pos) -> list[tuple[pos, pos] | pos]:
        """Helper function to replace pos values in lists.

        This is needed to update logical_pos etc. during the optimization.
        """
        result: list[tuple[pos, pos] | pos] = []
        for item in lst:
            if isinstance(item[0], int):
                result.append(new if item == old else item)
            else:
                result.append(cast("tuple[pos,pos]", tuple(new if sub == old else sub for sub in item)))
        return result

    def run_annealing(
        self,
        next_layers: list[list[pos | tuple[pos, pos]]],
        init_steiner_dct: steiner_type,
        max_iters: int,
        T_start: float,  # noqa: N803
        T_end: float,  # noqa: N803
        alpha: float,
        k_lookahead: int,
        radius: int,
        vdp_dict: dict[str | pos | tuple[pos, pos], list[pos]],
        layout: dict[int, pos],
    ) -> tuple[
        steiner_type | None,  # best_steiner
        int,  # best_cost
        list[
            dict[
                pos | tuple[pos, pos],
                list[pos],
            ]
        ]
        | None,  # best_schedule
        list[tuple[int, list[list[tuple[pos, pos] | pos]]]],  # cost_history
        dict[tuple[pos, pos] | tuple[pos, pos, pos], str],  # best_move_type_lst
        list[steiner_type],  # steiner_history
        list[nx.Graph],  # graph_history
    ]:
        """Plug together all previous methods to run annealing for k future layers.

        Args:
            next_layers (list[list[pos | tuple[pos,pos]]]): upcoming layers to draw into consideration
            init_steiner_dct (dict[tuple[pos, pos, pos] | tuple[pos, pos], list[list[pos]]]): current steiner
            max_iters (int): max number of iterations of each simulated annealing
            T_start (float): start temperature of simulated annealing
            T_end (float): end temperature of simulated annealing
            alpha (float): factor to reduce temp
            k_lookahead (int): number of logical layers to draw into account
            radius (int): number of edges within which one can choose a new ancilla position
            vdp_dict (dict[ str | pos | tuple[pos, pos], list[pos]]): routing of current layer
            layout (dict[int, pos]): layout.

        Returns:
            best_steiner
            best_cost
            best_schedule
            cost_history
            best_move_type_lst
            steiner_history
            graph_history

        """
        factory_times_copy = self.factory_times.copy()
        if T_start < T_end:
            msg = "T_start must be larger than T_end"  # pragma: no cover
            raise ValueError(msg)
        if alpha >= 1.0 or alpha <= 0:
            msg = "alpha must be between 0 and 1"  # pragma: no cover
            raise ValueError(msg)

        self.logical_pos_temp = self.logical_pos.copy()  # type: ignore[assignment]

        steiner_dct = init_steiner_dct.copy()
        if self.metric == "crossing":
            cost = self.count_crossings(
                cast("list[list[tuple[pos,pos]]]", next_layers[:k_lookahead]),
                self.logical_pos_temp,  # type: ignore[arg-type]
            )  # overwrite this in the upcoming loop
        elif self.metric == "exact":
            schedule, _ = self.find_total_vdp_layers_dyn(
                next_layers[:k_lookahead],
                self.logical_pos_temp,  # type: ignore[arg-type]
                factory_times_copy,
                layout,
            )  # initially the self.logical pos can be used. later you need a logical_pos outside of self
            cost = len(schedule) if schedule is not None else lock_penalty
        else:
            msg = "Other metrics than crossing and exact not implemented yet."  # pragma: no cover
            raise NotImplementedError(msg)
        best_steiner: steiner_type | None = steiner_dct
        best_cost = cost
        best_move_type_lst = {}
        layout.copy()
        best_schedule = schedule.copy() if self.metric == "exact" and schedule is not None else None
        # ! adapt the logical positions etc
        cost_history = [(cost, next_layers[:k_lookahead])]  # add initial cost to cost history
        steiner_history = []
        graph_history = []

        T = T_start  # noqa: N806
        for _step in range(max_iters):
            candidate, g_temp_temp = self.perturbation(steiner_dct, radius, vdp_dict)
            graph_history.append(g_temp_temp)
            # !NOT NEEDED ANYMORE
            if candidate is None:  # i.e. if no other element could be found
                warnings.warn(  # type: ignore[unreachable]
                    "No new neighborhood could be explored. Either you are stuck in a local minimum or simply used to manny iters",
                    stacklevel=2,
                )
                break  # early break

            # after the perturbation you have to update the logical pos, otherwise you will run into issues
            next_layers_copy = next_layers.copy()
            move_type_lst_temp: dict[tuple[pos, pos] | tuple[pos, pos, pos], str] = {}
            # compute the cost of the candidate
            # 1. change the position of the target/control to the ancilla spot for all paths (adapt next_layer)
            logical_pos_temp = self.logical_pos_temp.copy()  # type: ignore[attr-defined]
            layout_rev = {j: i for i, j in layout.items()}
            layout_mod = layout.copy()
            for key_candidate in candidate:
                if len(key_candidate) == 3:
                    (a, b, terminal) = key_candidate
                    # randomly choose whether we shift control to ancilla or target to ancilla
                    move_type = random.choice(["target", "control"])  # noqa: S311
                    move_type_lst_temp.update({(a, b, terminal): move_type})
                    if move_type == "target":
                        for j, next_layer in enumerate(next_layers_copy):  # update all future layers
                            next_layers_copy[j] = self.replace_pos(next_layer, b, terminal)
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(logical_pos_temp, b, terminal)
                        label = layout_rev[b]
                        layout_mod[label] = terminal
                    else:
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(next_layer, a, terminal)
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(logical_pos_temp, a, terminal)
                        label = layout_rev[a]
                        layout_mod[label] = terminal
                elif len(key_candidate) == 2:
                    (a, terminal) = key_candidate
                    # move type is fix, one can only move the "a"
                    move_type = "singlequbit"
                    move_type_lst_temp.update({(a, terminal): move_type})
                    # update layers
                    for j, next_layer in enumerate(next_layers_copy):
                        next_layers_copy[j] = self.replace_pos(next_layer, a, terminal)
                    logical_pos_temp = self.replace_pos(logical_pos_temp, a, terminal)
                    label = layout_rev[a]
                    layout_mod[label] = terminal
                else:
                    msg = "Something wrong with keys of candidate tree"  # type: ignore[unreachable] # pragma: no cover
                    raise RuntimeError(msg)
            # 2. compute the crossing metric for next_layer
            layers_for_metric = next_layers_copy[:k_lookahead]
            if self.metric == "crossing":
                candidate_cost = self.count_crossings(
                    cast("list[list[tuple[pos,pos]]]", layers_for_metric), logical_pos_temp
                )
            elif self.metric == "exact":
                schedule, _ = self.find_total_vdp_layers_dyn(
                    layers_for_metric, logical_pos_temp, factory_times_copy, layout_mod
                )
                candidate_cost = len(schedule) if schedule is not None else lock_penalty
            else:
                msg = "Other metrics than crossing and exact not implemented yet."  # pragma: no cover
                raise NotImplementedError(msg)

            # except ValueError:
            #    continue #skip if some config locks the qubits such that you cannot even evaluate count crossings
            delta = candidate_cost - cost
            if delta < 0 or random.random() < np.exp(-delta / T):  # noqa: S311
                steiner_dct, cost = candidate, candidate_cost
                if cost < best_cost:  # update the best cost
                    best_steiner, best_cost = steiner_dct.copy(), cost
                    best_move_type_lst = move_type_lst_temp.copy()
                    best_schedule = schedule.copy()  # type: ignore[union-attr]
                    layout_mod.copy()
                cost_history.append((cost, layers_for_metric))
            steiner_history.append(candidate)
            # cool
            T = max(T_end, T * alpha)  # noqa: N806

        # if there is no improvement possible at all, make sure you return a none best steiner
        if len(best_move_type_lst) == 0:
            best_steiner = None
            logger.info("No Steiner improvement possible in this layer.")
        else:
            logger.info("Steiner found for this layer.")

        logger.info("Final Temperature T = %.6e", T)

        return (
            best_steiner,
            best_cost,
            best_schedule,
            cost_history,
            best_move_type_lst,
            steiner_history,
            graph_history,
        )

    def reduce_steiner_moves(
        self,
        steiner_dct: steiner_type,
        move_type_lst: dict[tuple[pos, pos] | tuple[pos, pos, pos], str],
        next_layers: list[list[tuple[pos, pos] | pos]],
        best_cost: int,
        k_lookahead: int,
        layout: dict[int, pos],
    ) -> tuple[
        steiner_type,
        dict[tuple[pos, pos] | tuple[pos, pos, pos], str],
        list[
            dict[
                pos | tuple[pos, pos],
                list[pos],
            ]
        ]
        | None,
    ]:  # steiner_reduced, move_type_lst_red, final_schedule
        """Given some tree solution, make sure that you effectively use as least movements as possible.

        This means, we go through small subsets of the tree solution and check at what point we reach the same optimized cost.
        This can scale horribly if you have too many trees in your solution.
        """
        factory_times_copy = self.factory_times.copy()
        flag = False
        best_dct_temp: steiner_type | None = None
        best_schedule = None
        for r in range(1, len(steiner_dct) + 1):
            for subset in itertools.combinations(steiner_dct.items(), r):
                # translate everything into the setup of the steiner_dct movement
                logical_pos_temp = self.logical_pos_temp.copy()  # type: ignore[attr-defined]
                # dct_temp = {(a,b,terminal): (path1, path2) for (a,b,terminal), (path1, path2) in subset}
                dct_temp = {key: (path1, path2) for key, (path1, path2) in subset}  # allows different types of keys
                next_layers_copy = next_layers.copy()
                layout_mod = layout.copy()
                layout_rev = {j: i for i, j in layout.items()}
                for key_subset in dct_temp:
                    if len(key_subset) == 3:
                        (a, b, terminal) = key_subset
                    elif len(key_subset) == 2:
                        (a, terminal) = key_subset
                    else:
                        msg = "something wrong with subset steiner keys"  # type: ignore[unreachable] # pragma: no cover
                        raise RuntimeError(msg)

                    # randomly choose whether we shift control to ancilla or target to ancilla
                    move_type = move_type_lst[key_subset]
                    if move_type == "target":
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(next_layer, b, terminal)
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(logical_pos_temp, b, terminal)
                        label = layout_rev[b]
                        layout_mod[label] = terminal
                    elif move_type in {
                        "control",
                        "singlequbit",
                    }:  # we denote the control a and the singlequibt a, so this can be summarized
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(next_layer, a, terminal)
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(logical_pos_temp, a, terminal)
                        label = layout_rev[a]
                        layout_mod[label] = terminal
                    else:
                        msg = f"other move type than expected: {move_type}"  # pragma: no cover
                        raise RuntimeError(msg)
                # 2. compute the crossing metric for next_layer
                try:
                    if self.metric == "crossing":
                        candidate_cost = self.count_crossings(
                            cast("list[list[tuple[pos,pos]]]", next_layers_copy[:k_lookahead]), logical_pos_temp
                        )
                    elif self.metric == "exact":
                        schedule, _ = self.find_total_vdp_layers_dyn(
                            next_layers_copy[:k_lookahead],
                            logical_pos_temp,
                            factory_times_copy,
                            layout_mod,
                        )
                        candidate_cost = len(schedule) if schedule is not None else lock_penalty
                    else:
                        msg = "Other metrics than crossing and exact not implemented yet."  # pragma: no cover
                        raise NotImplementedError(msg)
                except ValueError:
                    continue

                if candidate_cost == best_cost:
                    flag = True
                    best_dct_temp = cast("steiner_type", dct_temp)
                    best_schedule = schedule.copy() if self.metric == "exact" else None  # type: ignore[union-attr]
                    break
            if flag:
                break

        if flag:
            steiner_reduced = (
                cast(
                    "steiner_type", best_dct_temp
                )  # {(a,b,terminal): (path1, path2) for (a,b,terminal), (path1, path2) in subset}
            )
            # move_type_lst_red = {(a,b,terminal): move_type_lst[(a,b,terminal)] for (a,b,terminal), (_, _) in best_dct_temp.items()}
            move_type_lst_red = {key: move_type_lst[key] for key, (_, _) in best_dct_temp.items()}  # type: ignore[union-attr]
            final_schedule = best_schedule.copy()  # type: ignore[union-attr]
        else:  # return the inputs if nothing is found
            steiner_reduced = steiner_dct
            move_type_lst_red = move_type_lst
            final_schedule = None
        return steiner_reduced, move_type_lst_red, final_schedule

    def idle_move_back(
        self,
        schedule: Any,  # noqa: ANN401
        danger_qubits: dict[pos, int],
        available_gaps: list[pos],
        danger_qubits_temp: dict[pos, int],
        available_gaps_temp: list[pos],
        layout: dict[int, pos],
        layers: list[list[pos | tuple[pos, pos]]],
        reduce_time_stamp: bool,
        jump_harvesting: bool,
        best_schedule: list[
            dict[
                pos | tuple[pos, pos],
                list[pos],
            ]
        ]
        | None,
    ) -> tuple[
        Any,  # schedule
        dict[pos, int],  # danger_qubits
        list[pos],  # available gaps
        dict[int, pos],  # layout
        list[list[pos | tuple[pos, pos]]],  # layers
    ]:
        """Subroutine of `optimize_layers` to move back qubits in dangerous positions asap."""
        # ruff: noqa: PLR1702
        # instead of adding another schedule_temp, take schedule[-1], adapt it and replace
        schedule_temp = schedule[-1].copy()
        flag_idle_move = False

        # distinguish between the cases with jump_harvest = True and False. If true, you need to check more than schedule[-1] but also the future layers from "best_schedule" which was retrieved during SA.
        # we want to avoid moves if the qubits are included in the k_lookahead layers since we want to guarantee that the routing computed for the metric in SA is the same as effectively used in k_lookahead layers to exploit the SA optimization fully without destroying stuff
        if jump_harvesting and best_schedule is None:
            msg = "If `jump_harvest = True` you need to give a best_schedule as input for idle_move_back!"  # pragma: no cover
            raise ValueError(msg)

        # !Try to move re-allocated qubits back into the left gaps (does not need to be the original position, in case some other gap is closer)
        # !todo priority ordering of the danger_qubits (those which appear earlier in upcoming layers must be attempted to be moved back first)
        danger_qubits_copy = danger_qubits.copy()
        logical_pos_temp = list(layout.values())
        next_layers_copy = layers.copy()
        layout_rev = {j: i for i, j in layout.items()}
        layout_mod = layout.copy()
        # filter the danger_qubits to those which are idling right now? to avoid useless runs
        flattened_vdp_dict_current = [item for pair in schedule_temp["vdp_dict"] for item in pair]
        if jump_harvesting:  # include the (remaining) k_lookahead layers for the current jump because we do not want to alter the stuff from SA for multiple k_lookahead
            for layer in best_schedule:  # type: ignore[union-attr]
                for key in layer:
                    flattened_vdp_dict_current.extend((key[0], key[1]))

        danger_qubits_idling = {
            qubit: time for (qubit, time) in danger_qubits_copy.items() if qubit not in flattened_vdp_dict_current
        }
        danger_qubits_idling = {qubit: time for (qubit, time) in danger_qubits_idling.items() if time <= 0}
        idle_move_labels = []
        vdp_dict = schedule_temp["vdp_dict"]
        for danger_qubit in danger_qubits_idling:
            # todo order the available gaps regarding how close they are to the current danger_qubit
            # go through gaps and take the one to which a path is available
            path_idle = None
            for gap in available_gaps:  # !todo order available_gaps according to distance
                # skip the gap if it is currently occupied by some path
                flag_skip = False

                # create a graph copy where all already occupied ancillas from vdp_dict are removed
                g_copy = self.g.copy()
                initial_nodes = set(g_copy.nodes())
                if schedule_temp["steiner"] is not None:
                    for steiner in schedule_temp["steiner"].values():
                        for node in steiner[0]:
                            if node == gap:
                                flag_skip = True
                            if node != danger_qubit:  # and node != gap:
                                g_copy.remove_node(node)
                        for node in steiner[1]:
                            if node == gap:
                                flag_skip = True
                            if (
                                node in g_copy.nodes() and node != danger_qubit
                            ):  # and node != gap: #at least one node in steiner[1] is already in steiner[0]
                                g_copy.remove_node(node)
                for path in schedule_temp["vdp_dict"].values():
                    for node in path:
                        if node == gap:
                            flag_skip = True
                        if node in g_copy.nodes() and node != danger_qubit:  # and node != gap:
                            g_copy.remove_node(node)

                for post in layout_mod.values():
                    if (
                        post in g_copy.nodes() and post not in {danger_qubit, gap}
                    ):  # !just in case you find a bug, this was node != gap before, i dont know why this worked before i moved this into an own method
                        g_copy.remove_node(post)
                final_nodes = set(g_copy.nodes())
                initial_nodes - final_nodes

                if flag_skip:
                    continue

                path_idle = None
                try:
                    path_idle = nx.dijkstra_path(g_copy, gap, danger_qubit)
                    # danger_qubits_copy.remove(danger_qubit)
                    del danger_qubits_copy[danger_qubit]
                    available_gaps.remove(gap)
                    label_idle = f"idle_{danger_qubit}_to_{gap}"
                    vdp_dict.update({label_idle: path_idle})
                    idle_move_labels.append(label_idle)
                    logical_pos_temp = cast(
                        "list[pos]",
                        self.replace_pos(cast("list[pos|tuple[pos,pos]]", logical_pos_temp), danger_qubit, gap),
                    )
                    for j, next_layer in enumerate(next_layers_copy):  # update all future layers
                        next_layers_copy[j] = self.replace_pos(next_layer, danger_qubit, gap)
                    label = layout_rev[danger_qubit]
                    layout_mod[label] = gap
                    layout_rev = {j: i for i, j in layout_mod.items()}  # !update layout_rev
                    flag_idle_move = True
                    break  # because if path found you do not want to find another path to the same danger qubit
                except nx.NetworkXNoPath:
                    continue
            if path_idle is None:
                logger.info(f"No idling path back could be found for danger qubit {danger_qubit}")

        # reduce the time stamp of danger_qubits_copy by 1
        if reduce_time_stamp:
            danger_qubits_copy = {qubit: time - 1 for (qubit, time) in danger_qubits_copy.items()}

        danger_qubits = danger_qubits_copy.copy()
        # add those danger qubits and empty spots which where added in this layer
        danger_qubits |= danger_qubits_temp
        available_gaps += available_gaps_temp

        # if an element appears both in danger_qubits and available_gaps, this hints to the case that a qubit in a danger position was moved again. hence delete the double elements
        shared_elements = set(danger_qubits.keys()) & set(available_gaps)
        danger_qubits = {qubit: time for (qubit, time) in danger_qubits.items() if qubit not in shared_elements}
        available_gaps = [x for x in available_gaps if x not in shared_elements]

        # !update terminal_pairs, logical_pos etc
        schedule_temp["vdp_dict"] = vdp_dict
        self.logical_pos = logical_pos_temp.copy()
        self.logical_pos_temp = cast("None", logical_pos_temp.copy())  # weird cast but mypy demands it
        schedule_temp["logical_pos"] = self.logical_pos.copy()
        schedule_temp["idle_move_label"] = idle_move_labels.copy()
        layers = next_layers_copy.copy()
        layout = layout_mod.copy()

        if flag_idle_move:
            schedule[-1] = schedule_temp.copy()

        return schedule, danger_qubits, available_gaps, layout, layers

    def optimize_layers(
        self,
        terminal_pairs: list[tuple[pos, pos] | pos],
        layout: dict[int, pos],
        max_iters: int,
        T_start: float,  # noqa: N803
        T_end: float,  # noqa: N803
        alpha: float,
        radius: int,
        k_lookahead: int,
        steiner_init_type: str,
        jump_harvesting: bool,
        reduce_steiner: bool,
        idle_move_type: str,
        reduce_init_steiner: bool = False,
        stimtest: bool = False,
    ) -> tuple[Any, list[tuple[int, Any]]]:
        """Optimize the positions in batches of size k_lookahead.

        This means we do the following:

        1. find an initial layer structure

        2. route the first layer, and push remainder into next layers

        3. run SA with k_lookahead layers (i.e. k layers are counted in the metric)

        4. move idling qubits back (in different points in time depending on `idle_move_type.`)
        Repeat this layer by layer.

        `idle_move_type`: str


        Args:
            terminal_pairs (list[tuple[pos, pos]  |  pos]): Circuit of CNOT + T gates in terms of qubit positions `pos` on the lattice.
            layout (dict[int, pos]): mapping between logical qubit labels `int` and their positions `pos` on the lattice.
            max_iters (int): Maximum  number of iterations per simulated annealing run
            T_start (float): Start temperature of each simulated annealing run
            T_end (float): End temperature of each simulated annealing run
            alpha (float): Factor by which the temperature is decreased per iteration in a simulated annealing run. T*alpha per iteration.
            radius (int): Radius in which the 3rd terminal of the tree can be randomly placed (radius in terms of edges on the macroscopic routing graph)
            k_lookahead (int): Number of logical lookahead layers on which the optimization is done per simulated annealing.
            steiner_init_type (str): recommended to use `full_random`. on_path_random: terminal placement on the routing path - or full_random: fully random placement on the macroscopic routing graph.
            jump_harvesting (bool): Recommended to use True. False: Default sliding window method. True: 1 layer is used for steiner search of k future layers. but then you do not just iterate through EACH layer but you skip the k layers, because you do not want to destroy the optimization by optimizing too much!
            reduce_steiner (bool): Recommended to use True. Decides whether the trees are reduced again (True), such that possibly useless movements are removed. Using False may reduce rutnime however.
            idle_move_type (str): Recommended to use `later`. asap: moving back is done as frequent as possible. this may destroy however structure of the predicted routing from the steiner search. later: means that moving back is only done when the steiner search is done. if a locking occurs, extra layers with moving back are necessary, but no moving back during the routing of the k_lookahead layers with jump_harvesting = True.
            reduce_init_steiner (bool, optional): Defaults to False and is recommended not to be changed.
            stimtest (bool, optional): If True, at the end of the computation the schedule is sanity checked. Defaults to False but recommended to use.

        Returns:
            tuple[Any, list[tuple[int, Any]]]: schedule and improvement_history.
        """
        if idle_move_type not in {"asap", "later"}:
            msg = "`move_idle_type` must be `asap` or `later`"  # pragma: no cover
            raise ValueError(msg)

        schedule: Any = []
        filename = f"schedule_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}.pkl"  # logging filename

        available_gaps: list[pos] = []  # a list of gap positions which are free due to moves
        danger_qubits: dict[
            pos, int
        ] = {}  # a dict of current dangerous qubits and time label. after k layers they should be moved back again.
        improvement_history: list[tuple[int, Any]] = []

        # initialize a layering, but this will by dynamically adapted via pushing
        if self.use_dag:
            dag = dag_helper.terminal_pairs_into_dag(terminal_pairs, layout)
            layers = [dag_helper.extract_layer_from_dag(dag, layout, layer) for layer in range(len(list(dag.layers())))]
        else:
            layers = self.split_layer_terminal_pairs(terminal_pairs)
        if len(layers) == 1:
            msg = "Your choice of terminal pairs does not lead to multiple layers. This method is not suitable for your input."  # pragma: no cover
            raise RuntimeError(msg)
        # if some qubit is locked due to replacement, you will end up in an infinite loop here. make sure this does not happen
        no_progress_counter = 0
        it = -1
        while len(layers) != 0:
            flag_skip_steiner = False
            it += 1
            schedule_temp: dict[str, Any] = {
                "steiner": None,
                "vdp_dict": None,
                "move_type": None,
                "logical_pos": None,
                "layout": None,
                "cost_history": None,
                "idle_move_label": None,
            }
            schedule_temp_for_later = schedule_temp.copy()
            # find vdp solution for the front layer (we adapt layers dynamically, meaning we delete stuff which is already routed)
            vdp_dict, terminal_pairs_remainder, self.factory_times = self.find_max_vdp_set(
                layers[0], None, self.factory_times
            )
            logger.info(
                "Iteration %d: |vdp_dict|=%d, pushing |terminal_pairs_remainder|=%d, remaining |layers|=%d",
                it,
                len(vdp_dict),
                len(terminal_pairs_remainder),
                len(layers),
            )
            if len(vdp_dict) == 0:
                no_progress_counter += 1
            if no_progress_counter > self.t:
                warnings.warn(
                    "For more than t (=reset time) layers you could not route anything. Most likely a qubit you need got locked in.",
                    stacklevel=2,
                )
                return schedule, improvement_history
            if len(layers) == 1 and len(terminal_pairs_remainder) == 0:
                # no more steiner tree needed
                schedule_temp["vdp_dict"] = vdp_dict
                schedule_temp["logical_pos"] = self.logical_pos.copy()
                schedule_temp["layout"] = layout.copy()
                schedule.append(schedule_temp)
                with pathlib.Path(filename).open("wb") as f:
                    pickle.dump(schedule, f)
                break
            # remove what is already routed from `layers` + push the remainder into the next layers
            if self.use_dag:
                layers, dag = dag_helper.push_remainder_into_layers_dag(
                    dag, terminal_pairs_remainder, layout, layers[0]
                )
            else:
                layers = self.push_remainder_into_layers(layers, terminal_pairs_remainder)

            # update factory times
            for key in self.factory_times:
                if self.factory_times[key] != 0:
                    self.factory_times[key] -= 1

            if idle_move_type == "later":
                # this does idle moves BEFORE we do the steiner search, i.e. later the metric and real routing will coincide.
                # if vdp_dict is empty, this also adds layers of pure idle moves to avoid getting stuck.
                reduce_time_stamp = True
                schedule_temp["vdp_dict"] = vdp_dict

                # if an element appears both in danger_qubits and available_gaps, this hints to the case that a qubit in a danger position was moved again. hence delete the double elements
                # this is done in idle_move_back, but needs to be done beforehand too
                shared_elements = set(danger_qubits.keys()) & set(available_gaps)
                danger_qubits = {qubit: time for (qubit, time) in danger_qubits.items() if qubit not in shared_elements}
                available_gaps = [x for x in available_gaps if x not in shared_elements]

                # do not overwrite schedule, because this idle move back is too early to easily directly overwrite schedule
                # extract important info from schedule_temp_temp into schedule_temp
                schedule_temp_temp, danger_qubits, available_gaps, layout, layers = self.idle_move_back(
                    [schedule_temp],
                    danger_qubits,
                    available_gaps,
                    {},
                    [],
                    layout,
                    layers,
                    reduce_time_stamp,
                    False,
                    None,
                )
                vdp_dict = schedule_temp_temp[-1]["vdp_dict"]
                assert len(schedule_temp_temp) == 1, "internal error if this is not right"
                schedule_temp = schedule_temp_temp[-1]

            # find optimal steiner tree(s) for current layer
            if reduce_init_steiner:
                steiner_dct = self.initialize_steiner(
                    cast("dict[str|pos|tuple[pos,pos], list[pos]]", vdp_dict),
                    steiner_init_type,
                    layers=layers,
                    k_lookahead=k_lookahead,
                )
                if len(steiner_dct) == 0:
                    best_steiner_init = None
                else:
                    (
                        best_steiner_init,
                        best_cost,
                        best_schedule,
                        cost_history,
                        move_type_lst,
                        _steiner_history,
                        _graph_history,
                    ) = self.run_annealing(
                        layers,
                        steiner_dct,
                        max_iters,
                        T_start,
                        T_end,
                        alpha,
                        k_lookahead,
                        radius=radius,
                        vdp_dict=schedule_temp["vdp_dict"],
                        layout=layout,
                    )
                    # store improvement history
                    improvement_history.append((best_cost, cost_history[0]))
            else:
                steiner_dct = self.initialize_steiner(
                    cast("dict[str|pos|tuple[pos,pos], list[pos]]", vdp_dict),
                    steiner_init_type,
                    layers=None,
                    k_lookahead=None,
                )
                (
                    best_steiner_init,
                    best_cost,
                    best_schedule,
                    cost_history,
                    move_type_lst,
                    _steiner_history,
                    _graph_history,
                ) = self.run_annealing(
                    layers,
                    steiner_dct,
                    max_iters,
                    T_start,
                    T_end,
                    alpha,
                    k_lookahead,
                    radius=radius,
                    vdp_dict=schedule_temp["vdp_dict"],
                    layout=layout,
                )
                # store improvement history
                improvement_history.append((best_cost, cost_history[0]))

            # do not use a steiner if the SA could not find a good best_steiner. then it is set to none
            if best_steiner_init is None:  # break earlier, similar to above
                schedule_temp["vdp_dict"] = vdp_dict
                schedule_temp["logical_pos"] = self.logical_pos.copy()
                schedule_temp["layout"] = layout.copy()
                schedule.append(schedule_temp)
                with pathlib.Path(filename).open("wb") as f:
                    pickle.dump(schedule, f)
                flag_skip_steiner = True

            if flag_skip_steiner is False:
                if reduce_steiner:
                    best_steiner, move_type_lst, best_schedule_temp = self.reduce_steiner_moves(
                        cast("steiner_type", best_steiner_init),
                        move_type_lst,
                        layers,
                        best_cost,
                        k_lookahead,
                        layout,
                    )
                    if best_steiner_init is not None:
                        if len(best_steiner) < len(best_steiner_init):
                            logger.info("Complexity of Steiner could be reduced.")
                        best_schedule = best_schedule_temp.copy()  # type: ignore[union-attr]

                else:
                    best_steiner = cast("steiner_type", best_steiner_init)  # only rename

                # update the logical pos etc for the next iteration
                logical_pos_temp = self.logical_pos_temp.copy()  # type: ignore[attr-defined]
                layout_rev = {j: i for i, j in layout.items()}
                layout_mod = layout.copy()
                schedule_temp["layout"] = (
                    layout.copy()
                )  # add current layout, the adapted layout is for the next iteration.
                next_layers_copy = layers.copy()
                available_gaps_temp = []  # need temporary list, because you do not want to add it to the log already since you cannot move those newly added danger qubits in this current layer, you need to do it in the next
                danger_qubits_temp = {}
                for key_tree in best_steiner:
                    if len(key_tree) == 3:
                        (a, b, terminal) = key_tree
                    elif len(key_tree) == 2:
                        (a, terminal) = key_tree
                    else:
                        msg = "something wrong with keys of best_steiner"  # type: ignore[unreachable] # pragma: no cover
                        raise RuntimeError(msg)

                    # update the available_gaps and danger_qubits (add and remove accordingly later)
                    move_type = move_type_lst[key_tree]
                    if terminal in available_gaps:  # if terminal is in available_gaps you need to remove it
                        available_gaps.remove(terminal)
                    elif not jump_harvesting:  # case distinction to make sure that not k-1=1-1=0 if lookahead=1
                        danger_qubits_temp.update({terminal: k_lookahead})
                    else:
                        danger_qubits_temp.update({terminal: k_lookahead - 1})

                    if move_type == "target":
                        for j, next_layer in enumerate(next_layers_copy):  # update all future layers
                            next_layers_copy[j] = self.replace_pos(next_layer, b, terminal)
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(logical_pos_temp, b, terminal)
                        label = layout_rev[b]
                        layout_mod[label] = terminal
                        available_gaps_temp.append(b)
                    elif move_type in {"control", "singlequbit"}:
                        for j, next_layer in enumerate(next_layers_copy):
                            next_layers_copy[j] = self.replace_pos(next_layer, a, terminal)
                        # update temporary logical pos such that correct nodes are removed from g_temp in perturbation method
                        logical_pos_temp = self.replace_pos(logical_pos_temp, a, terminal)
                        label = layout_rev[a]
                        layout_mod[label] = terminal
                        available_gaps_temp.append(a)
                    else:
                        msg = f"other move type than expected: {move_type}"  # pragma: no cover
                        raise RuntimeError(msg)
                self.logical_pos = logical_pos_temp.copy()
                self.logical_pos_temp = logical_pos_temp.copy()
                layers = next_layers_copy.copy()
                layout = layout_mod.copy()
                # add everything to solution
                schedule_temp["steiner"] = best_steiner
                schedule_temp["vdp_dict"] = vdp_dict
                schedule_temp["move_type"] = move_type_lst
                schedule_temp["logical_pos"] = self.logical_pos.copy()
                schedule_temp["cost_history"] = cost_history

                schedule.append(schedule_temp)
            else:
                danger_qubits_temp = {}  # trivial lists to avoid error below
                available_gaps_temp = []

            # attempt idle moving back if asap
            if idle_move_type == "asap":
                if not jump_harvesting:
                    reduce_time_stamp = True
                    schedule, danger_qubits, available_gaps, layout, layers = self.idle_move_back(
                        schedule,
                        danger_qubits,
                        available_gaps,
                        danger_qubits_temp,
                        available_gaps_temp,
                        layout,
                        layers,
                        reduce_time_stamp,
                        jump_harvesting,
                        None,
                    )
                elif jump_harvesting and flag_skip_steiner:  # type: ignore[redundant-expr]
                    reduce_time_stamp = True
                    schedule, danger_qubits, available_gaps, layout, layers = self.idle_move_back(
                        schedule,
                        danger_qubits,
                        available_gaps,
                        danger_qubits_temp,
                        available_gaps_temp,
                        layout,
                        layers,
                        reduce_time_stamp,
                        False,
                        None,
                    )  # because otherwise error
                elif jump_harvesting and not flag_skip_steiner:  # type: ignore[redundant-expr]
                    reduce_time_stamp = False
                    schedule, danger_qubits, available_gaps, layout, layers = self.idle_move_back(
                        schedule,
                        danger_qubits,
                        available_gaps,
                        danger_qubits_temp,
                        available_gaps_temp,
                        layout,
                        layers,
                        reduce_time_stamp,
                        jump_harvesting,
                        best_schedule,
                    )
                else:
                    msg = "Other combinations of jump_harvesting and flag_skip_steiner relevant???"  # type: ignore[unreachable] # pragma: no cover
                    raise RuntimeError(msg)
            elif idle_move_type == "later":
                # since the idle move is before the steiner search we do not need danger_qubits_temp actually, but needs to be added nevertheless.
                danger_qubits |= danger_qubits_temp.copy()
                available_gaps += available_gaps_temp.copy()

            with pathlib.Path(filename).open("wb") as f:
                pickle.dump(schedule, f)

            flag_finished = False
            layers_k = layers[:k_lookahead].copy()
            layers_after_k = layers[k_lookahead:].copy()
            if (
                jump_harvesting and best_steiner_init is not None and k_lookahead > 1
            ):  # if no best steiner found, just standard further iterations
                # route the next k_lookahead-1 layers without steiner optimization to "harvest" the full potential of previous optimization and without disturbing the previous optimization by new steiner moves
                # make the temp files empty because otherwise repetitive action which is superfluous
                available_gaps_temp = []
                danger_qubits_temp = {}
                if self.metric != "exact":
                    msg = "if `jump_harvesting=True` you also need to use the exact metric"  # pragma: no cover
                    raise ValueError(msg)
                # this is appears like a redundant routing step, but the routes from best_schedule will not necessarily fully coincide with the routing here, since idle_move_back can make it happen that routings can be shorter than in best_schedule
                flag_identical_schedules = True  # the schedules of best_schedule and the routing here can be the same. however, if there is some idle move it can happen that the routing here becomes better than in the computation of the metric
                vdp_dict_present_temp = []  # !DELETE THIS AGAIN ONLY FOR DEBUGGING

                # this routing is effectively redundant, but if idle_type = asap it is important to route again because the schedule can alter due to moves.
                # while len(best_schedule) > 1 : #1 not 0 because the very last layer should be used for a steiner opt again in next it
                if self.use_dag:
                    terminal_pairs_temp = []
                    for layer_temp in layers_k:
                        terminal_pairs_temp += layer_temp
                    dag_k = dag_helper.terminal_pairs_into_dag(terminal_pairs_temp, layout)
                while len(layers_k) > 1:  # loop based on layers_k because otherwise "asap" may run into skipped gates.
                    # initialize another schedule temp
                    schedule_temp = schedule_temp_for_later.copy()
                    # route
                    vdp_dict, terminal_pairs_remainder, self.factory_times = self.find_max_vdp_set(
                        layers_k[0], self.logical_pos, self.factory_times
                    )
                    if len(layers_k) == 1 and len(terminal_pairs_remainder) == 0:
                        # no further steps needed
                        schedule_temp["vdp_dict"] = vdp_dict
                        schedule_temp["logical_pos"] = self.logical_pos.copy()
                        schedule_temp["layout"] = layout.copy()
                        schedule.append(schedule_temp)
                        with pathlib.Path(filename).open("wb") as f:
                            pickle.dump(schedule, f)
                        flag_finished = True
                        break

                    # update factory times
                    for key in self.factory_times:
                        if self.factory_times[key] != 0:
                            self.factory_times[key] -= 1

                    vdp_dict_present_temp.append(vdp_dict)
                    # check whether vdp_dict is equivalent to current best_schedule and remove it from best schedule
                    if (
                        flag_identical_schedules
                    ):  # only test as long as we expect identical schedules. this is not all the time the case.
                        matching = True
                        for vdp_key in cast("list[dict[pos | tuple[pos, pos],list[pos],]]", best_schedule)[0]:
                            if vdp_key not in vdp_dict:
                                matching = False
                        if idle_move_type == "asap":  # if asap idle move type then there's no problem if matching wrong
                            del cast("list[dict[pos | tuple[pos, pos],list[pos],]]", best_schedule)[0]

                        elif idle_move_type == "later":
                            if matching:
                                del cast("list[dict[pos | tuple[pos, pos],list[pos],]]", best_schedule)[0]
                            else:
                                msg = "Mismatch between exact metric routing and real routing in jump harvest. If you do not care you should turn on `idle_move_type == asap`"  # pragma: no cover
                                raise RuntimeError(msg)

                    # push
                    if self.use_dag:
                        layers_k, dag_k = dag_helper.push_remainder_into_layers_dag(
                            dag_k, terminal_pairs_remainder, layout, layers_k[0]
                        )
                    else:
                        layers_k = self.push_remainder_into_layers(layers_k, terminal_pairs_remainder)

                    # update and add another schedule_temp
                    schedule_temp["vdp_dict"] = vdp_dict
                    schedule_temp["logical_pos"] = self.logical_pos.copy()  # redundant info
                    schedule_temp["layout"] = layout.copy()  # redundant info
                    schedule.append(schedule_temp)  # this schedule is adapted in case "asap"

                    # also try idle moves back
                    reduce_time_stamp = False
                    if idle_move_type == "asap":
                        schedule, danger_qubits, available_gaps, layout, layers_k = self.idle_move_back(
                            schedule,
                            danger_qubits,
                            available_gaps,
                            danger_qubits_temp,
                            available_gaps_temp,
                            layout,
                            layers_k,
                            reduce_time_stamp,
                            jump_harvesting,
                            best_schedule,
                        )
                        # what qubits where moved?
                        new_idle_moves = [
                            x for x in schedule[-1]["vdp_dict"] if isinstance(x, str) and x.startswith("idle")
                        ]
                        danger_gap_list = []
                        for label_idle in new_idle_moves:
                            parts = label_idle.split("_")
                            danger_qubit_str = parts[1]
                            gap_str = parts[3]
                            danger_qubit = tuple(map(int, danger_qubit_str.strip("()").split(",")))  # into tuple again
                            gap = tuple(map(int, gap_str.strip("()").split(",")))
                            danger_gap_list.append((danger_qubit, gap))
                        # also layers_after_k need to be updated if there was something moved back
                        for danger_qubit, gap in danger_gap_list:
                            for j, next_layer in enumerate(layers_after_k):  # update all future layers
                                layers_after_k[j] = self.replace_pos(
                                    next_layer, cast("pos", danger_qubit), cast("pos", gap)
                                )

                # update global layers here. with the terminal pairs remainder which was the last in the above loop
                layers = layers_k.copy() + layers_after_k.copy()

                # after adding, reinitialize the layers structure again from the dag
                # reinitialize dag here, because structure changed!!!
                if self.use_dag:
                    terminals_temp_for_reinit = []
                    for layer in layers:
                        terminals_temp_for_reinit += layer

                    dag = dag_helper.terminal_pairs_into_dag(terminals_temp_for_reinit, layout)
                    # also update layers
                    layers = []
                    for layer2 in range(len(list(dag.layers()))):
                        layers.append(dag_helper.extract_layer_from_dag(dag, layout, layer2))

                # reduce the teimstamps in dagner_qubits by k_lookahead (because initialized this way), but in the previous loop we avoided the stepwise reduction of the time labes
                danger_qubits = {qubit: time - k_lookahead + 1 for (qubit, time) in danger_qubits.items()}

            if flag_finished:
                break  # otherwise last layer may appear twice

            with pathlib.Path(filename).open("wb") as f:
                pickle.dump(schedule, f)

        if not tst.check_num_gates(terminal_pairs, schedule):
            warnings.warn(
                "The number of input gates and the routed gates in the schedule do NOT coincide!!", stacklevel=2
            )
        else:
            logger.info("Number of gates in schedule and initial terminal_pairs coincides (:")

        if stimtest:
            if tst.check_order_dyn_gates(terminal_pairs, schedule):
                logger.info("Stim test succeeded: Pushing gates does not cause trouble(:")
            else:
                warnings.warn("Stim test failed: Pushing gates causes trouble):", stacklevel=2)

        # test whether something overlapping
        if tst.check_duplicate_nodes_per_layer(schedule):
            logger.info("No duplicates found in any layer of the schedule - hence all good(:")
        if tst.check_path_on_logical(schedule):
            logger.info("No path/tree is placed on a logical pos. All good(:")
        if tst.test_times_t_gates_opt(schedule, self.t, self.factory_pos):
            logger.info("All good with the reset times (:")

        return schedule, improvement_history
