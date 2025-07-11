# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hill Climbing with random restarts for Routing Layouts."""

from __future__ import annotations

import multiprocessing
import operator
import pickle  # noqa: S403
import random
from pathlib import Path
from typing import Any, TypedDict, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import rainbow
from tqdm import tqdm

from .lattice_router import (
    HexagonalLattice,
    ShortestFirstRouter,
    ShortestFirstRouterTGates,
    ShortestFirstRouterTGatesDyn,
)
from .misc import translate_layout_circuit

random.seed(45)


class HistoryTemp(TypedDict, total=False):
    """Type for history dictionaries."""

    scores: list[int]
    layout_init: dict[int | str, Any]  # tuple[int, int] | list[tuple[int, int]]]
    layout_final: dict[int | str, Any]  # tuple[int, int] | list[tuple[int, int]]]


def save_to_file(path: str, data: Any) -> None:  # noqa: ANN401
    """Safely saves data to a file."""
    with Path(path).open("wb") as pickle_file:
        pickle.dump(data, pickle_file)


class HillClimbing:
    """Hill Climbing with random restarts for routing layouts."""

    def __init__(
        self,
        max_restarts: int,
        max_iterations: int,
        circuit: list[tuple[int, int] | int],
        layout_type: str,
        m: int,
        n: int,
        metric: str,
        possible_factory_positions: list[tuple[int, int]] | None = None,
        num_factories: int | None = None,
        free_rows: list[str] | None = None,
        t: int | None = None,
        optimize_factories: bool = False,
        custom_layout: list[list[tuple[int, int]] | nx.Graph] | None = None,
        routing: str = "static",
    ) -> None:
        """Initializes the Hill Climbing with Random Restarts algorithm.

        IMPORTANT: ALWAYS USE CUSTOM LAYOUTS TO MAKE SURE THAT THE CONNECTIVITY IS CORRECT AND REPRESENTS CC CORRECTLY
        (e.g. you have to remove edges between directly neighboring logical nodes, because a path between those would not allow to put a ancilla for LS)

        Args:
            max_restarts (int): Maximum number of random restarts.
            max_iterations (int): Maximum number of iterations per restart.
            circuit (list[tuple[int,int]]): list of qubits to connect (terminal pairs aka cnots)
            layout_type (str): (row, sparse, pair, hex, custom)
            m (int): number of rows of hexagons in the lattice
            n (int): number of columns of hexagons in the lattices
            metric (str): "crossing", "routing", "distance"
            possible_factory_positions (list[tuple[int,int]] | None): possible locations for the factories (must follow nx labeling of hex. lattice and must be placed outside the generated layout)
                ! important: these positions must already respect the "free_rows"!, only the data_qubt_locs are changed depending on free_rows.
            num_factories (int | None): Number of factories to be used (subset of possible_factory_positions).
            free_rows (list[str] | None): Adds one or more rows to lattice, either top or right (easier to implement than also adding bottom, left). Defaults to None.
            t (int): waiting time for factories. Defaults to None
            optimize_factories (int): decides whether factories are optimized or not. Defaults to false.
            custom_layout (list[list[tuple[int,int]] | nx.Graph] | None): Defaults to None because custom layouts not assumed to be standard. The first list in the list should be
                a `data_qubits_loc` of the node locations of data qubits and nx.Graph the corresponding graph (possibly differing from the standard networkx hex graph shape)
                With custom_layout one can avoid using the `free_rows` related stuff.
            routing (str): Defaults to static. Can be "static" or "dynamic". Chooses the routing scheme, whether the layout-agnostic initial layers are dynamically adapted or not.
                Strictly speaking, this is only relevant if the metric `crossing` is used, as this is the only moment when the routing scheme is used in hill climbing.

        Raises:
            ValueError: _description_
        """
        if routing not in {"static", "dynamic"}:
            msg = "Wrong input value for `routing`. Must be `static` or `dynamic`."
            raise ValueError(msg)
        self.routing = routing
        # if circuit includes also single ints (i.e. T gates on qubit i), then ensure, that possible_factory_positions and num_factories are not None
        if any(type(el) is int for el in circuit):
            assert possible_factory_positions is not None, (
                "If T gates included in circuit, `possible_factory_positions` must NOT be None."
            )
            assert num_factories is not None, "If T gates included in circuit, `num_factories` must NOT be None."
            assert t is not None, "If T gates included in circuit, `num_factories` must NOT be None."
            if possible_factory_positions is not None:
                assert len(possible_factory_positions) >= num_factories, (
                    f"`possible_factory_positions` must have more or equal elements than `num_factories`. But {len(possible_factory_positions)} ? {num_factories}"
                )
        else:
            assert optimize_factories is False, "If no T gates present, optimize_factories must be false."
        self.possible_factory_positions = possible_factory_positions
        self.num_factories = num_factories
        self.optimize_factories = optimize_factories
        self.t = t
        self.m = m
        self.n = n
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        assert layout_type in {"row", "sparse", "pair", "hex", "custom"}, "Unknown layout_type!"
        if layout_type == "custom":
            assert free_rows is None, (
                "If custom layout is chosen, the `free_rows` parameter has no meaning and should be set to None."
            )
            assert custom_layout is not None, (
                "If custom layout is chosen, the `custom_layout` must include a data_qubit_locs list as well as the custom graph."
            )
        else:
            assert custom_layout is None, "If no custom layout, also `custom_layout`should be None."
        self.layout_type = layout_type
        lat = HexagonalLattice(m, n)
        self.lat = lat
        if layout_type == "row":
            data_qubit_locs = lat.gen_layout_row()
        elif layout_type == "sparse":
            data_qubit_locs = lat.gen_layout_sparse()
        elif layout_type == "pair":
            data_qubit_locs = lat.gen_layout_pair()
        elif layout_type == "hex":
            data_qubit_locs = lat.gen_layout_hex()
        elif layout_type == "custom":
            if custom_layout is not None:  # only for mypy
                data_qubit_locs = custom_layout[0]
                self.lat.G = custom_layout[1]
        else:
            msg = "unknown layout type"
            raise ValueError(msg)
        assert data_qubit_locs is not None
        self.data_qubit_locs = data_qubit_locs
        assert metric in {"crossing", "routing", "distance"}
        self.metric = metric
        self.circuit = circuit
        if any(type(el) is int for el in circuit):
            flattened_qubit_labels = [
                num for tup in self.circuit for num in (tup if isinstance(tup, tuple) else (tup,))
            ]
        else:
            flattened_qubit_labels = [
                num for tup in self.circuit if isinstance(tup, tuple) for num in tup
            ]  # isinstance only added for mypy
        self.q = max(flattened_qubit_labels) + 1
        if self.q < len(self.data_qubit_locs):
            self.data_qubit_locs = self.data_qubit_locs[: self.q]  # cut-off unnecessary qubit spots.
        assert len(list(set(flattened_qubit_labels))) == self.q, (
            "The available qubits must allow a continuous labeling."
        )
        assert len(data_qubit_locs) >= self.q, (
            "The lattice must be able to host the number of qubits given in the circuit"
        )
        if possible_factory_positions is not None:
            assert set(data_qubit_locs) & set(possible_factory_positions) == set(), (
                "The factory possitions are not allowed to intersect with the logical data qubit locations."
            )

        valid_values = {"right", "top", "left"}
        if free_rows is not None:
            assert set(free_rows) == set(free_rows) & valid_values, (
                "free_rows must only contain 'right' or 'top' and no duplicates."
            )
            # increase the lattice size
            if "right" in free_rows and "top" not in free_rows:  # and "left" not in free_rows:
                self.n += 1
            elif "right" not in free_rows and "top" in free_rows:  # and "left" not in free_rows:
                self.m += 1
            elif "right" in free_rows and "top" in free_rows:  # and "left" not in free_rows:
                self.n += 1
                self.m += 1
            # elif "right" in free_rows and "top" in free_rows and "left" in free_rows:
            #    self.n += 2
            #    self.m += 1
            # elif "right" in free_rows and "top" not in free_rows and "left" in free_rows:
            #    self.n += 2
            # elif "right" not in free_rows and "top" in free_rows and "left" in free_rows:
            #    self.n += 1
            #    self.m += 1
        self.free_rows = free_rows

    @staticmethod
    def add_left_g(g: nx.Graph) -> nx.Graph:
        """Adds a row to the left of the graph (will have negative labels but would like to avoid to change self.data_qubit_locs).

        Args:
            g (nx.Graph): original graph

        Returns:
            g (nx.Graph): graph with additional left column

        """
        pos = nx.get_node_attributes(g, "pos")
        y_diff = 0.8660254037844386

        # filter out leftmost column (x=0)
        left_col = [node for node in g.nodes if node[0] == 0]
        # add another node in this x=0 row
        new_top = (0, max(el[1] for el in left_col) + 1)
        pos_old = pos[0, max(el[1] for el in left_col)]
        left_col.append(new_top)
        pos_new = (pos_old[0] - 0.5, pos_old[1] + y_diff)
        g.add_node(new_top, pos=pos_new)
        g.add_edge(new_top, (0, max(el[1] for el in left_col) - 1))
        pos = nx.get_node_attributes(g, "pos")

        left_nodes = []  # gather together nodes and add them to the graph as well as the horizontal edge
        for node in left_col:
            if node[1] % 2 != 0:
                new_node = (-1, node[1])
                pos_old = pos[node]
                pos_new = (pos_old[0] - 1, pos_old[1])
                g.add_node(new_node, pos=pos_new)
                g.add_edge(new_node, node)
                left_nodes.append(new_node)
        pos = nx.get_node_attributes(g, "pos")

        # add the x=-2 row
        for node1, node2 in zip(left_nodes, left_nodes[1:]):
            new_node = (-2, node1[1] - 1)
            pos_old = pos[node1]
            posy = pos_old[1] + y_diff
            pos_new = (-1.5, posy)
            g.add_node(new_node, pos=pos_new)
            g.add_edge(new_node, node1)
            g.add_edge(new_node, node2)

        return g

    def evaluate_solution(self, layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]]) -> int:
        """Evaluates the layout=solution according to self.metric."""
        terminal_pairs = translate_layout_circuit(self.circuit, layout)
        # factory_positions: list[tuple[int, int]]
        # if type(layout["factory_positions"]) is list[tuple[int,int]] or list:
        factory_positions_temp = layout["factory_positions"]
        if isinstance(factory_positions_temp, list):
            factory_positions: list[tuple[int, int]] = factory_positions_temp
        else:
            msg = "`layout['factory_positions']` must be of type list[tuple[int,int]] but this is not even a list."
            raise TypeError(msg)

        # else:
        #    msg = f"factory positions of layout must be list[tuple[int,int]]. But you got {type(layout['factory_positions'])}"
        #    raise TypeError(msg)
        router: ShortestFirstRouter | ShortestFirstRouterTGatesDyn | ShortestFirstRouterTGates
        if any(type(el) is int for el in self.circuit):
            if self.t is not None:
                if self.routing == "static":
                    router = ShortestFirstRouterTGates(
                        m=self.m, n=self.n, terminal_pairs=terminal_pairs, factory_positions=factory_positions, t=self.t
                    )
                elif self.routing == "dynamic":
                    router = ShortestFirstRouterTGatesDyn(
                        m=self.m, n=self.n, terminal_pairs=terminal_pairs, factory_positions=factory_positions, t=self.t
                    )
                if self.layout_type == "custom":  # Must update the router's g to the customized g
                    router.G = self.lat.G.copy()
                else:  # if not custom, update self.lat.G by the router's G because m,n might differ from initial values.
                    self.lat.G = router.G  # also add to self
                    if self.free_rows is not None:  # for mypy  # noqa: SIM102
                        if "left" in self.free_rows:
                            router.G = self.add_left_g(router.G)
                            self.lat.G = router.G  # also add to self
            else:
                msg = "t must be not None if the circuit contains single integers for T gates."
                raise ValueError(msg)
        else:  # only CNOTs
            if self.routing == "static":
                terminal_pairs_cast = cast("list[tuple[tuple[int, int], tuple[int, int]]]", terminal_pairs)
                router = ShortestFirstRouter(m=self.m, n=self.n, terminal_pairs=terminal_pairs_cast)
            elif self.routing == "dynamic":  # !todo adapt this, because if only cnots, t might not be defined
                t = cast("int", self.t)
                router = ShortestFirstRouterTGatesDyn(
                    m=self.m, n=self.n, terminal_pairs=terminal_pairs, factory_positions=factory_positions, t=t
                )
            if self.layout_type == "custom":  # Must update the router's g to the customized g
                router.G = self.lat.G.copy()
            else:
                self.lat.G = router.G  # also add to self (but should be redundant right? self.lat.G and router.G should be the same anyways if no T gates present)
        cost: int
        if self.metric == "crossing":
            if self.optimize_factories and any(type(el) is int for el in self.circuit):
                router = cast("ShortestFirstRouterTGates | ShortestFirstRouterTGatesDyn", router)
                cost = np.sum(router.count_crossings_per_layer(t_crossings=True))
            elif self.optimize_factories is False and any(type(el) is int for el in self.circuit):
                router = cast("ShortestFirstRouterTGates | ShortestFirstRouterTGatesDyn", router)
                cost = np.sum(router.count_crossings_per_layer(t_crossings=False))
            else:
                cost = np.sum(router.count_crossings_per_layer())
        elif self.metric == "distance":
            router = cast("ShortestFirstRouter", router)
            distances = router.measure_terminal_pair_distances()
            cost = np.sum(distances)
            if any(type(el) is int for el in self.circuit):
                raise NotImplementedError
        elif self.metric == "routing":
            if self.routing == "static":
                vdp_layers = router.find_total_vdp_layers()
            elif self.routing == "dynamic":
                router = cast("ShortestFirstRouterTGatesDyn", router)
                vdp_layers = router.find_total_vdp_layers_dyn()
            cost = len(vdp_layers)
        return cost

    def gen_random_qubit_assignment(self) -> dict[int | str, tuple[int, int] | list[tuple[int, int]]]:
        """Yields a random qubit assignment given the `data_qubit_locs`."""
        layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {}
        perm = list(range(self.q))
        random.shuffle(perm)
        for i, j in zip(
            perm, self.data_qubit_locs
        ):  # this also respects custom layouts, because we adapted self.data_qubit_locs in case of layout_type="custom"
            layout.update({i: (int(j[0]), int(j[1]))})  # otherwise might be np.int64

        # Add generation of random choice of factory positions
        factory_positions = []
        if any(type(el) is int for el in self.circuit):
            possible_factory_positions = cast("list[tuple[int,int]]", self.possible_factory_positions)
            num_factories = cast("int", self.num_factories)
            factory_positions = random.sample(possible_factory_positions, num_factories)
        layout.update({"factory_positions": factory_positions})

        return layout

    def gen_neighborhood(
        self, layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]]
    ) -> list[dict[int | str, tuple[int, int] | list[tuple[int, int]]]]:
        """Creates the Neighborhood of a given layout by going through each terminal pair and swapping their positions.

        If there are no T gates, there will be l=len(terminal_pairs) elements in the neighborhood.

        Args:
            layout (dict): qubit label assignment on the lattice. keys = qubit label, value = node label

        Returns:
            list[dict]: List of layouts constituting the neighborhood.
        """
        neighborhood = []
        for pair in self.circuit:
            if isinstance(pair, tuple):  # only for cnots
                layout_copy = layout.copy()
                # intermediate storage of the nodes
                q_0_pos = layout_copy[pair[0]]
                q_1_pos = layout_copy[pair[1]]
                # swap
                layout_copy[pair[1]] = q_0_pos
                layout_copy[pair[0]] = q_1_pos

                """
                if any(type(el) is int for el in self.circuit) and self.optimize_factories: #if T gates present
                    #adapt the layout["factory_positions"].
                    current = layout_copy["factory_positions"].copy()
                    complement = [el for el in self.possible_factory_positions if el not in current] #all positions from possible positions which are not in factoy_positions
                    for el in current:
                        current_copy = current.copy()
                        current_copy.remove(el) #remove el from current_copy
                        for el_c in complement:
                            current_copy_copy = current_copy.copy()
                            current_copy_copy.append(el_c)
                            layout_copy["factory_positions"] = current_copy_copy.copy()
                            neighborhood.append(layout_copy.copy())
                else:
                """
                neighborhood.append(layout_copy.copy())

        return neighborhood

    def _parallel_hill_climbing(
        self, restart: int
    ) -> tuple[int, dict[int | str, tuple[int, int] | list[tuple[int, int]]], int, HistoryTemp]:
        """Helper method for parallel execution of hill climbing restarts.

        Args:
            restart (int): The restart index.

        Returns:
            Tuple of (restart index, best solution, best score, history for this restart)
        """
        base_seed = 45  # You can change this to any fixed value
        seed = base_seed + restart
        random.seed(seed)

        current_solution = self.gen_random_qubit_assignment()
        current_score = self.evaluate_solution(current_solution)
        history_temp: HistoryTemp = {"scores": [], "layout_init": current_solution.copy()}

        for _ in range(self.max_iterations):
            neighbors = self.gen_neighborhood(current_solution)
            if not neighbors:
                break  # No neighbors, end this restart

            # Find the best neighbor
            neighbor_scores = [(neighbor, self.evaluate_solution(neighbor)) for neighbor in neighbors]
            best_neighbor, best_neighbor_score = min(
                neighbor_scores, key=operator.itemgetter(1)
            )  # Min for minimization

            # If no improvement, stop searching in this path
            if best_neighbor_score >= current_score:
                break

            # Update current solution
            current_solution, current_score = best_neighbor, best_neighbor_score
            history_temp["scores"].append(current_score)

        history_temp.update({"layout_final": current_solution.copy()})
        return restart, current_solution, current_score, history_temp

    def run(
        self, prefix: str, suffix: str, parallel: bool, processes: int = 8
    ) -> tuple[dict[int | str, tuple[int, int] | list[tuple[int, int]]], int, int, dict[int, HistoryTemp]]:
        """Executes the Hill Climbing algorithm with random restarts.

        Args:
            prefix (str): prefix to add to the log file's paths.
            suffix (str): suffix to add to the log file's paths.
            parallel (bool): decides whether to use multiprocessing or not
            processes (int): number of processes (=number of available physical kernels)

        Returns:
            best_solution: The best solution found.
            best_score: The score of the best solution.
        """
        best_solution = None
        best_rep = None
        best_score = float("inf")  # Use '-inf' for maximization, 'inf' for minimization
        score_history: dict[int, HistoryTemp] = {}
        path = (
            prefix
            + f"hill_climbing_data_q{self.q}_numcnots{len(self.circuit)}_layout{self.layout_type}_metric{self.metric}_parallel{parallel}"
            + suffix
        )
        self.path_histories = path

        if parallel:
            # Parallel Execution
            with multiprocessing.Pool(processes=processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._parallel_hill_climbing, range(self.max_restarts)),
                        total=self.max_restarts,
                        desc="Hill Climbing Restarts...",
                    )
                )

                for restart, solution, score, history in results:
                    score_history[restart] = history
                    save_to_file(path, score_history)

                    if score < best_score:
                        best_solution, best_score = solution, score
                        best_rep = restart

        else:  # sequential
            for restart in tqdm(range(self.max_restarts), desc="Hill Climbing Restarts..."):
                base_seed = 45  # You can change this to any fixed value
                seed = base_seed + restart
                random.seed(seed)

                current_solution = self.gen_random_qubit_assignment()
                current_score = self.evaluate_solution(current_solution)
                history_temp: HistoryTemp = {"scores": [], "layout_init": current_solution.copy()}
                for _ in range(self.max_iterations):
                    neighbors = self.gen_neighborhood(current_solution)
                    if not neighbors:
                        break  # No neighbors, end this restart

                    # Find the best neighbor
                    neighbor_scores = [(neighbor, self.evaluate_solution(neighbor)) for neighbor in neighbors]
                    best_neighbor, best_neighbor_score = min(
                        neighbor_scores, key=operator.itemgetter(1)
                    )  # Change to min for minimization

                    # If no improvement, stop searching in this path
                    if best_neighbor_score >= current_score:
                        break

                    # Update current solution
                    current_solution, current_score = best_neighbor, best_neighbor_score
                    history_temp["scores"].append(current_score)

                history_temp.update({"layout_final": current_solution.copy()})
                score_history.update({restart: history_temp})
                with Path(path).open("wb") as pickle_file:
                    pickle.dump(score_history, pickle_file)

                # Update global best solution if current is better
                if current_score < best_score:
                    best_solution, best_score = current_solution, current_score
                    best_rep = restart

        best_solution = cast("dict[int | str, tuple[int, int] | list[tuple[int, int]]]", best_solution)
        best_score = cast("int", best_score)
        best_rep = cast("int", best_rep)
        return best_solution, best_score, best_rep, score_history

    def plot_history(
        self,
        score_history: dict[int, HistoryTemp],
        filename: str = "./hc_history_plot.pdf",
        size: tuple[float, float] = (5, 5),
    ) -> None:
        """Plots the scores for each restart and iteration.

        Args:
            score_history (dict): Score history from HillClimber.run
            filename (str, optional): Path to store the plot. Defaults to "./hc_history_plot.pdf".
            size (tuple[float,float], optional): Size of the plot. Defaults to (3.5,3.5).
        """
        plt.figure(figsize=size)
        for rep, history in score_history.items():
            scores = history["scores"]
            plt.plot(
                range(len(scores)),
                scores,
                "x-",
                color=rainbow(rep / len(list(score_history.keys()))),
                label=f"Restart {rep}",
            )
        plt.legend()
        plt.ylabel(f"{self.metric}")
        plt.xlabel("Hill Climbing Iteration")
        plt.title(f"$q=${self.q}, Layout-Type = {self.layout_type}, Num CNOTS = {len(self.circuit)}")
        plt.savefig(filename)
