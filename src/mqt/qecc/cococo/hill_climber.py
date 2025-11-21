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
from tqdm import tqdm
from .types import HistoryTemp

from .utils_routing import BasicRouter
from .layouts import translate_layout_circuit

random.seed(45)


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
        metric: str,
        t: int,
        custom_layout: list[list[tuple[int, int]] | nx.Graph],
        use_dag: bool,
        valid_path: str,
        possible_factory_positions: list[tuple[int, int]] | None = None,
        num_factories: int | None = None,
        optimize_factories: bool = False,
    ) -> None:
        """Initializes the Hill Climbing with Random Restarts algorithm.

        IMPORTANT: ALWAYS USE CUSTOM LAYOUTS TO MAKE SURE THAT THE CONNECTIVITY IS CORRECT AND REPRESENTS CC CORRECTLY
        (e.g. you have to remove edges between directly neighboring logical nodes, because a path between those would not allow to put a ancilla for LS)

        Args:
            max_restarts (int): Maximum number of random restarts.
            max_iterations (int): Maximum number of iterations per restart.
            circuit (list[tuple[int,int]]): list of qubits to connect (terminal pairs aka cnots)
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
        Raises:
            ValueError: _description_
        """
        # if circuit includes also single ints (i.e. T gates on qubit i), then ensure, that possible_factory_positions and num_factories are not None
        if any(type(el) is int for el in circuit):
            assert (
                possible_factory_positions is not None
            ), "If T gates included in circuit, `possible_factory_positions` must NOT be None."
            assert (
                num_factories is not None
            ), "If T gates included in circuit, `num_factories` must NOT be None."
            assert (
                t is not None
            ), "If T gates included in circuit, `num_factories` must NOT be None."
            if possible_factory_positions is not None:
                assert (
                    len(possible_factory_positions) >= num_factories
                ), f"`possible_factory_positions` must have more or equal elements than `num_factories`. But {len(possible_factory_positions)} ? {num_factories}"
        else:
            assert (
                optimize_factories is False
            ), "If no T gates present, optimize_factories must be false."
        self.possible_factory_positions = possible_factory_positions
        self.num_factories = num_factories
        self.optimize_factories = optimize_factories
        self.t = t
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations

        self.use_dag = use_dag
        self.valid_path = valid_path
        if valid_path not in {"cc", "sc"}:
            raise ValueError("Incorrect value for `valid_path`.")

        # custom layout types
        data_qubit_locs = custom_layout[0]
        self.g = custom_layout[1]

        self.data_qubit_locs = data_qubit_locs
        assert metric in {"crossing", "exact", "distance"}
        self.metric = metric
        self.circuit = circuit
        if any(type(el) is int for el in circuit):
            flattened_qubit_labels = [
                num
                for tup in self.circuit
                for num in (tup if isinstance(tup, tuple) else (tup,))
            ]
        else:
            flattened_qubit_labels = [
                num for tup in self.circuit if isinstance(tup, tuple) for num in tup
            ]  # isinstance only added for mypy
        self.q = max(flattened_qubit_labels) + 1
        if self.q < len(self.data_qubit_locs):
            self.data_qubit_locs = self.data_qubit_locs[
                : self.q
            ]  # cut-off unnecessary qubit spots.
        assert (
            len(list(set(flattened_qubit_labels))) == self.q
        ), "The available qubits must allow a continuous labeling."
        assert (
            len(data_qubit_locs) >= self.q
        ), "The lattice must be able to host the number of qubits given in the circuit"
        if possible_factory_positions is not None:
            assert (
                set(data_qubit_locs) & set(possible_factory_positions) == set()
            ), "The factory possitions are not allowed to intersect with the logical data qubit locations."

    def evaluate_solution(
        self, layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]]
    ) -> int:
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

        factory_times = {el: self.t for el in factory_positions_temp}
        clean_layout = {k: v for k, v in layout.items() if k != "factory_positions"}

        router: BasicRouter

        router = BasicRouter(
            self.g,
            self.data_qubit_locs,
            factory_positions_temp,
            self.valid_path,
            self.t,
            metric=self.metric,
            use_dag=self.use_dag,
        )

        layers = router.split_layer_terminal_pairs(terminal_pairs)
        terminal_pairs_t = []
        for layer in layers:
            terminal_pairs_t += layer

        cost: int
        if self.metric == "crossing":
            if self.optimize_factories and any(type(el) is int for el in self.circuit):
                cost = np.sum(
                    router.count_crossings_per_layer(layers, t_crossings=True)
                )
            elif self.optimize_factories is False and any(
                type(el) is int for el in self.circuit
            ):
                cost = np.sum(
                    router.count_crossings_per_layer(layers, t_crossings=False)
                )
            else:
                cost = np.sum(router.count_crossings_per_layer(layers))
        elif self.metric == "distance":
            distances = router.measure_terminal_pair_distances()
            cost = np.sum(distances)
            if any(type(el) is int for el in self.circuit):
                raise NotImplementedError
        elif self.metric == "exact":
            vdp_layers, _ = router.find_total_vdp_layers_dyn(
                layers,
                logical_pos=self.data_qubit_locs,
                factory_times=factory_times,
                layout=clean_layout,
                testing=False,  # testing does not work with adapted layouts
            )
            cost = len(vdp_layers)
        return cost

    def gen_random_qubit_assignment(
        self,
    ) -> dict[int | str, tuple[int, int] | list[tuple[int, int]]]:
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
            possible_factory_positions = cast(
                "list[tuple[int,int]]", self.possible_factory_positions
            )
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

                neighborhood.append(layout_copy.copy())

        return neighborhood

    def _parallel_hill_climbing(
        self, restart: int
    ) -> tuple[
        int, dict[int | str, tuple[int, int] | list[tuple[int, int]]], int, HistoryTemp
    ]:
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
        history_temp: HistoryTemp = {
            "scores": [],
            "layout_init": current_solution.copy(),
        }

        for _ in range(self.max_iterations):
            neighbors = self.gen_neighborhood(current_solution)
            if not neighbors:
                break  # No neighbors, end this restart

            # Find the best neighbor
            neighbor_scores = [
                (neighbor, self.evaluate_solution(neighbor)) for neighbor in neighbors
            ]
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
    ) -> tuple[
        dict[int | str, tuple[int, int] | list[tuple[int, int]]],
        int,
        int,
        dict[int, HistoryTemp],
    ]:
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
            + f"hill_climbing_data_q{self.q}_numcnots{len(self.circuit)}_metric{self.metric}_parallel{parallel}"
            + suffix
        )
        self.path_histories = path

        if parallel:
            # Parallel Execution
            with multiprocessing.Pool(processes=processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            self._parallel_hill_climbing, range(self.max_restarts)
                        ),
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
            for restart in tqdm(
                range(self.max_restarts), desc="Hill Climbing Restarts..."
            ):
                base_seed = 45  # You can change this to any fixed value
                seed = base_seed + restart
                random.seed(seed)

                current_solution = self.gen_random_qubit_assignment()
                current_score = self.evaluate_solution(current_solution)
                history_temp: HistoryTemp = {
                    "scores": [],
                    "layout_init": current_solution.copy(),
                }
                for _ in range(self.max_iterations):
                    neighbors = self.gen_neighborhood(current_solution)
                    if not neighbors:
                        break  # No neighbors, end this restart

                    # Find the best neighbor
                    neighbor_scores = [
                        (neighbor, self.evaluate_solution(neighbor))
                        for neighbor in neighbors
                    ]
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

        best_solution = cast(
            "dict[int | str, tuple[int, int] | list[tuple[int, int]]]", best_solution
        )
        best_score = cast("int", best_score)
        best_rep = cast("int", best_rep)
        return best_solution, best_score, best_rep, score_history
