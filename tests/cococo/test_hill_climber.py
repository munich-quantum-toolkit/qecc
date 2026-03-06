# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the Hill Climbing."""

from __future__ import annotations

from typing import cast

from mqt.qecc.cococo import hill_climber, layouts

pos = tuple[int, int]


def test_neighborhood():
    """Checks an example neighborhood."""
    layout_type = "hex"
    m, n = 1, 1
    factories: list[pos] = []
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges=False)
    custom_layout = [data_qubit_locs, g]
    t = 0
    use_dag = True
    valid_path = "cc"
    metric = "crossing"
    circuit: list[tuple[int, int] | int] = [(5, 4), (3, 1), (2, 0)]
    max_restarts = 2  # just random value
    max_iterations = 2
    hc = hill_climber.HillClimbing(
        max_restarts,
        max_iterations,
        circuit,
        metric,
        t,
        custom_layout,
        use_dag,
        valid_path,
        possible_factory_positions=factories,
        num_factories=len(factories),
        optimize_factories=False,
    )
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {
        1: (0, 0),
        5: (1, 0),
        3: (2, 0),
        0: (2, 1),
        4: (1, 1),
        2: (0, 1),
    }  # type to make mypy happy

    neighborhood = hc.gen_neighborhood(layout)
    aim_neighborhood = [
        {1: (0, 0), 5: (1, 1), 3: (2, 0), 0: (2, 1), 4: (1, 0), 2: (0, 1)},
        {1: (2, 0), 5: (1, 0), 3: (0, 0), 0: (2, 1), 4: (1, 1), 2: (0, 1)},
        {1: (0, 0), 5: (1, 0), 3: (2, 0), 0: (0, 1), 4: (1, 1), 2: (2, 1)},
    ]

    error_message = "Generates unexpected neighborhood."
    assert neighborhood == aim_neighborhood, error_message


def test_crossing_metric():
    """Checks an example cost."""
    layout_type = "hex"
    m, n = 1, 1
    factories: list[pos] = []
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges=False)
    custom_layout = [data_qubit_locs, g]
    t = 0
    use_dag = True
    valid_path = "cc"
    metric = "crossing"
    circuit: list[tuple[int, int] | int] = [(5, 4), (3, 1), (2, 0)]
    max_restarts = 2  # just random value
    max_iterations = 2
    hc = hill_climber.HillClimbing(
        max_restarts,
        max_iterations,
        circuit,
        metric,
        t,
        custom_layout,
        use_dag,
        valid_path,
        possible_factory_positions=factories,
        num_factories=len(factories),
        optimize_factories=False,
    )
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {
        1: (0, 0),
        5: (1, 0),
        3: (2, 0),
        0: (2, 1),
        4: (1, 1),
        2: (0, 1),
        "factory_positions": [],
    }

    cost = hc.evaluate_solution(layout)
    expected_cost = 8

    error_message = "Generates unexpected crossing cost value."
    assert cost == expected_cost, error_message

    # also with exact metric
    metric = "exact"
    max_restarts = 2  # just random value
    max_iterations = 2
    hc = hill_climber.HillClimbing(
        max_restarts,
        max_iterations,
        circuit,
        metric,
        t,
        custom_layout,
        use_dag,
        valid_path,
        possible_factory_positions=factories,
        num_factories=len(factories),
        optimize_factories=False,
    )
    cost = hc.evaluate_solution(layout)
    expected_cost = 2

    error_message = "Generates unexpected exact cost value."
    assert cost == expected_cost, error_message

    # random qubit assignment
    assert len(hc.gen_random_qubit_assignment()) == len(layout), "Weird result for random qubit assignment."


def test_translate_layout_circuit():
    """Checks translate_layout_circuit from misc."""
    pairs: list[tuple[int, int] | int] = [
        10,
        (10, 9),
        (1, 16),
        (22, 23),
        (13, 2),
        8,
        (0, 7),
        (1, 5),
        (19, 8),
        (17, 2),
        6,
        (2, 19),
        (6, 11),
        (5, 0),
        (13, 12),
        19,
        (15, 11),
        (18, 14),
        (0, 10),
        (17, 12),
        8,
        (9, 6),
        (1, 12),
        (15, 22),
        (22, 1),
        (21, 20),
        (4, 20),
        (12, 4),
        (17, 20),
        19,
        (19, 13),
        14,
        (11, 23),
        (10, 8),
        (23, 19),
        (16, 21),
        (10, 3),
        (19, 0),
        (21, 1),
        (3, 0),
        14,
        (10, 12),
        (13, 4),
        (19, 17),
        (22, 20),
        11,
        (16, 9),
        18,
        22,
        (8, 18),
    ]

    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {
        0: (2, 5),
        1: (2, 6),
        2: (3, 6),
        3: (3, 5),
        4: (4, 5),
        5: (4, 6),
        6: (2, 9),
        7: (2, 10),
        8: (3, 10),
        9: (3, 9),
        10: (4, 9),
        11: (4, 10),
        12: (2, 13),
        13: (2, 14),
        14: (3, 14),
        15: (3, 13),
        16: (4, 13),
        17: (4, 14),
        18: (2, 17),
        19: (2, 18),
        20: (3, 18),
        21: (3, 17),
        22: (4, 17),
        23: (4, 18),
        "factory_positions": [
            (0, 3),
            (0, 9),
            (0, 15),
            (6, 6),
            (6, 12),
            (6, 18),
            (5, 3),
            (2, 2),
        ],
    }

    terminal_pairs_desired = [
        (4, 9),
        ((4, 9), (3, 9)),
        ((2, 6), (4, 13)),
        ((4, 17), (4, 18)),
        ((2, 14), (3, 6)),
        (3, 10),
        ((2, 5), (2, 10)),
        ((2, 6), (4, 6)),
        ((2, 18), (3, 10)),
        ((4, 14), (3, 6)),
        (2, 9),
        ((3, 6), (2, 18)),
        ((2, 9), (4, 10)),
        ((4, 6), (2, 5)),
        ((2, 14), (2, 13)),
        (2, 18),
        ((3, 13), (4, 10)),
        ((2, 17), (3, 14)),
        ((2, 5), (4, 9)),
        ((4, 14), (2, 13)),
        (3, 10),
        ((3, 9), (2, 9)),
        ((2, 6), (2, 13)),
        ((3, 13), (4, 17)),
        ((4, 17), (2, 6)),
        ((3, 17), (3, 18)),
        ((4, 5), (3, 18)),
        ((2, 13), (4, 5)),
        ((4, 14), (3, 18)),
        (2, 18),
        ((2, 18), (2, 14)),
        (3, 14),
        ((4, 10), (4, 18)),
        ((4, 9), (3, 10)),
        ((4, 18), (2, 18)),
        ((4, 13), (3, 17)),
        ((4, 9), (3, 5)),
        ((2, 18), (2, 5)),
        ((3, 17), (2, 6)),
        ((3, 5), (2, 5)),
        (3, 14),
        ((4, 9), (2, 13)),
        ((2, 14), (4, 5)),
        ((2, 18), (4, 14)),
        ((4, 17), (3, 18)),
        (4, 10),
        ((4, 13), (3, 9)),
        (2, 17),
        (4, 17),
        ((3, 10), (2, 17)),
    ]

    terminal_pairs_result = layouts.translate_layout_circuit(pairs, layout)

    assert terminal_pairs_result == terminal_pairs_desired, (
        "The translation from layout and circuit to terminal pairs has at least one problem."
    )


def hillclimbing_run():
    """Does an error occur when the hill climbing is run?"""
    factories = [(-4, 11), (3, -1), (5, -1), (9, 2), (8, 0), (5, 13)]
    m, n = 6, 2
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable("pair", m, n, factories, remove_edges=False)
    t = 4

    pairs = [
        (0, 3),
        19,
        (4, 8),
        (19, 13),
        (10, 12),
        (10, 8),
        (7, 8),
        (18, 22),
        (11, 23),
        16,
        (15, 8),
        (20, 1),
        (22, 23),
        (0, 10),
        (2, 9),
        (1, 5),
        (6, 22),
        (14, 20),
        21,
        (19, 2),
        (13, 2),
        (10, 3),
        (4, 22),
        (9, 23),
        (8, 13),
        (13, 2),
        (2, 15),
        20,
        (9, 3),
        (23, 4),
        17,
        23,
        (10, 0),
    ]

    custom_layout = [data_qubit_locs, g]

    hc = hill_climber.HillClimbing(
        max_restarts=3,
        max_iterations=100,
        circuit=cast("list[pos|int]", pairs),
        metric="exact",
        t=t,
        custom_layout=custom_layout,
        use_dag=True,
        valid_path="cc",
        possible_factory_positions=factories,
        num_factories=len(factories),
        optimize_factories=False,
    )

    parallel = True
    processes = 4  # depending on your hardware
    prefix = "./"  # adapt the path depending on where you want to have stored the output
    suffix = "test_hc"
    best_solution1, best_score1, _best_rep1, _score_history1 = hc.run(prefix, suffix, parallel, processes)
    best_solution2, best_score2, _best_rep2, _score_history2 = hc.run(prefix, suffix, False, processes)

    solution_test = {
        20: (0, 1),
        3: (1, 1),
        5: (4, 1),
        18: (5, 1),
        6: (0, 3),
        21: (1, 3),
        14: (4, 3),
        17: (5, 3),
        22: (0, 5),
        11: (1, 5),
        12: (4, 5),
        4: (5, 5),
        1: (0, 7),
        16: (1, 7),
        7: (4, 7),
        19: (5, 7),
        0: (0, 9),
        10: (1, 9),
        9: (4, 9),
        2: (5, 9),
        23: (0, 11),
        15: (1, 11),
        13: (4, 11),
        8: (5, 11),
        "factory_positions": [(-4, 11), (8, 0), (9, 2), (5, 13), (5, -1), (3, -1)],
    }

    assert best_solution1 == solution_test, (
        "Somehow the solution (of the parallel run) does not match up with a previous run's solution. Issue with seed or more serious?"
    )
    assert best_solution2 == solution_test, (
        "Somehow the solution (of the non parallel run) does not match up with a previous run's solution. Issue with seed or more serious?"
    )
    assert best_score1 == 11, "scores do not match (parallel)"
    assert best_score2 == 11, "scores do not match (non parallel)"
