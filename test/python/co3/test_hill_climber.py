# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the Hill Climbing."""

from __future__ import annotations

import mqt.qecc.co3 as co


def test_neighborhood():
    """Checks an example neighborhood."""
    layout_type = "row"
    m = 4
    n = 4
    metric = "crossing"
    circuit: list[tuple[int, int] | int] = [(5, 4), (3, 1), (2, 0)]
    max_restarts = 2  # just random value
    max_iterations = 2
    hc = co.utils.hill_climber.HillClimbing(max_restarts, max_iterations, circuit, layout_type, m, n, metric)
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {
        1: (1, 2),
        5: (1, 3),
        3: (2, 2),
        0: (2, 3),
        4: (3, 2),
        2: (3, 3),
    }  # type to make mypy happy

    neighborhood = hc.gen_neighborhood(layout)
    aim_neighborhood = [
        {1: (1, 2), 5: (3, 2), 3: (2, 2), 0: (2, 3), 4: (1, 3), 2: (3, 3)},
        {1: (2, 2), 5: (1, 3), 3: (1, 2), 0: (2, 3), 4: (3, 2), 2: (3, 3)},
        {1: (1, 2), 5: (1, 3), 3: (2, 2), 0: (3, 3), 4: (3, 2), 2: (2, 3)},
    ]

    error_message = "Generates unexpected neighborhood."
    assert neighborhood == aim_neighborhood, error_message


def test_crossing_metric():
    """Checks an example cost."""
    layout_type = "row"
    m = 4
    n = 4
    metric = "crossing"
    circuit: list[int | tuple[int, int]] = [(5, 4), (3, 1), (2, 0)]
    max_restarts = 2  # random number
    max_iterations = 2
    hc = co.utils.hill_climber.HillClimbing(max_restarts, max_iterations, circuit, layout_type, m, n, metric)
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {
        1: (1, 2),
        5: (1, 3),
        3: (2, 2),
        0: (2, 3),
        4: (3, 2),
        2: (3, 3),
        "factory_positions": [],
    }

    cost = hc.evaluate_solution(layout)
    expected_cost = 2

    error_message = "Generates unexpected crossing cost value."
    assert cost == expected_cost, error_message


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
        "factory_positions": [(0, 3), (0, 9), (0, 15), (6, 6), (6, 12), (6, 18), (5, 3), (2, 2)],
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

    terminal_pairs_result = co.translate_layout_circuit(pairs, layout)

    for el1, el2 in zip(terminal_pairs_desired, terminal_pairs_result):
        assert el1 == el2, "The translation from layout and circuit to terminal pairs has at least one problem."
