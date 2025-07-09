# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the Routing."""

from __future__ import annotations

import random

import pytest

import mqt.qecc.co3 as co


@pytest.mark.parametrize(
    ("pos_1", "pos_2", "expected_dist"),
    [((0, 0), (2, 4), 6), ((0, 1), (2, 7), 8), ((0, 5), (4, 5), 8), ((1, 7), (2, 7), 1), ((0, 0), (4, 9), 13)],
)
def test_distance_triangular(pos_1, pos_2, expected_dist):
    """Test Distance."""
    # Setup
    m = 4
    n = 4
    lat = co.HexagonalLattice(m, n)
    dist = lat.distance_triangular(pos_1, pos_2)

    error_message = f"Expected distance between {pos_1} and {pos_2} to be {expected_dist}, but got {dist}"

    assert dist == expected_dist, error_message


def test_shortest_first_router_1():
    """Test routing and check final layers."""
    terminal_pairs = [((1, 0), (1, 5)), ((4, 11), (4, 9)), ((4, 7), (2, 7)), ((3, 10), (1, 10))]
    m, n = 5, 5
    lat = co.ShortestFirstRouter(m, n, terminal_pairs)
    lat.vdp_layers = lat.find_total_vdp_layers()

    expected_vdp_layers = [
        {
            ((4, 11), (4, 9)): [(4, 11), (4, 10), (4, 9)],
            ((4, 7), (2, 7)): [(4, 7), (3, 7), (3, 6), (2, 6), (2, 7)],
            ((3, 10), (1, 10)): [(3, 10), (2, 10), (2, 9), (1, 9), (1, 10)],
            ((1, 0), (1, 5)): [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        }
    ]

    error_message = f"Expected vdp_layers to be {expected_vdp_layers}, but got {lat.vdp_layers}"
    assert lat.vdp_layers == expected_vdp_layers, error_message


def test_shortest_first_router_2():
    """Test number of resulting layers of routing example."""
    terminal_pairs = [((1, 9), (4, 7)), ((4, 11), (2, 7)), ((0, 5), (0, 2)), ((1, 6), (3, 6)), ((2, 6), (3, 5))]
    m, n = 5, 5
    lat = co.ShortestFirstRouter(m, n, terminal_pairs)
    lat.vdp_layers = lat.find_total_vdp_layers()
    error_message = f"Expected 2 layers, but got {len(lat.vdp_layers)}"
    assert len(lat.vdp_layers) == 2, error_message


def test_shortest_first_router_3():
    """Test routing and check final layers for another example."""
    terminal_pairs = [((3, 4), (2, 7)), ((3, 2), (3, 5)), ((0, 2), (1, 3)), ((2, 1), (1, 5)), ((0, 5), (0, 6))]
    m, n = 3, 3
    lat = co.ShortestFirstRouter(m, n, terminal_pairs)
    lat.vdp_layers = lat.find_total_vdp_layers()

    expected_vdp_layers = [
        {
            ((0, 5), (0, 6)): [(0, 5), (0, 6)],
            ((0, 2), (1, 3)): [(0, 2), (1, 2), (1, 3)],
            ((3, 4), (2, 7)): [(3, 4), (2, 4), (2, 5), (2, 6), (2, 7)],
        },
        {((2, 1), (1, 5)): [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5)]},
        {((3, 2), (3, 5)): [(3, 2), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 6), (3, 5)]},
    ]

    error_message = f"Expected vdp_layers to be {expected_vdp_layers}, but got {lat.vdp_layers}"
    assert lat.vdp_layers == expected_vdp_layers, error_message


def test_standard_qubit_locs():
    """Test allocation of qubits (no logical labels) for the standard layout."""
    data_qubit_locs_expected = [(1, 2), (1, 8), (2, 5), (2, 11), (3, 2), (3, 8), (4, 5), (4, 11), (5, 2), (5, 8)]
    m, n = 6, 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_sparse()
    error_message = "Generation of Standard Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message


def test_pair_qubit_locs():
    """Test allocation of qubits (no logical labels) for the pair layout."""
    data_qubit_locs_expected = [
        (1, 2),
        (1, 3),
        (1, 6),
        (1, 7),
        (1, 10),
        (1, 11),
        (3, 2),
        (3, 3),
        (3, 6),
        (3, 7),
        (3, 10),
        (3, 11),
        (5, 2),
        (5, 3),
        (5, 6),
        (5, 7),
        (5, 10),
        (5, 11),
    ]
    m, n = 6, 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_pair()
    error_message = "Generation of Pair Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message


def test_row_qubit_locs():
    """Test allocation of qubits (no logical labels) for the row layout."""
    data_qubit_locs_expected = [
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 2),
        (4, 3),
        (5, 2),
        (5, 3),
        (1, 6),
        (1, 7),
        (2, 6),
        (2, 7),
        (3, 6),
        (3, 7),
        (4, 6),
        (4, 7),
        (5, 6),
        (5, 7),
        (1, 10),
        (1, 11),
        (2, 10),
        (2, 11),
        (3, 10),
        (3, 11),
        (4, 10),
        (4, 11),
        (5, 10),
        (5, 11),
    ]
    m, n = 6, 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_row()
    error_message = "Generation of Row Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message


def test_hex_qubit_locs():
    """Tests allocation of qubits (no logical labels) for hexagonal layout on a m,n networkx grid."""
    data_qubit_locs_expected = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 2),
        (4, 3),
        (4, 4),
        (5, 2),
        (5, 3),
        (5, 4),
        (2, 6),
        (2, 7),
        (2, 8),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 11),
        (3, 12),
        (3, 13),
        (4, 11),
        (4, 12),
        (4, 13),
        (5, 7),
        (5, 8),
        (5, 9),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 12),
        (6, 13),
        (6, 14),
        (7, 12),
        (7, 13),
        (7, 14),
    ]
    m, n = 8, 8
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_hex()
    error_message = "Generation of Hex Layout in HexagonalLattice is faulty."
    assert data_qubit_locs == data_qubit_locs_expected, error_message


def test_dynamic_router():
    """Tests the dynamic routing."""
    pairs: list[tuple[int, int] | int] = [(0, 1), (2, 3), (4, 5), (0, 2), 4, (1, 5), (0, 1), (2, 3)]
    m = 3
    n = 4
    factory_locs = [(1, 7), (3, 7)]
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {
        2: (1, 2),
        5: (1, 3),
        1: (2, 2),
        3: (2, 3),
        0: (3, 2),
        4: (3, 3),
    }
    terminal_pairs = co.translate_layout_circuit(pairs, layout)
    router = co.ShortestFirstRouterTGatesDyn(m, n, terminal_pairs, factory_locs, t=1)
    vdp_layers = router.find_total_vdp_layers_dyn()

    desired_layers = [
        {
            ((3, 2), (2, 2)): [(3, 2), (2, 2)],
            ((3, 3), (1, 3)): [(3, 3), (3, 4), (2, 4), (2, 5), (1, 5), (1, 4), (1, 3)],
        },
        {((1, 2), (2, 3)): [(1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 4), (2, 3)]},
        {
            (3, 3): [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7)],
            ((3, 2), (1, 2)): [(3, 2), (3, 1), (3, 0), (2, 0), (2, 1), (1, 1), (1, 2)],
        },
        {((2, 2), (1, 3)): [(2, 2), (2, 1), (1, 1), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 3)]},
        {
            ((3, 2), (2, 2)): [(3, 2), (2, 2)],
            ((1, 2), (2, 3)): [(1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 4), (2, 3)],
        },
    ]
    assert vdp_layers == desired_layers, (
        "A test instance of routing dynamically vdp layers does not yield the desired result."
    )


def test_ordering_dyn_routing():
    """Tests the ordering of gates from dynamic routing by doing a statevector simulation between initial and post routing gate order with qiskit."""
    # generate layout
    q = 20
    t = 3
    m = 5
    n = 6
    lat = co.HexagonalLattice(m, n)
    data_qubit_locs = lat.gen_layout_row()
    factory_locs = [(1, 11), (3, 11), (5, 11)]
    # generate random circuit
    pairs = co.generate_random_circuit(q, min_depth=q, tgate=True, ratio=0.8)
    # generate random layout
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {}
    perm = list(range(len(data_qubit_locs)))
    random.shuffle(perm)
    for i, j in zip(
        perm, data_qubit_locs
    ):  # this also respects custom layouts, because we adapted self.data_qubit_locs in case of layout_type="custom"
        layout.update({i: (int(j[0]), int(j[1]))})  # otherwise might be np.int64

    terminal_pairs = co.translate_layout_circuit(pairs, layout)
    router = co.ShortestFirstRouterTGatesDyn(m, n, terminal_pairs, factory_locs, t)

    worked = co.compare_original_dynamic_gate_order(q, layout, router)
    assert worked is True, "The ordering of your gates seems to be messed up in the dynamic routing."
