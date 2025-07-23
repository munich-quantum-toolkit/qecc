# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the Routing."""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

test_dir = Path(__file__).resolve().parent
layouts_path = (test_dir / "../../../scripts/co3").resolve()
if str(layouts_path) not in sys.path:
    sys.path.insert(0, str(layouts_path))

import layouts  # type: ignore[import-not-found]
import numpy as np
import pytest
import qiskit as qk
from qiskit.quantum_info import random_statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

import mqt.qecc.co3 as co
from mqt.qecc.co3.utils.lattice_router import plot_lattice_paths


def compare_original_dynamic_gate_order(
    q: int, layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]], router: co.ShortestFirstRouterTGatesDyn
) -> bool:
    """Helper function for `test_ordering_dyn_routing`. Generates a qiskit circuit for both the order after doing dynamic routing and the original order.

    Hence, it is checked whether the many reorderings in dynamic routing are really safe and sound.

    Args:
        q (int): number of qubits
        router (co.ShortestFirstRouterTGatesDyn): router to be checked
        layout (dict): must be the same layout with which the router's terminal_pairs were initialized.

    Returns:
        bool: Whether the final states coincide, i.e. whether dynamic routing is safe.
    """
    assert q <= 20, (
        "Too many qubits cannot be simulated via qiskit statevector simulator anymore. Consder less than 20 qubits."
    )

    # run the dynamic routing once
    vdp_layers_dyn = router.find_total_vdp_layers_dyn()

    # original layers
    gates_previous = []
    for lst in router.layers_copy:
        gates_previous += lst

    # gate order after routing
    gates_routing = []
    for dct in vdp_layers_dyn:
        gates_routing += list(dct.keys())

    # warning if both lists are identical. Then, the test is trivial
    if gates_routing == gates_previous:
        warnings.warn(
            "The test of comparing initial and post-dyn-routing order is trivial if both gates are ordered the same way. Try again to sample a new random circuit.",
            category=RuntimeWarning,
            stacklevel=2,
        )

    reverse_mapping = {v: k for k, v in layout.items()}
    translated_previous: list[tuple[int | str, ...] | int | str] = []
    for item in gates_previous:
        if isinstance(item, tuple):  # If it's a tuple, check if it's a nested pair
            if isinstance(item[0], tuple):  # If it's a tuple of tuples (nested)
                translated_previous.append(tuple(reverse_mapping[sub] for sub in item))
            else:  # If it's a single tuple directly in the list
                translated_previous.append(reverse_mapping[item])

    translated_routing: list[tuple[int | str, ...] | int | str] = []
    for item in gates_routing:
        if isinstance(item, tuple):  # If it's a tuple, check if it's a nested pair
            if isinstance(item[0], tuple):  # If it's a tuple of tuples (nested)
                translated_routing.append(tuple(reverse_mapping[sub] for sub in item))
            else:  # If it's a single tuple directly in the list
                translated_routing.append(reverse_mapping[item])

    # initialize random state (s.t. CNOT and T are not trivially appplied)
    random_state = random_statevector(2**q)

    # build circuits for previous and after dyn routing order
    # original input order
    qc_previous = qk.QuantumCircuit(q)
    qc_previous.initialize(random_state, range(q))
    qc_previous.barrier()

    for op in translated_previous:
        if isinstance(op, tuple):  # Apply CNOT for (control, target)
            qc_previous.cx(op[0], op[1])
        else:  # Apply Hadamard for single qubit
            qc_previous.h(op)

    backend = AerSimulator(method="statevector")
    qc_previous.save_statevector()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_combine = pm.run(qc_previous)

    result = backend.run(qc_combine, shots=1)
    psi_out_complex = result.result()

    # after dyn routing order
    qc_routing = qk.QuantumCircuit(q)
    qc_routing.initialize(random_state, range(q))
    qc_routing.barrier()

    for op in translated_routing:
        if isinstance(op, tuple):  # Apply CNOT for (control, target)
            qc_routing.cx(op[0], op[1])
        else:  # Apply Hadamard for single qubit
            qc_routing.h(op)

    backend2 = AerSimulator(method="statevector")
    qc_routing.save_statevector()
    pm2 = generate_preset_pass_manager(backend=backend2, optimization_level=1)
    qc_combine = pm2.run(qc_routing)

    # result2 = backend2.run(qc_routing, shots=1)
    result2 = backend2.run(qc_combine, shots=1)
    psi_out_complex_2 = result2.result()

    diff = np.linalg.norm(psi_out_complex.data()["statevector"] - psi_out_complex_2.data()["statevector"])
    return math.isclose(diff, 0, abs_tol=1e-14)


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

    worked = compare_original_dynamic_gate_order(q, layout, router)
    assert worked is True, "The ordering of your gates seems to be messed up in the dynamic routing."


# ------mock tests for the plotting functions----------
def test_plot_lattice():
    """Just runs a plot for completeness."""
    lat = co.HexagonalLattice(3, 3)
    size = (5, 5)  # size of the plot
    lat.plot_lattice(size=size)


def test_plot_lattice_paths():
    """Just runs a plot for completeness."""
    lat = co.HexagonalLattice(10, 10)
    g = lat.G
    vdp_dict: dict[tuple[int, int] | tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {
        (2, 5): [(2, 5), (1, 5), (1, 4), (0, 4), (0, 3)]
    }  # only one path to a factory
    factories = [(0, 3)]
    size = (3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=mpl.MatplotlibDeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        plot_lattice_paths(g, vdp_dict, factory_locs=factories, layout=None, size=size)


# ---------test some more cases for the dynamic routing which we actually want to use--------------


def test_parallel_cnot():
    """Takes a very parallel circuit with CNOTs and checks the result."""
    factories: list[tuple[int, int]] = []
    g, data_qubit_locs, _ = layouts.gen_layout("hex", 24, factories)
    pairs: list[tuple[int, int] | int] = [
        (7, 5),
        (20, 6),
        (0, 12),
        (16, 15),
        (17, 9),
        (22, 13),
        (21, 8),
        (14, 11),
        (1, 10),
        (3, 2),
        (4, 19),
        (23, 18),
    ]
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {}
    for i, j in zip(range(len(data_qubit_locs)), data_qubit_locs):
        layout.update({i: (int(j[0]), int(j[1]))})

    terminal_pairs = co.translate_layout_circuit(pairs, layout)
    m, n = 10, 10  # random assignment since g is replaced anyways
    router = co.ShortestFirstRouterTGatesDyn(m, n, terminal_pairs, factories, t=2)
    router.G = g
    vdp_layers = router.find_total_vdp_layers_dyn()

    vdp_layers_ref = [
        {
            ((4, 10), (4, 9)): [(4, 10), (5, 10), (5, 9), (5, 8), (4, 8), (4, 9)],
            ((7, 10), (6, 6)): [(7, 10), (7, 9), (7, 8), (7, 7), (7, 6), (6, 6)],
            ((3, 6), (3, 5)): [(3, 6), (3, 7), (4, 7), (4, 6), (4, 5), (3, 5)],
            ((2, 4), (3, 9)): [(2, 4), (2, 3), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9)],
            ((7, 11), (3, 10)): [
                (7, 11),
                (8, 11),
                (8, 12),
                (8, 13),
                (7, 13),
                (7, 14),
                (6, 14),
                (6, 13),
                (5, 13),
                (5, 12),
                (4, 12),
                (4, 13),
                (3, 13),
                (3, 12),
                (2, 12),
                (2, 11),
                (2, 10),
                (3, 10),
            ],
        },
        {
            ((4, 11), (6, 7)): [(4, 11), (4, 12), (5, 12), (5, 11), (5, 10), (5, 9), (6, 9), (6, 8), (6, 7)],
            ((3, 4), (5, 7)): [(3, 4), (3, 3), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7)],
            ((6, 12), (5, 5)): [
                (6, 12),
                (6, 13),
                (6, 14),
                (7, 14),
                (7, 13),
                (8, 13),
                (8, 12),
                (8, 11),
                (8, 10),
                (8, 9),
                (7, 9),
                (7, 8),
                (7, 7),
                (7, 6),
                (7, 5),
                (7, 4),
                (6, 4),
                (6, 3),
                (5, 3),
                (5, 4),
                (5, 5),
            ],
        },
        {
            ((3, 11), (5, 6)): [
                (3, 11),
                (3, 12),
                (2, 12),
                (2, 11),
                (2, 10),
                (2, 9),
                (2, 8),
                (3, 8),
                (3, 7),
                (4, 7),
                (4, 6),
                (5, 6),
            ],
            ((7, 12), (6, 10)): [
                (7, 12),
                (7, 13),
                (7, 14),
                (6, 14),
                (6, 13),
                (5, 13),
                (5, 12),
                (5, 11),
                (5, 10),
                (5, 9),
                (6, 9),
                (6, 10),
            ],
            ((6, 5), (2, 5)): [
                (6, 5),
                (6, 4),
                (6, 3),
                (5, 3),
                (5, 4),
                (4, 4),
                (4, 3),
                (3, 3),
                (3, 2),
                (2, 2),
                (2, 3),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 5),
            ],
        },
        {
            ((2, 6), (6, 11)): [
                (2, 6),
                (2, 7),
                (2, 8),
                (3, 8),
                (3, 7),
                (4, 7),
                (4, 8),
                (5, 8),
                (5, 9),
                (5, 10),
                (5, 11),
                (6, 11),
            ]
        },
    ]

    assert vdp_layers == vdp_layers_ref, (
        "A test instance of routing dynamically vdp layers does not yield the desired result."
    )


def test_parallel_cnot_t_dyn():
    """Test a quite parallel circuit with both cnot + t gates with the dynamic routing."""
    factories = [
        (0, 3),
        (1, 8),
        (2, 13),
        (7, 3),
        (8, 8),
        (9, 13),
        (4, 2),
        (5, 14),
    ]
    g, data_qubit_locs, _ = layouts.gen_layout("hex", 24, factories)
    pairs: list[tuple[int, int] | int] = [
        (7, 5),
        20,
        (0, 12),
        6,
        (16, 15),
        (17, 9),
        (22, 13),
        21,
        (14, 11),
        (1, 10),
        (3, 2),
        8,
        4,
        19,
        23,
        18,
    ]
    layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]] = {}
    for i, j in zip(range(len(data_qubit_locs)), data_qubit_locs):
        layout.update({i: (int(j[0]), int(j[1]))})

    terminal_pairs = co.translate_layout_circuit(pairs, layout)
    m, n = 10, 10  # random assignment since g is replaced anyways
    router = co.ShortestFirstRouterTGatesDyn(m, n, terminal_pairs, factories, t=2)
    router.G = g
    vdp_layers = router.find_total_vdp_layers_dyn()

    vdp_layers_ref = [
        {
            ((4, 10), (4, 9)): [(4, 10), (5, 10), (5, 9), (5, 8), (4, 8), (4, 9)],
            ((3, 6), (3, 5)): [(3, 6), (3, 7), (4, 7), (4, 6), (4, 5), (3, 5)],
            ((2, 4), (3, 9)): [(2, 4), (2, 3), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9)],
            ((4, 11), (6, 7)): [
                (4, 11),
                (4, 12),
                (5, 12),
                (5, 13),
                (6, 13),
                (6, 14),
                (7, 14),
                (7, 13),
                (8, 13),
                (8, 12),
                (8, 11),
                (8, 10),
                (8, 9),
                (7, 9),
                (7, 8),
                (6, 8),
                (6, 7),
            ],
        },
        {
            ((3, 4), (5, 7)): [(3, 4), (3, 3), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7)],
            ((7, 11), (3, 10)): [
                (7, 11),
                (8, 11),
                (8, 12),
                (8, 13),
                (7, 13),
                (7, 14),
                (6, 14),
                (6, 13),
                (5, 13),
                (5, 12),
                (4, 12),
                (4, 13),
                (3, 13),
                (3, 12),
                (2, 12),
                (2, 11),
                (2, 10),
                (3, 10),
            ],
        },
        {
            (6, 12): [(6, 12), (6, 13), (5, 13), (5, 14)],
            (7, 10): [(7, 10), (7, 9), (8, 9), (8, 8)],
            (2, 6): [(2, 6), (2, 7), (1, 7), (1, 8)],
            (5, 5): [(5, 5), (5, 4), (4, 4), (4, 3), (4, 2)],
            (6, 6): [(6, 6), (7, 6), (7, 5), (7, 4), (7, 3)],
            (7, 12): [(7, 12), (7, 13), (8, 13), (8, 12), (9, 12), (9, 13)],
            (6, 11): [(6, 11), (5, 11), (5, 12), (4, 12), (4, 13), (3, 13), (3, 12), (2, 12), (2, 13)],
        },
        {
            ((3, 11), (5, 6)): [
                (3, 11),
                (3, 12),
                (2, 12),
                (2, 11),
                (2, 10),
                (2, 9),
                (2, 8),
                (3, 8),
                (3, 7),
                (4, 7),
                (4, 6),
                (5, 6),
            ],
            ((6, 5), (2, 5)): [
                (6, 5),
                (6, 4),
                (6, 3),
                (5, 3),
                (5, 4),
                (4, 4),
                (4, 3),
                (3, 3),
                (3, 2),
                (2, 2),
                (2, 3),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 5),
            ],
        },
        {(6, 10): [(6, 10), (6, 9), (6, 8), (7, 8), (7, 7), (8, 7), (8, 8)]},
    ]

    assert vdp_layers == vdp_layers_ref, (
        "A test instance of routing dynamically vdp layers does not yield the desired result."
    )
