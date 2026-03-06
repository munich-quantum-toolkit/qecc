# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the functions in circuit_construction and dag_helper."""

from __future__ import annotations

from typing import cast

from mqt.qecc.cococo import circuit_construction, dag_helper, layouts

pos = tuple[int, int]

# ------------------with respect to naive sequential layer structure---------------------


def _split_layers_cnot(
    circuit: list[tuple[int, int] | int],
) -> list[list[tuple[int, int] | int]]:
    """Split the circuit into initial layers.

    The input can also have mere ints, but this is only for mypy. This helper function only assumes tuple[int,int] in the circuit.
    """
    result = []
    current_group: list[tuple[int, int] | int] = []
    seen: set[int] = set()

    for tup in circuit:
        if isinstance(tup, tuple):
            a, b = tup
        else:
            msg = "This function can handle only pure cnot circuits."
            raise TypeError(msg)
        if a in seen or b in seen:
            result.append(current_group)
            current_group = []
            seen = set()
        current_group.append((a, b))
        seen.update([a, b])

    if current_group:
        result.append(current_group)

    return result


def test_generate_max_parallel_circuit():
    """Tests generate_max_parallel_circuit."""
    q = 4
    min_depth = q * 2
    circuit = circuit_construction.generate_max_parallel_circuit(q, min_depth)
    split = _split_layers_cnot(circuit)
    for el in split:
        assert len(el) == q // 2

    q = 10
    min_depth = q * 4
    circuit = circuit_construction.generate_max_parallel_circuit(q, min_depth)
    split = _split_layers_cnot(circuit)
    for el in split:
        assert len(el) == q // 2

    q = 22
    min_depth = q * 4
    circuit = circuit_construction.generate_max_parallel_circuit(q, min_depth)
    split = _split_layers_cnot(circuit)
    for el in split:
        assert len(el) == q // 2


def test_generate_min_parallel_circuit():
    """Tests generate_min_parallel_circuit."""
    q = 10
    min_depth = q * 3

    for layer_size in range(2, 6):
        circuit = circuit_construction.generate_min_parallel_circuit(q, min_depth, layer_size)
        split = _split_layers_cnot(circuit)
        for el in split:
            assert len(el) == layer_size


# --------------------dag---------------------


def test_random_sequential_circuit_dag():
    """Tests create_random_sequential_circuit_dag."""
    layers = 3
    q = 10
    j = 5

    num_gates = j * layers
    dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates)
    layers_dag = len(list(dag.layers()))
    assert layers_dag == layers, "The circuit construction does not yield the correct dag layer number."

    dag_2 = dag_helper.pairs_into_dag_agnostic(cast("list[tuple[int,int]|int]", pairs), q)
    layers_dag_2 = len(list(dag_2.layers()))
    assert layers_dag_2 == layers, "the dag from the pairs does not yield the correct dag layer number."

    cx_count = dag_helper.count_cx_gates_per_layer(dag)
    for cx in cx_count:
        assert cx == j, "The CX count per layer is not right."


def test_remainder_dag_helper():
    """Tests the remaining functions from dag_helper."""
    layout_type = "hex"
    m = 2
    n = 2
    factories = [(5, -1), (10, 0), (7, 7), (12, 5), (-2, 2)]
    _g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges=False)
    pairs = [
        (1, 5),
        0,
        (10, 3),
        (8, 18),
        (13, 2),
        (4, 20),
        (9, 6),
        (11, 23),
        15,
        7,
        (16, 21),
        (22, 23),
        (0, 10),
        (23, 19),
        (10, 12),
        (2, 19),
        (21, 1),
        (13, 4),
        22,
        (14, 13),
        (22, 1),
        (15, 11),
        (22, 20),
        (6, 11),
        (10, 8),
        13,
        19,
        (17, 20),
    ]
    layout = {}
    for i, j in zip(range(len(data_qubit_locs)), data_qubit_locs, strict=False):
        layout.update({i: (int(j[0]), int(j[1]))})
    terminal_pairs = layouts.translate_layout_circuit(
        cast("list[tuple[int, int] | int]", pairs),
        cast("dict[int | str, tuple[int, int] | list[tuple[int, int]]]", layout),
    )  # let's stick to the simple layout

    dag = dag_helper.terminal_pairs_into_dag(terminal_pairs, layout)

    depth = 6
    assert len(list(dag.layers())) == depth, "The depth is not as expected."

    layer0 = [
        0,
        (1, 5),
        7,
        (9, 6),
        (10, 3),
        (13, 2),
        15,
        (8, 18),
        (4, 20),
        (16, 21),
        (11, 23),
    ]

    layer0_test = dag_helper.extract_layer_from_dag_agnostic(dag, 0)

    assert layer0 == layer0_test, "Layer extraction does not work as anticipated."

    layer0_layout = dag_helper.extract_layer_from_dag(dag, layout, 0)
    layer_0_trans = layouts.translate_layout_circuit(
        cast("list[tuple[int, int] | int]", layer0),
        cast("dict[int | str, tuple[int, int] | list[tuple[int, int]]]", layout),
    )

    assert layer0_layout == layer_0_trans, "Layer extraction or translation does not work."

    terminal_pairs_remainder = [(0, 0)]

    layers_updated, _ = dag_helper.push_remainder_into_layers_dag(
        dag, cast("list[tuple[pos,pos]|pos]", terminal_pairs_remainder), layout, layer0_layout
    )

    layers_updated_aim = [
        [
            (0, 0),
            ((2, 3), (1, 1)),
            ((1, 4), (7, 2)),
            ((6, 5), (1, 0)),
            ((7, 5), (8, 5)),
        ],
        [
            ((0, 0), (6, 2)),
            ((5, 1), (7, 2)),
            ((3, 3), (2, 3)),
            (7, 5),
            ((8, 5), (7, 4)),
        ],
        [((6, 2), (1, 3)), (2, 3), ((2, 0), (7, 4)), ((7, 5), (1, 0))],
        [((6, 2), (7, 1)), (7, 4), ((7, 5), (8, 4))],
        [((3, 4), (8, 4))],
    ]

    assert layers_updated == layers_updated_aim, "Pushing into DAG does not work as anticipated."
