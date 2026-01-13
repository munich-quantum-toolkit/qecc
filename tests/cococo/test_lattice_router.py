# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the Routing."""

from __future__ import annotations

from typing import cast

import mqt.qecc.cococo.utils_routing as utils
from mqt.qecc.cococo import circuit_construction, layouts

pos = tuple[int, int]


def test_basicrouter():
    """Test the BasicRouter class. By running some instance with testing==True."""
    layout_type = "triple"
    m = 2
    n = 2
    factories: list[pos] = [(22, 2), (5, -2), (10, -2), (9, 4), (17, -2)]
    remove_edges = False
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    layout = dict(enumerate(data_qubit_locs))
    t = 4

    q = len(data_qubit_locs)
    # j = 8
    num_gates = q * 2
    pairs = circuit_construction.generate_random_circuit(q, num_gates, tgate=True, ratio=0.8)  # circuit with t gates

    terminal_pairs = layouts.translate_layout_circuit(
        pairs, cast("dict[int|str,pos|list[pos]]", layout)
    )  # let's stick to the simple layout

    router = utils.BasicRouter(
        g,
        data_qubit_locs,
        factories,
        valid_path="cc",
        t=t,
        metric="exact",
        use_dag=True,
    )
    layers = router.split_layer_terminal_pairs(terminal_pairs)
    try:
        _vdp_layers, _ = router.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, router.factory_times, layout, testing=True
        )
    except Exception as exc:
        msg = "Something is wrong with the BasicRouter."
        raise ValueError(msg) from exc


def test_basicrouter_2():
    """Test the BasicRouter class without dag structure but naive layering. By running some instance with testing==True."""
    layout_type = "triple"
    m = 2
    n = 2
    factories: list[pos] = [(22, 2), (5, -2), (10, -2), (9, 4), (17, -2)]
    remove_edges = False
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    layout = dict(enumerate(data_qubit_locs))
    t = 4

    q = len(data_qubit_locs)
    # j = 8
    num_gates = q * 2
    pairs = circuit_construction.generate_random_circuit(q, num_gates, tgate=True, ratio=0.8)  # circuit with t gates

    terminal_pairs = layouts.translate_layout_circuit(
        pairs, cast("dict[int|str,pos|list[pos]]", layout)
    )  # let's stick to the simple layout

    router = utils.BasicRouter(
        g,
        data_qubit_locs,
        factories,
        valid_path="cc",
        t=t,
        metric="exact",
        use_dag=False,
    )
    layers = router.split_layer_terminal_pairs(terminal_pairs)
    try:
        _vdp_layers, _ = router.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, router.factory_times, layout, testing=True
        )
    except Exception as e:
        msg = "Something is wrong with the BasicRouter."
        raise ValueError(msg) from e


def test_basicrouter_3_sc_validpath():
    """Even though not used here, the SC variant of the valid path is tested."""
    layout_type = "triple"
    m = 2
    n = 2
    factories: list[pos] = [(22, 2), (5, -2), (10, -2), (9, 4), (17, -2)]
    remove_edges = False
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    layout = dict(enumerate(data_qubit_locs))
    t = 4

    q = len(data_qubit_locs)
    num_gates = q * 2
    pairs = circuit_construction.generate_random_circuit(q, num_gates, tgate=False, ratio=1.0)  # circuit with t gates

    terminal_pairs = layouts.translate_layout_circuit(
        pairs, cast("dict[int|str,pos|list[pos]]", layout)
    )  # let's stick to the simple layout

    router = utils.BasicRouter(
        g,
        data_qubit_locs,
        factories,
        valid_path="sc",
        t=t,
        metric="exact",
        use_dag=True,
    )
    layers = router.split_layer_terminal_pairs(terminal_pairs)
    try:
        _vdp_layers, _ = router.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, router.factory_times, layout, testing=True
        )
    except Exception as e:
        msg = "Something is wrong with the BasicRouter."
        raise ValueError(msg) from e


def test_teleportationrouter():
    """Test the TeleportationRouter class. By running some instance with testing==True."""
    layout_type = "triple"
    m = 2
    n = 2
    factories: list[pos] = [(22, 2), (5, -2), (10, -2), (9, 4), (17, -2)]
    remove_edges = False
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    layout = dict(enumerate(data_qubit_locs))
    t = 4

    q = len(data_qubit_locs)
    # j = 8
    num_gates = int(q * 1.2)
    # _dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
    #    j,
    #    q,
    #    num_gates,
    # )
    pairs = circuit_construction.generate_random_circuit(q, num_gates, tgate=True, ratio=0.8)  # circuit with t gates

    terminal_pairs = layouts.translate_layout_circuit(
        pairs, cast("dict[int|str, pos|list[pos]]", layout)
    )  # let's stick to the simple layout

    router = utils.TeleportationRouter(
        g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag=True, seed=1
    )

    max_iters = 100
    T_start = 100.0  # noqa: N806
    T_end = 0.1  # noqa: N806
    alpha = 0.95
    t = 4  # mock value for cnot circuit
    radius = 10
    k_lookahead = 5

    steiner_init_type = "full_random"
    jump_harvesting = True

    reduce_steiner = True
    idle_move_type = "later"

    try:
        _schedule, _ = router.optimize_layers(
            terminal_pairs,
            layout,
            max_iters,
            T_start,
            T_end,
            alpha,
            radius=radius,
            k_lookahead=k_lookahead,
            steiner_init_type=steiner_init_type,
            jump_harvesting=jump_harvesting,
            reduce_steiner=reduce_steiner,
            idle_move_type=idle_move_type,
            reduce_init_steiner=False,
            stimtest=True,
        )
    except Exception as e:
        msg = "Something is wrong with the TeleportationRouter class."
        raise ValueError(msg) from e

    # also run with reduce_init_steiner=True and other steiner init type --> increase coverage
    router = utils.TeleportationRouter(
        g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag=True, seed=1
    )
    try:
        _schedule, _ = router.optimize_layers(
            terminal_pairs,
            layout,
            max_iters,
            T_start,
            T_end,
            alpha,
            radius=radius,
            k_lookahead=k_lookahead,
            steiner_init_type="on_path_random",
            jump_harvesting=jump_harvesting,
            reduce_steiner=reduce_steiner,
            idle_move_type=idle_move_type,
            reduce_init_steiner=True,
            stimtest=True,
        )
    except Exception as e:
        msg = "Something is wrong with the TeleportationRouter class (on path_random, reduce init steiner true)."
        raise ValueError(msg) from e


def test_count_crossings():
    """Count the crossings in the BasicRouter class."""
    layout_type = "triple"
    m = 3
    n = 4
    factories: list[pos] = []
    remove_edges = False
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)

    # adapt data_qubit_locs such that we provocate a locking
    data_qubit_locs.append((8, 5))
    data_qubit_locs.append((2, 3))

    t = 4

    len(data_qubit_locs)

    terminal_pairs: list[pos | tuple[pos, pos]] = [((8, 4), (5, 0)), ((8, 4), (36, 0)), ((2, 2), (13, 4))]

    router = utils.BasicRouter(
        g,
        data_qubit_locs,
        factories,
        valid_path="cc",
        t=t,
        metric="exact",
        use_dag=True,
    )
    layers = router.split_layer_terminal_pairs(terminal_pairs)
    num_crossings = router.count_crossings(cast("list[list[tuple[pos,pos]]]", layers), data_qubit_locs)
    assert num_crossings == 600, "count_crossings does not work as expected."

    terminal_pairs = [((0, 0), (8, 5)), ((2, 4), (7, 0))]
    layers = router.split_layer_terminal_pairs(terminal_pairs)

    crossings_lst = router.count_crossings_per_layer(layers, t_crossings=False)
    assert sum(crossings_lst) == 2, "count_crossings_per_layer is wrong."

    # with t
    factories = [(-1, -2)]
    g, data_qubit_locs, _factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    terminal_pairs = [((0, 0), (8, 5)), ((2, 4), (7, 0)), (2, 2)]
    layers = router.split_layer_terminal_pairs(terminal_pairs)
    router = utils.BasicRouter(
        g,
        data_qubit_locs,
        factories,
        valid_path="cc",
        t=t,
        metric="exact",
        use_dag=True,
    )
    crossings_lst = router.count_crossings_per_layer(layers, t_crossings=True)
    assert sum(crossings_lst) == 4, "count_crossings_per_layer is wrong."
