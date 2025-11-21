# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the Routing."""

from __future__ import annotations

import mqt.qecc.cococo.utils_routing as utils
from mqt.qecc.cococo import circuit_construction, layouts


def test_basicrouter():
    """Test the BasicRouter class. By running some instance with testing==True."""

    layout_type = "triple"
    m = 4
    n = 4
    factories = []
    remove_edges = False
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
        layout_type, m, n, factories, remove_edges
    )
    layout = {i: j for i, j in enumerate(data_qubit_locs)}
    t = 2  # mock because we have only cnots here in the example

    q = len(data_qubit_locs)
    j = 8
    num_gates = q * 2
    dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
        j,
        q,
        num_gates,
    )

    terminal_pairs = layouts.translate_layout_circuit(
        pairs, layout
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
        vdp_layers, _ = router.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, router.factory_times, layout, testing=True
        )
    except:
        raise ValueError("Something is wrong with the BasicRouter.")


def test_TeleportationRouter():
    """Test the TeleportationRouter class. By running some instance with testing==True."""
    layout_type = "triple"
    m = 2
    n = 2
    factories = []
    remove_edges = False
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
        layout_type, m, n, factories, remove_edges
    )
    layout = {i: j for i, j in enumerate(data_qubit_locs)}
    t = 2  # mock because we have only cnots here in the example

    q = len(data_qubit_locs)
    j = 8
    num_gates = q * 2
    dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
        j,
        q,
        num_gates,
    )

    terminal_pairs = layouts.translate_layout_circuit(
        pairs, layout
    )  # let's stick to the simple layout

    router = utils.TeleportationRouter(
        g,
        data_qubit_locs,
        factories,
        valid_path="cc",
        t=t,
        metric="exact",
        use_dag=True,
        seed=1
    )

    max_iters = 100
    T_start = 100.0
    T_end = 0.1
    alpha = 0.95
    t = 4  # mock value for cnot circuit
    radius = 10
    k_lookahead = 5

    steiner_init_type = "full_random"
    jump_harvesting = True

    reduce_steiner = True
    idle_move_type = "later"

    try:
        schedule, _ = router.optimize_layers(
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
    except:
        raise ValueError("Something is wrong with the TeleportationRouter class.")
