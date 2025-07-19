# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Constructs random circuits of different types."""

from __future__ import annotations

import evaluation as ev
import layouts as layout
import matplotlib.pyplot as plt

# This notebook must be run from the directory /mqt-qecc/src/mqt/qecc/co3/plots
import mqt.qecc.co3 as co

plt.rcParams["font.family"] = "Times New Roman"


path = "./results/circuit_types_q24_250718_d"

# HEX

g, data_qubit_locs, factory_ring = layout.gen_layout("hex", 24, [])
custom_layout_q24_hex_f8 = [data_qubit_locs, g]

#g, data_qubit_locs, factory_ring = co.plots.gen_layout("hex", 42, [])
#custom_layout_q42_hex_f8 = [data_qubit_locs, g]

#g, data_qubit_locs, factory_ring = co.plots.gen_layout("hex", 60, [])
#custom_layout_q60_hex_f8 = [data_qubit_locs, g]

# ROW
g, data_qubit_locs, factory_ring = layout.gen_layout("row", 24, [])
custom_layout_q24_row_f8 = [data_qubit_locs, g]

#g, data_qubit_locs, factory_ring = co.plots.gen_layout("row", 42, [])
#custom_layout_q42_row_f8 = [data_qubit_locs, g]

#g, data_qubit_locs, factory_ring = co.plots.gen_layout("row", 60, [])
#custom_layout_q60_row_f8 = [data_qubit_locs, g]


# PAIR
g, data_qubit_locs, factory_ring = layout.gen_layout("pair", 24, [])
custom_layout_q24_pair_f8 = [data_qubit_locs, g]

#g, data_qubit_locs, factory_ring = co.plots.gen_layout("pair", 42, [])
#custom_layout_q42_pair_f8 = [data_qubit_locs, g]

#g, data_qubit_locs, factory_ring = co.plots.gen_layout("pair", 60, [])
#custom_layout_q60_pair_f8 = [data_qubit_locs, g]

# -----------------------------

hc_params = {
    "metric": "crossing",
    "max_restarts": 10,
    "max_iterations": 50,
    "routing": "dynamic",
    "optimize_factories": False,
    "free_rows": None,
    "parallel": True,
    "processes": 8,
}

instances = [
    # q=24, depth 2q
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_pair_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "hex", "circuit_type": "parallelmax", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "row", "circuit_type": "parallelmax", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_pair_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "pair", "circuit_type": "parallelmax", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "hex", "circuit_type": "sequential", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "row", "circuit_type": "sequential", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*2, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_pair_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "pair", "circuit_type": "sequential", "graphtype": "CC"},
    # q=24, depth 4q
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_hex_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "hex",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_row_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "row",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_pair_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "pair",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_hex_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "hex",
        "circuit_type": "parallelmax",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_row_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "row",
        "circuit_type": "parallelmax",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_pair_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "pair",
        "circuit_type": "parallelmax",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_hex_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "hex",
        "circuit_type": "sequential",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_row_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "row",
        "circuit_type": "sequential",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 4,
        "min_depth": 24 * 4,
        "tgate": False,
        "ratio": 1.0,
        "custom_layout": custom_layout_q24_pair_f8,
        "factory_locs": [],
        "layout_type": "custom",
        "layout_name": "pair",
        "circuit_type": "sequential",
        "graphtype": "CC",
    },
    # q=24, depth 8q
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_pair_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "hex", "circuit_type": "parallelmax", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "row", "circuit_type": "parallelmax", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_pair_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "pair", "circuit_type": "parallelmax", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "hex", "circuit_type": "sequential", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "row", "circuit_type": "sequential", "graphtype": "CC"},
    # {"q": 24, "t": 4, "min_depth": 24*8, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_pair_f8, "factory_locs" : [], "layout_type" : "custom", "layout_name": "pair", "circuit_type": "sequential", "graphtype": "CC"},
]


reps = 50
both_metric = False
res_lst = ev.collect_data_space_time(instances, hc_params, reps, path, both_metric)
