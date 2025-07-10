# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generates data for space time comparison."""

from __future__ import annotations

import pickle  # noqa: S403
from pathlib import Path

import evaluation as ev
import layouts as layout

path = "./results/space_vs_time_q24"
# --------- HEX --------------

# q=24
factories_q24_hex = [(0, 3), (1, 8), (2, 13), (7, 3), (8, 8), (9, 13), (4, 2), (5, 14)]
g, data_qubit_locs, factory_ring = layout.gen_layout("hex", 24, factories_q24_hex)
custom_layout_q24_hex = [data_qubit_locs, g]

# q=42
factories_q42_hex = [(0, 3), (1, 8), (2, 13), (12, 12), (3, 18), (11, 9), (10, 4), (7, 3)]
g, data_qubit_locs, factory_ring = layout.gen_layout("hex", 42, factories_q42_hex)
custom_layout_q42_hex = [data_qubit_locs, g]

# q=60
factories_q60_hex = [(0, 3), (1, 8), (2, 13), (12, 12), (3, 18), (11, 9), (10, 4), (13, 17)]
g, data_qubit_locs, factory_ring = layout.gen_layout("hex", 60, factories_q60_hex)
custom_layout_q60_hex = [data_qubit_locs, g]

# ------------- ROW -------------

# q=24
factories_q24_row = [(0, 3), (0, 9), (0, 15), (6, 6), (6, 12), (6, 18), (5, 3), (2, 2)]
g, data_qubit_locs, factory_ring = layout.gen_layout("row", 24, factories_q24_row)
custom_layout_q24_row = [data_qubit_locs, g]

# q=42 #same factories
factories_q42_row = [(0, 3), (0, 9), (0, 15), (6, 6), (6, 12), (6, 18), (5, 3), (2, 2)]
g, data_qubit_locs, factory_ring = layout.gen_layout("row", 42, factories_q42_row)
custom_layout_q42_row = [data_qubit_locs, g]

# q=60 # same factories
factories_q60_row = [(0, 3), (0, 9), (0, 15), (6, 6), (6, 12), (6, 18), (5, 3), (2, 2)]
g, data_qubit_locs, factory_ring = layout.gen_layout("row", 60, factories_q60_row)
custom_layout_q60_row = [data_qubit_locs, g]

# ----------- PAIR ---------------

# q=24
factories_q24_pair = [(0, 5), (0, 11), (0, 17), (7, 3), (8, 8), (8, 14), (8, 20), (2, 2)]
g, data_qubit_locs, factory_ring = layout.gen_layout("pair", 24, factories_q24_pair)
custom_layout_q24_pair = [data_qubit_locs, g]

# q=42 #duplicate factories
factories_q42_pair = [(0, 5), (0, 11), (0, 17), (7, 3), (8, 8), (8, 14), (8, 20), (2, 2)]
g, data_qubit_locs, factory_ring = layout.gen_layout("pair", 42, factories_q42_pair)
custom_layout_q42_pair = [data_qubit_locs, g]

# q=60 #duplicate factories
factories_q60_pair = [(0, 5), (0, 11), (0, 17), (7, 3), (8, 8), (8, 14), (8, 20), (2, 2)]
g, data_qubit_locs, factory_ring = layout.gen_layout("pair", 60, factories_q60_pair)
custom_layout_q60_pair = [data_qubit_locs, g]


# --------- sparse -------------

# q=24
factories_q24_sparse = [(0, 5), (0, 15), (0, 21), (5, 30), (8, 29), (9, 21), (8, 11), (6, 1)]
g, data_qubit_locs, factory_ring = layout.gen_layout("sparse", 24, factories_q24_sparse)
custom_layout_q24_sparse = [data_qubit_locs, g]


hc_params = {
    "metric": "crossing",
    "max_restarts": 8,
    "max_iterations": 50,
    "routing": "dynamic",
    "optimize_factories": False,  # True, #want to count in factory crossings as well
    "free_rows": None,
    "parallel": True,
    "processes": 8,
}

instances = [
    # be aware of order such that same params not multiple samples
    # q=24
    # ratio = 0.9
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.9,
        "custom_layout": custom_layout_q24_hex,
        "factory_locs": factories_q24_hex,
        "layout_type": "custom",
        "layout_name": "hex",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.9,
        "custom_layout": custom_layout_q24_row,
        "factory_locs": factories_q24_row,
        "layout_type": "custom",
        "layout_name": "row",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.9,
        "custom_layout": custom_layout_q24_pair,
        "factory_locs": factories_q24_pair,
        "layout_type": "custom",
        "layout_name": "pair",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    # {"q": 24, "t": 8, "min_depth": 24*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q24_sparse, "factory_locs" : factories_q24_sparse, "layout_type" : "custom", "layout_name": "sparse", "circuit_type": "random", "graphtype": "CC"},
    # ratio = 0.7
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.7,
        "custom_layout": custom_layout_q24_hex,
        "factory_locs": factories_q24_hex,
        "layout_type": "custom",
        "layout_name": "hex",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.7,
        "custom_layout": custom_layout_q24_row,
        "factory_locs": factories_q24_row,
        "layout_type": "custom",
        "layout_name": "row",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.7,
        "custom_layout": custom_layout_q24_pair,
        "factory_locs": factories_q24_pair,
        "layout_type": "custom",
        "layout_name": "pair",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    # {"q": 24, "t": 8, "min_depth": 24*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q24_sparse, "factory_locs" : factories_q24_sparse, "layout_type" : "custom", "layout_name": "sparse", "circuit_type": "random", "graphtype": "CC"},
    # ratio = 0.5
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.5,
        "custom_layout": custom_layout_q24_hex,
        "factory_locs": factories_q24_hex,
        "layout_type": "custom",
        "layout_name": "hex",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.5,
        "custom_layout": custom_layout_q24_row,
        "factory_locs": factories_q24_row,
        "layout_type": "custom",
        "layout_name": "row",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.5,
        "custom_layout": custom_layout_q24_pair,
        "factory_locs": factories_q24_pair,
        "layout_type": "custom",
        "layout_name": "pair",
        "circuit_type": "random",
        "graphtype": "CC",
    },
    # {"q": 24, "t": 8, "min_depth": 24*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q24_sparse, "factory_locs" : factories_q24_sparse, "layout_type" : "custom", "layout_name": "sparse", "circuit_type": "random", "graphtype": "CC"},
    # q=42
    # ratio = 0.9
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q42_hex, "factory_locs" : factories_q42_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q42_row, "factory_locs" : factories_q42_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q42_pair, "factory_locs" : factories_q42_pair, "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # ratio = 0.7
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q42_hex, "factory_locs" : factories_q42_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q42_row, "factory_locs" : factories_q42_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q42_pair, "factory_locs" : factories_q42_pair, "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # ratio = 0.5
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q42_hex, "factory_locs" : factories_q42_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q42_row, "factory_locs" : factories_q42_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 42, "t": 8, "min_depth": 42*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q42_pair, "factory_locs" : factories_q42_pair, "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # q=60
    # ratio = 0.9
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q60_hex, "factory_locs" : factories_q60_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q60_row, "factory_locs" : factories_q60_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.9, "custom_layout": custom_layout_q60_pair, "factory_locs" : factories_q60_pair, "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # ratio = 0.7
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q60_hex, "factory_locs" : factories_q60_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q60_row, "factory_locs" : factories_q60_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.7, "custom_layout": custom_layout_q60_pair, "factory_locs" : factories_q60_pair, "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
    # ratio = 0.5
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q60_hex, "factory_locs" : factories_q60_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q60_row, "factory_locs" : factories_q60_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random", "graphtype": "CC"},
    # {"q": 60, "t": 8, "min_depth": 60*4, "tgate": True, "ratio": 0.5, "custom_layout": custom_layout_q60_pair, "factory_locs" : factories_q60_pair, "layout_type" : "custom", "layout_name": "pair", "circuit_type": "random", "graphtype": "CC"},
]

reps = 10
both_metric = False

res_lst = ev.collect_data_space_time(instances, hc_params, reps, path, both_metric)

with Path(path).open("rb") as f:
    res_lst = pickle.load(f)  # noqa: S301
