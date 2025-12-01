"""Reproduce figures in the gist of Fig. 10 in 2504.10591"""

from __future__ import annotations

import evaluation as ev
import cococo.layouts as layouts
import matplotlib.pyplot as plt

# This notebook must be run from the directory /mqt-qecc/src/mqt/qecc/co3/plots

plt.rcParams["font.family"] = "Times New Roman"


path = "./test_results_fig10"

# HEX

g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
    "hex", 2, 2, [], remove_edges=False
)
custom_layout_q24_hex_f8 = [data_qubit_locs, g]


# ROW
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
    "hex", 6, 4, [], remove_edges=False
)
custom_layout_q24_row_f8 = [data_qubit_locs, g]


# PAIR
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
    "hex", 3, 4, [], remove_edges=False
)
custom_layout_q24_pair_f8 = [data_qubit_locs, g]


# -----------------------------

hc_params = {
    "max_restarts": 10,
    "max_iterations": 50,
    "metric": "crossing",
    "use_dag": False,  # because reproduce old stuff
    "valid_path": "cc",
    "optimize_factories": False,
    "parallel": True,
    "processes": 8,
}

instances = [
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_pair_f8,
        "factory_locs": [],
        "layout_name": "pair",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f8,
        "factory_locs": [],
        "layout_name": "row",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f8,
        "factory_locs": [],
        "layout_name": "hex",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_pair_f8,
        "factory_locs": [],
        "layout_name": "pair",
        "circuit_type": "parallelmax",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f8,
        "factory_locs": [],
        "layout_name": "row",
        "circuit_type": "parallelmax",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f8,
        "factory_locs": [],
        "layout_name": "hex",
        "circuit_type": "parallelmax",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_pair_f8,
        "factory_locs": [],
        "layout_name": "pair",
        "circuit_type": "sequential",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f8,
        "factory_locs": [],
        "layout_name": "row",
        "circuit_type": "sequential",
    },
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f8,
        "factory_locs": [],
        "layout_name": "hex",
        "circuit_type": "sequential",
    },
]


reps = 50
both_metric = False
res_lst = ev.collect_data_space_time(instances, hc_params, reps, path, both_metric)
