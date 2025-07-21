# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Creates results for circuits with 24 qubits. Run this from the src/mqt/qecc/co3/plots directory."""

from __future__ import annotations

import pickle  # noqa: S403
from pathlib import Path

import evaluation as ev
import layouts as layout

path = "./results/f_vs_time_q24_ratio08_small_row_250718_2_8_f_t_temp"  # add you desired path here

# ROW
factories_q24_row = [(0, 3), (0, 9), (0, 15), (6, 6), (6, 12), (6, 18), (5, 3), (2, 2)]

g, data_qubit_locs, factory_ring = layout.gen_layout("row", 24, factories_q24_row)
custom_layout_q24_row_f8_cc = [data_qubit_locs, g.copy()]

g, data_qubit_locs, factory_ring = layout.gen_layout("row", 24, factories_q24_row[:2])
custom_layout_q24_row_f4_cc = [data_qubit_locs, g.copy()]

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
    # CC GRAPH
    # f=8
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f8_cc,
        "factory_locs": factories_q24_row,
        "layout_type": "custom",
        "layout_name": "row",
        "graphtype": "CC",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f8_cc,
        "factory_locs": factories_q24_row,
        "layout_type": "custom",
        "layout_name": "row",
        "graphtype": "CC",
        "circuit_type": "random",
    },
    # f=4
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f4_cc,
        "factory_locs": factories_q24_row[:2],
        "layout_type": "custom",
        "layout_name": "row",
        "graphtype": "CC",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_row_f4_cc,
        "factory_locs": factories_q24_row[:2],
        "layout_type": "custom",
        "layout_name": "row",
        "graphtype": "CC",
        "circuit_type": "random",
    },
]


reps = 50
both_metric = True  # both metrics heuristic and exact are computed


res_lst = ev.collect_data_space_time(instances, hc_params, reps, path, both_metric)

with Path(path).open("rb") as f:
    res_lst_2 = pickle.load(f) # noqa: S301

path += "_metricrouting"

with Path(path).open("rb") as f:
    res_lst_routing = pickle.load(f)  # noqa: S301

for i, res in enumerate(res_lst_routing):
    layout_type = res["instances"][i]["layout_name"]
    q = res["instances"][i]["q"]
    ratio = res["instances"][i]["ratio"]
    factories = len(res["instances"][i]["factory_locs"])
    num_init_list = res["num_init_lst"]
    num_final_list = res["num_final_lst"]
    improvement_lst = []
    for ni, nf in zip(num_init_list, num_final_list):
        improvement_lst.append((ni - nf) / ni)

for i, res in enumerate(res_lst_2):
    layout_type = res["instances"][i]["layout_name"]
    q = res["instances"][i]["q"]
    ratio = res["instances"][i]["ratio"]
    factories = len(res["instances"][i]["factory_locs"])
    num_init_list = res["num_init_lst"]
    num_final_list = res["num_final_lst"]
    improvement_lst = []
    for ni, nf in zip(num_init_list, num_final_list):
        improvement_lst.append((ni - nf) / ni)
