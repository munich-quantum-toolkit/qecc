"""Creates results for circuits with 24 qubits. Run this from the src/mqt/qecc/co3/plots directory."""

from __future__ import annotations

import pickle  # noqa: S403
from pathlib import Path

import evaluation as ev
import cococo.layouts as layouts

path = "./test_results"  # add you desired path here

factories_q24_hex = [
    (5, -1),
    (10, 0),
    (7, 7),
    (12, 5),
]

layout_type = "hex"
m, n = 2, 2

g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
    layout_type, m, n, factories_q24_hex, remove_edges=False
)
custom_layout_q24_hex_f4_cc = [data_qubit_locs, g.copy()]

g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(
    layout_type, m, n, factories_q24_hex[:2], remove_edges=False
)
custom_layout_q24_hex_f2_cc = [data_qubit_locs, g.copy()]

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
    # CC GRAPH
    # f=8
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f4_cc,
        "factory_locs": factories_q24_hex,
        "layout_name": "hex",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f4_cc,
        "factory_locs": factories_q24_hex,
        "layout_name": "hex",
        "circuit_type": "random",
    },
    # f=4
    {
        "q": 24,
        "t": 2,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f2_cc,
        "factory_locs": factories_q24_hex[:2],
        "layout_name": "hex",
        "circuit_type": "random",
    },
    {
        "q": 24,
        "t": 8,
        "min_depth": 24 * 4,
        "tgate": True,
        "ratio": 0.8,
        "custom_layout": custom_layout_q24_hex_f4_cc,
        "factory_locs": factories_q24_hex[:2],
        "layout_name": "hex",
        "circuit_type": "random",
    },
]


reps = 2
both_metric = True  # both metrics heuristic and exact are computed


res_lst = ev.collect_data_space_time(instances, hc_params, reps, path, both_metric)

with Path(path).open("rb") as f:
    res_lst_2 = pickle.load(f)  # noqa: S301

path += "_metricrouting"

with Path(path).open("rb") as f:
    res_lst_routing = pickle.load(f)  # noqa: S301

for i, res in enumerate(res_lst_routing):
    layout_type = res["instances"][i]["layout_name"]
    q = res["instances"][i]["q"]
    ratio = res["instances"][i]["ratio"]
    t = res["instances"][i]["t"]
    factories = len(res["instances"][i]["factory_locs"])
    num_init_list = res["num_init_lst"]
    num_final_list = res["num_final_lst"]
    improvement_lst = []
    for ni, nf in zip(num_init_list, num_final_list):
        improvement_lst.append((ni - nf) / ni)
    print("layout type", layout_type)
    print("factories", factories)
    print("t", t)
    print("imrprovements", improvement_lst)

for i, res in enumerate(res_lst_2):
    layout_type = res["instances"][i]["layout_name"]
    q = res["instances"][i]["q"]
    ratio = res["instances"][i]["ratio"]
    t = res["instances"][i]["t"]
    factories = len(res["instances"][i]["factory_locs"])
    num_init_list = res["num_init_lst"]
    num_final_list = res["num_final_lst"]
    improvement_lst = []
    for ni, nf in zip(num_init_list, num_final_list):
        improvement_lst.append((ni - nf) / ni)
    print("layout type", layout_type)
    print("factories", factories)
    print("t", t)
    print("imrprovements", improvement_lst)
