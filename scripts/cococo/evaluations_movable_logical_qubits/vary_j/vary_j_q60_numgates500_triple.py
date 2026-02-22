# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark runs for variation of j."""

import datetime
import json
import pathlib
import pickle  # noqa: S403
import time

import matplotlib.pyplot as plt
import numpy as np

import mqt.qecc.cococo.utils_routing as utils
from mqt.qecc.cococo import circuit_construction, dag_helper, layouts

# ----params-----

factories = []
m, n = 2, 5
layout_type = "triple"
g, data_qubit_locs, _ = layouts.gen_layout_scalable(
    layout_type, m, n, factories, remove_edges=False
)  # !important, no edges removed here!!!
layout = dict(enumerate(data_qubit_locs))

q = len(data_qubit_locs)
print("q = ", len(data_qubit_locs))

n_circ = 10  # go through 5 circuits

j_lst = [2, 3, 4, 5, 10, 15, 20, 25, 30]
j_lst_str = [str(el) for el in j_lst]
j_str = "_".join(map(str, j_lst))

num_gates = 500

use_dag = True

reps = 20

date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
# date_str = "2025-11-08"
path = f"varyj_layouttype{layout_type}_q{q}_m{m}_n{n}_j{j_str}_numgates{num_gates}_dvaries_ncirc{n_circ}_{date_str}_usedag{use_dag}.pkl"

# ------------params opt-----------------
valid_path = "cc"

max_iters = 100
T_start = 100.0
T_end = 0.1
alpha = 0.95
t = 4  # mock value for cnot circuit
radius = 10
k_lookahead = 5
metric = "exact"

steiner_init_type = "full_random"
jump_harvesting = True
stimtest = True

reduce_steiner = True
idle_move_type = "later"
reduce_init_steiner = False


# -------------runs------------

if __name__ == "__main__":
    results_opt = []
    results_st = []

    start = time.time()
    for j in j_lst:
        print(f"=======j={j}, num_gates={num_gates}======")
        # load example circuits and cut off ncirc
        d = int(
            np.ceil(num_gates / j)
        )  # round up because last layer will not be full of j gates but be a layer nevertheless.
        path_circuits = f"true_seq_circs_j{j}_q{q}_numgates{num_gates}d{d}_x{reps}.json"
        try:
            with pathlib.Path(path_circuits).open(encoding="utf-8") as f:
                pairs_lst = json.load(f)
        except FileNotFoundError:
            print("new circs sampled")
            pairs_lst = []
            for _ in range(reps):
                dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates)
                pairs_lst.append(pairs)
            with pathlib.Path(path_circuits).open("w", encoding="utf-8") as f:
                json.dump(pairs_lst, f)
        # turn into tuples again
        pairs_lst = [[(el[0], el[1]) for el in pairs] for pairs in pairs_lst]
        pairs_lst = pairs_lst[:n_circ]

        results_opt_temp = []
        results_st_temp = []

        for i, pairs in enumerate(pairs_lst):
            print(f"-------i={i}--------")
            # into terminal pairs
            terminal_pairs = layouts.translate_layout_circuit(pairs, layout)

            # check logical depth
            dag = dag_helper.terminal_pairs_into_dag(terminal_pairs, layout)
            layers = [dag_helper.extract_layer_from_dag(dag, layout, layer) for layer in range(len(list(dag.layers())))]
            if len(layers) != d:
                msg = f"The number of logical layers {len(layers)} does not coincide with desired depth {d}"
                raise ValueError(msg)

            # run standard
            seed = 0
            quilt = utils.BasicRouter(
                g,
                data_qubit_locs,
                factories,
                valid_path,
                t,
                metric,
                use_dag=use_dag,
            )  # reinitialize because logical pos changes
            split_layers = quilt.split_layer_terminal_pairs(terminal_pairs)
            vdp_layers, _ = quilt.find_total_vdp_layers_dyn(split_layers, data_qubit_locs, {}, layout)
            results_st_temp.append(vdp_layers)

            # run opt
            quilt = utils.TeleportationRouter(
                g,
                data_qubit_locs,
                factories,
                valid_path,
                t,
                metric,
                use_dag=use_dag,
                seed=seed,
            )
            schedule, history = quilt.optimize_layers(
                terminal_pairs,
                layout,
                max_iters,
                T_start,
                T_end,
                alpha,
                radius,
                k_lookahead,
                steiner_init_type,
                jump_harvesting,
                reduce_steiner,
                idle_move_type,
                reduce_init_steiner=reduce_init_steiner,
                stimtest=stimtest,
            )
            results_opt_temp.append(schedule)

        results_opt.append(results_opt_temp)
        results_st.append(results_st_temp)
        save = [results_opt, results_st]
        with pathlib.Path(path).open("wb") as f:
            pickle.dump(save, f)

    end = time.time()
    print(f"Simulation took {(end - start) / 60} minutes")

    print(path)
    # reload
    with pathlib.Path(path).open("rb") as f:
        saved = pickle.load(f)  # noqa: S301
    results_opt, results_st = saved

    # ---------extract data-----------

    # compute mean and std of standard approach
    depths_st = [[len(el) for el in sublist] for sublist in results_st]
    depths_mean_st = [np.mean(lst) for lst in depths_st]
    depths_std_st = [np.std(lst) for lst in depths_st]
    print("depths_mean_st", depths_mean_st)

    # opt
    depths_opt = [[len(el) for el in sublist] for sublist in results_opt]
    depths_mean_opt = [np.mean(lst) for lst in depths_opt]
    depths_std_opt = [np.std(lst) for lst in depths_opt]

    print("depths_mean_opt", depths_mean_opt)

    deltas_lst = [
        [st - opt for st, opt in zip(sublist_st, sublist_opt, strict=False)]
        for sublist_st, sublist_opt in zip(depths_st, depths_opt, strict=False)
    ]
    deltas_mean = [np.mean(lst) for lst in deltas_lst]
    deltas_std = [np.std(lst) for lst in deltas_lst]

    print("deltas mean", deltas_mean)

    # ---------plot------------
    size = (3.5, 2.5)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 10

    plt.figure(figsize=size)
    plt.errorbar(
        j_lst,
        depths_mean_opt,
        yerr=depths_std_opt,
        fmt="*-",
        color="lightseagreen",
        capsize=3,
        label=r"$\tilde{d}_{\mathrm{opt}}$",
    )
    plt.errorbar(
        j_lst,
        depths_mean_st,
        yerr=depths_std_st,
        fmt=".-",
        color="darkorchid",
        capsize=3,
        label=r"$\tilde{d}_{\mathrm{st}}$",
    )

    plt.xlabel("$g$")
    plt.ylabel(rf"$\tilde{d}$")

    # plt.xscale("log", base=2)

    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xticks(j_lst, j_lst_str)
    plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "plot_total_" + path.replace(".pkl", ".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.clf()

    # -------------

    plt.figure(figsize=size)
    plt.errorbar(j_lst, deltas_mean, yerr=deltas_std, fmt=".-", capsize=3)

    # plt.xlabel("Num. gates per logical layer $j$")
    plt.xlabel("$g$")
    plt.ylabel(r"$\Delta = \tilde{d}_{st}-\tilde{d}_{opt}$")

    # plt.xscale("log", base=2)

    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xticks(j_lst, j_lst_str)
    # plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
    # plt.legend()
    plt.tight_layout()

    plt.savefig(
        "plot_absdelta_" + path.replace(".pkl", ".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.clf()
