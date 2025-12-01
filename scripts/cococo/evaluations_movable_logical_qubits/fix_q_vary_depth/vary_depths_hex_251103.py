import cococo.utils_routing as utils
import cococo.layouts as layouts
import cococo.internal_testing as tst
import cococo.circuit_construction as circuit_construction
import cococo.dag_helper as dag_helper
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pickle


from datetime import datetime
import time

import json

# -----params-------

j = 8
q = 120
d = 320
num_gates = 2560
reps = 20


# ------load circuits------
path = f"true_seq_circs_j{j}_q{q}_numgates{num_gates}d{d}_x{reps}.json"
with open(path, "r") as f:
    pairs_lst = json.load(f)
# turn into tuples again
pairs_lst = [[(el[0], el[1]) for el in pairs] for pairs in pairs_lst]

# ------geometry params---------
factories = []
m, n = 4, 5
layout_type = "hex"
g, data_qubit_locs, _ = layouts.gen_layout_scalable(
    layout_type, m, n, factories, remove_edges=False
)  #!important, no edges removed here!!!
layout = {i: j for i, j in enumerate(data_qubit_locs)}

assert q == len(data_qubit_locs), "given q does not coincide with your chosen layout"

sigma = 1
n_circ = 10

gates_list = [320, 640, 1280, 2560]
depths_list = [40, 80, 160, 320]
for num_gates, depth in zip(gates_list, depths_list):
    assert num_gates == depth * j, "depths list and gates_list does not coincide"
gates_str = "_".join(map(str, gates_list))

use_dag = True

date_str = datetime.now().strftime("%Y-%m-%d")
path = f"circuit_depths_m{m}_n{n}_layout{layout_type}_sigma{sigma}_ncirc{n_circ}_num_gates{gates_str}_{date_str}_usedag{use_dag}_p.pkl"


# -----params opt-------

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

testing = True

terminal_pairs_list = []
for pairs in pairs_lst[:n_circ]:
    print("Number of gates", len(pairs))
    terminal_pairs = layouts.translate_layout_circuit(pairs, layout)
    terminal_pairs_list.append(terminal_pairs)

print("terminal pairs lst")
for el in terminal_pairs_list:
    print(el)


# -----------run--------------
results_list_st = (
    []
)  # list of sublists, in each sublist results of each of the num_circs
results_list_opt = []
histories = []
layers_tot = []
# run layer for layer
start = time.time()
for num_gates in gates_list:
    print("num_gates: ", num_gates)
    results_temp_st = []
    results_temp_opt = []
    histories_temp_opt = []
    layers_temp = []
    for circ in terminal_pairs_list:
        temp_circuit = circ[:num_gates]  # cut off the gates

        # standard
        j = 0
        quilt = utils.BasicRouter(
            g,
            data_qubit_locs,
            factories,
            valid_path,
            t,
            metric,
            use_dag=use_dag,
        )  # reinitialize because logical pos changes
        layers = quilt.split_layer_terminal_pairs(terminal_pairs)
        vdp_layers, _ = quilt.find_total_vdp_layers_dyn(
            layers, data_qubit_locs, {}, layout
        )
        results_temp_st.append(vdp_layers.copy())

        # log
        dag = dag_helper.terminal_pairs_into_dag(temp_circuit, layout)
        layers = []
        for layer in range(len(list(dag.layers()))):
            layers.append(dag_helper.extract_layer_from_dag(dag, layout, layer))
        layers_temp.append(layers.copy())

        # opt
        schedule_list_opt_temp = []
        history_temp = []
        for j in range(sigma):
            print(f"-----sigma run ={j}-----")
            quilt = utils.TeleportationRouter(
                g,
                data_qubit_locs,
                factories,
                valid_path,
                t,
                metric,
                use_dag=use_dag,
                seed=j,
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
            schedule_list_opt_temp.append(schedule)
            history_temp.append(history)
        results_temp_opt.append(schedule_list_opt_temp)
        histories_temp_opt.append(history_temp)

    results_list_st.append(results_temp_st)
    results_list_opt.append(results_temp_opt)
    histories.append(histories_temp_opt)
    layers_tot.append(layers_temp)

    save = [results_list_st, results_list_opt, histories, layers_tot]
    with open(path, "wb") as f:
        pickle.dump(save, f)
end = time.time()


# reload
with open(path, "rb") as f:
    saved = pickle.load(f)
[results_list_st, results_list_opt, histories, layers_tot] = saved


# ---------extract data-----------

# compute mean and std of standard approach
depths_st = [[len(el) for el in sublist] for sublist in results_list_st]
depths_mean_st = [np.mean(lst) for lst in depths_st]
depths_std_st = [np.std(lst) for lst in depths_st]
print("depths_mean_st", depths_mean_st)

# logical
depths_log = [[len(el) for el in sublist] for sublist in layers_tot]
depths_mean_log = [np.mean(lst) for lst in depths_log]
depths_std_log = [np.std(lst) for lst in depths_log]

print("depths_mean_log", depths_mean_log)


labels = [f"({g},\n {d})" for g, d in zip(gates_list, depths_list)]


# mean and std of standard approach, choose best results among sigma runs
depths_opt = []
depths_mean_opt = []
depths_std_opt = []
for i, results in enumerate(results_list_opt):
    print(f"-----log. depth {i}------")
    depths_n = []  # the n results from which we take the mean
    for j, run in enumerate(results):
        print(f"------run number {j}-------")
        lengths = [len(schedule) for schedule in run]
        print("depths of sigma opt runs:", lengths)
        min_depth = min(lengths)
        depths_n.append(min_depth)
    depths_opt.append(depths_n)
    print("best depths per run circuit: ", depths_n)
    depths_mean_opt.append(np.mean(depths_n))
    depths_std_opt.append(np.std(depths_n))

# figsize
size = (4, 3)


# plot total abs

plt.figure(figsize=size)
plt.errorbar(
    gates_list,
    depths_mean_opt,
    yerr=depths_std_opt,
    fmt="*-",
    color="lightseagreen",
    capsize=3,
    label="Depth Opt.",
)
plt.errorbar(
    gates_list,
    depths_mean_st,
    yerr=depths_std_st,
    fmt=".-",
    color="darkorchid",
    capsize=3,
    label="Depth St.",
)
plt.errorbar(
    gates_list,
    depths_mean_log,
    yerr=depths_std_log,
    fmt=".--",
    color="yellowgreen",
    capsize=3,
    label="Depth Log.",
)
plt.xlabel("Num. Gates / Log. Depth")
plt.ylabel("Depth of Schedule")
plt.xscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.legend()
plt.tight_layout()

plt.savefig("plot_total_" + path.replace(".pkl", ".pdf"))
plt.clf()


# plot improvement d opt - c / d st -c
improvements_mean = []
improvements_std = []
for lst_opt, lst_st, c_list in zip(depths_opt, depths_st, depths_log):
    temp = []
    for el_opt, el_st, c in zip(lst_opt, lst_st, c_list):
        temp.append((el_opt - c) / (el_st - c))
    print("improvements temp", temp)
    improvements_mean.append(np.mean(temp))
    improvements_std.append(np.std(temp))

plt.figure(figsize=size)
plt.errorbar(gates_list, improvements_mean, yerr=improvements_std, fmt=".-", capsize=3)
plt.ylabel(r"$\frac{d_{opt}-c}{d_{st}-c}$")
plt.xlabel("Num. Gates / Log. Depth")
plt.xscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.tight_layout()
plt.savefig("plot_improvements_total_" + path.replace(".pkl", ".pdf"))
plt.clf()

# plot absolute differences
diff_opt_mean = []
diff_opt_std = []
for lst_opt, lst_st in zip(depths_opt, depths_st):
    differences = [el_st - el_opt for el_st, el_opt in zip(lst_st, lst_opt)]
    print("abs differences", differences)
    diff_opt_mean.append(np.mean(differences))
    diff_opt_std.append(np.std(differences))
plt.figure(figsize=size)
plt.errorbar(gates_list, diff_opt_mean, yerr=diff_opt_std, fmt=".-", capsize=3)
plt.ylabel(r"$\Delta = d_{st} - d_{opt}$")
plt.xlabel("Num. Gates / Log. Depth")
plt.xscale("log")
# plt.xticks(gates_list, gates_list)
plt.xticks(gates_list, labels)
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig("plot_abs_reductions_total_" + path.replace(".pkl", ".pdf"))
