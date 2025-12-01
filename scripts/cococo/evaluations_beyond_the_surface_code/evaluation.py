"""Evaluate the routing for different Layouts."""

from __future__ import annotations

import itertools
import logging
import pickle  # noqa: S403
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import cococo.circuit_construction as circuit_construction
import cococo.hill_climber as hill_climber
import cococo.layouts as layouts
import cococo.utils_routing as utils


if TYPE_CHECKING:
    from numpy.typing import NDArray


def collect_data_space_time(
    instances: list[dict[str, Any]],
    hc_params: dict[str, Any],
    reps: int,
    path: str,
    both_metric: bool = False,
) -> list[dict[str, Any]]:
    """
    Collects the data for a run which will compare space and time cost.

    Args:
        instances (list[dict]): A list of dicts which collects parameters of instances to be run.
            Each dict must contain these keys: q, t, ratio, min_depth,... also "layout_name" must be added to track the shape, since "layout_type" will be "custom".
            for generating random circuits.
        hc_params (dict): contains a value for metric, max_restarts, max_iterations, routing, optimize_factories, free_rows, parallel
        reps (int): Number of random circuits per instance (each optimized with hc and routed)
        path (str): where to store the res_lst (also intermediate save points)
        both_metric (bool): if False, just what metric is defined in hc_params. if True, both the crossing and the routing metric are used
            this is necessary as running collect_data_space_time two separate times would use different sampled circuits for both runs.#

    Returns:
        list[dict]: results dictionary for each instance (same order as instances.)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    instances_set = {
        "q",
        "t",
        "min_depth",
        "tgate",
        "ratio",
        "custom_layout",
        "factory_locs",
        "layout_name",
        "circuit_type",
    }
    for instance in instances:
        assert set(instance.keys()) == instances_set, "Wrong input for `instances`."
    # if no "circuit type" given, choose standard
    for instance in instances:
        if "circuit_type" not in set(instance.keys()):
            instance.update({"circuit_type": "random"})
    hc_par_set = {
        "max_restarts",
        "max_iterations",
        "metric",
        "use_dag",
        "valid_path",
        "optimize_factories",
        "parallel",
        "processes",
    }
    assert set(hc_params.keys()) == hc_par_set, "Wrong input for `hc_params`."
    res_lst = []

    # extract values from hcparams
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]
    metric = hc_params["metric"]
    use_dag = hc_params["use_dag"]
    valid_path = hc_params["valid_path"]
    optimize_factories = hc_params["optimize_factories"]
    parallel = hc_params["parallel"]
    processes = hc_params["processes"]

    if both_metric:
        # both metric can only be done if hc_params has "crossing in it"
        assert (
            hc_params["metric"] == "crossing"
        ), "For both_metric == True you need to choose crossing in hc_params, to ensure that we have in the end both crossing and routing metric"
        res_lst_routing = []

    # sample circuits (should be the same for those instances for which q, min_depth, tgate, ratio) coincide
    # thus initialize with first instance's value and only change them if those values differ for the new instance
    circuits = []
    for _ in range(reps):
        if instances[0]["circuit_type"] == "random":
            circuit = circuit_construction.generate_random_circuit(
                instances[0]["q"],
                instances[0]["min_depth"],
                instances[0]["tgate"],
                instances[0]["ratio"],
            )
        elif instances[0]["circuit_type"] == "parallelmax":
            assert (
                instances[0]["tgate"] is False
            ), "For Maximally parallel circuit type, we can only do CNOTs."
            assert (
                instances[0]["ratio"] == 1.0
            ), "For Maximally parallel circuit type, the ratio must be 1.0 as we can do only CNOTS"
            circuit = circuit_construction.generate_max_parallel_circuit(
                q=instances[0]["q"], min_depth=instances[0]["min_depth"]
            )
        elif instances[0]["circuit_type"] == "sequential":
            assert (
                instances[0]["tgate"] is False
            ), "For seq. circuit type, we can only do CNOTs."
            assert (
                instances[0]["ratio"] == 1.0
            ), "For seq. circuit type, the ratio must be 1.0 as we can do only CNOTS"
            layer_size = 2
            circuit = circuit_construction.generate_min_parallel_circuit(
                q=instances[0]["q"],
                min_depth=instances[0]["min_depth"],
                layer_size=layer_size,
            )
        else:
            msg = "No other circuit types than random, sequential, parallelmax"
            raise NotImplementedError(msg)
        circuits.append(circuit)
        for _circ in circuits:
            pass

    for l, instance in enumerate(instances):  # noqa: E741
        logger = logging.getLogger(__name__)

        # check whether new values for q, min_depth, tgate, ratio and circuit_type. If yes sample new circuits, otherwise keep them
        if l != 0:
            if (
                instance["q"] == instances[l - 1]["q"]
                and instance["min_depth"] == instances[l - 1]["min_depth"]
                and instance["tgate"] == instances[l - 1]["tgate"]
                and instance["ratio"] == instances[l - 1]["ratio"]
                and instance["circuit_type"] == instances[l - 1]["circuit_type"]
            ):
                logger.info("previous circuits kept")
            else:
                circuits = []
                for _ in range(reps):
                    if instance["circuit_type"] == "random":
                        circuit = circuit_construction.generate_random_circuit(
                            instance["q"],
                            instance["min_depth"],
                            instance["tgate"],
                            instance["ratio"],
                        )
                    elif instance["circuit_type"] == "parallelmax":
                        assert (
                            instance["tgate"] is False
                        ), "For Maximally parallel circuit type, we can only do CNOTs."
                        assert (
                            instance["ratio"] == 1.0
                        ), "For Maximally parallel circuit type, the ratio must be 1.0 as we can do only CNOTS"
                        circuit = circuit_construction.generate_max_parallel_circuit(
                            instance["q"], instance["min_depth"]
                        )
                    elif instance["circuit_type"] == "sequential":
                        assert (
                            instance["tgate"] is False
                        ), "For seq. circuit type, we can only do CNOTs."
                        assert (
                            instance["ratio"] == 1.0
                        ), "For seq. circuit type, the ratio must be 1.0 as we can do only CNOTS"
                        layer_size = 2
                        circuit = circuit_construction.generate_min_parallel_circuit(
                            q=instance["q"],
                            min_depth=instance["min_depth"],
                            layer_size=layer_size,
                        )
                    else:
                        msg = "No other circuit types than random, sequential, parallelmax"
                        raise NotImplementedError(msg)
                    circuits.append(circuit)
                logger.info("new circuits sampled")
            for _circ in circuits:
                pass

        logger.info(f"=======Instance {l}=======")
        time = []
        time2 = []

        # extract variables from instances
        q = instance["q"]
        t = instance["t"]
        min_depth = instance["min_depth"]
        tgate = instance["tgate"]
        ratio = instance["ratio"]
        custom_layout = instance["custom_layout"]
        g = custom_layout[1]
        space = len(list(g.nodes()))
        data_qubit_locs = custom_layout[0]
        factory_locs = instance["factory_locs"]
        layout_name = instance["layout_name"]

        init_layout_lst = []
        final_layout_lst = []
        num_final_lst = []
        num_init_lst = []
        # for routing metric
        init_layout_lst2 = []
        final_layout_lst2 = []
        num_final_lst2 = []
        num_init_lst2 = []

        for circuit in circuits:
            # do hill climbing
            hc = hill_climber.HillClimbing(
                max_restarts,
                max_iterations,
                circuit,
                metric,
                t,
                custom_layout,
                use_dag,
                valid_path,
                factory_locs,
                num_factories=len(factory_locs),
                optimize_factories=optimize_factories,
            )
            # hard coded for now
            prefix = "./results"
            suffix = "test_251116"
            best_solution, _, best_rep, score_history = hc.run(
                prefix, suffix, parallel, processes
            )

            # do the initial routing
            input_layout = score_history[best_rep]["layout_init"]
            input_layout_cleaned = {
                key: val
                for key, val in input_layout.items()
                if key != "factory_positions"
            }
            init_layout_lst.append(input_layout)
            factory_positions = input_layout["factory_positions"]
            terminal_pairs = layouts.translate_layout_circuit(circuit, input_layout)
            router = utils.BasicRouter(
                g, data_qubit_locs, factory_positions, valid_path, t, metric, use_dag
            )
            # update routing graph
            layers = router.split_layer_terminal_pairs(terminal_pairs)
            vdp_layers_initial_dyn, _ = router.find_total_vdp_layers_dyn(
                layers,
                data_qubit_locs,
                router.factory_times,
                input_layout_cleaned,
                testing=True,
            )
            num_initial_dyn = len(vdp_layers_initial_dyn)
            num_init_lst.append(num_initial_dyn)

            # do the optimized routing
            input_layout = score_history[best_rep]["layout_final"]
            input_layout_cleaned = {
                key: val
                for key, val in input_layout.items()
                if key != "factory_positions"
            }
            final_layout_lst.append(input_layout)
            factory_positions = input_layout["factory_positions"]
            terminal_pairs = layouts.translate_layout_circuit(circuit, input_layout)
            router = utils.BasicRouter(
                g, data_qubit_locs, factory_positions, valid_path, t, metric, use_dag
            )
            # update routing graph
            layers = router.split_layer_terminal_pairs(terminal_pairs)
            vdp_layers_final_dyn, _ = router.find_total_vdp_layers_dyn(
                layers,
                data_qubit_locs,
                router.factory_times,
                input_layout_cleaned,
                testing=True,
            )
            num_final_dyn = len(vdp_layers_final_dyn)
            num_final_lst.append(num_final_dyn)

            # add time
            time.append(num_final_dyn)
        logger.info(f"time = {time}")
        logger.info(
            {"space": space, "time_mean": np.mean(time), "time_std": np.std(time)}
        )
        res_lst.append(
            {
                "space": space,
                "time_mean": np.mean(time),
                "time_std": np.std(time),
                "num_init_lst": num_init_lst,
                "num_final_lst": num_final_lst,
                "init_layout_lst": init_layout_lst,
                "final_layout_lst": final_layout_lst,
                "instances": instances,
                "hc_params": hc_params,
                "circuits": circuits,
            }
        )
        with Path(path).open("wb") as f:
            pickle.dump(res_lst, f)

        # lot of redundant code but too lazy right now
        if both_metric:
            logger.info("Run of additional hc with metric=routing")
            for circuit in circuits:
                # generate random circ
                # circuit = co.generate_random_circuit(q, min_depth, tgate, ratio)
                metric = "exact"
                # do hill climbing
                hc = hill_climber.HillClimbing(
                    max_restarts,
                    max_iterations,
                    circuit,
                    metric,
                    t,
                    custom_layout,
                    use_dag,
                    valid_path,
                    factory_locs,
                    num_factories=len(factory_locs),
                    optimize_factories=optimize_factories,
                )
                # hard coded for now
                prefix = "./results"
                suffix = "test_250218_2"
                _, _, best_rep, score_history = hc.run(
                    prefix, suffix, parallel, processes
                )

                # do the initial routing
                input_layout = score_history[best_rep]["layout_init"]
                input_layout_cleaned = {
                    key: val
                    for key, val in input_layout.items()
                    if key != "factory_positions"
                }
                init_layout_lst2.append(input_layout)
                factory_positions = input_layout["factory_positions"]
                terminal_pairs = layouts.translate_layout_circuit(circuit, input_layout)
                router = utils.BasicRouter(
                    g,
                    data_qubit_locs,
                    factory_positions,
                    valid_path,
                    t,
                    metric,
                    use_dag,
                )
                # update routing graph
                layers = router.split_layer_terminal_pairs(terminal_pairs)
                vdp_layers_initial_dyn, _ = router.find_total_vdp_layers_dyn(
                    layers,
                    data_qubit_locs,
                    router.factory_times,
                    input_layout_cleaned,
                    testing=True,
                )
                num_initial_dyn = len(vdp_layers_initial_dyn)
                num_init_lst2.append(num_initial_dyn)

                # do the optimized routing
                input_layout = score_history[best_rep]["layout_final"]
                input_layout_cleaned = {
                    key: val
                    for key, val in input_layout.items()
                    if key != "factory_positions"
                }
                final_layout_lst2.append(input_layout)
                factory_positions = input_layout["factory_positions"]
                terminal_pairs = layouts.translate_layout_circuit(circuit, input_layout)
                router = utils.BasicRouter(
                    g,
                    data_qubit_locs,
                    factory_positions,
                    valid_path,
                    t,
                    metric,
                    use_dag,
                )
                # update routing graph
                layers = router.split_layer_terminal_pairs(terminal_pairs)
                vdp_layers_final_dyn, _ = router.find_total_vdp_layers_dyn(
                    layers,
                    data_qubit_locs,
                    router.factory_times,
                    input_layout_cleaned,
                    testing=True,
                )
                num_final_dyn = len(vdp_layers_final_dyn)
                num_final_lst2.append(num_final_dyn)

                # add time
                time2.append(num_final_dyn)
            logger.info(f"time = {time}")
            logger.info(
                {"space": space, "time_mean": np.mean(time2), "time_std": np.std(time2)}
            )
            res_lst_routing.append(
                {
                    "space": space,
                    "time_mean": np.mean(time2),
                    "time_std": np.std(time2),
                    "num_init_lst": num_init_lst2,
                    "num_final_lst": num_final_lst2,
                    "init_layout_lst": init_layout_lst2,
                    "final_layout_lst": final_layout_lst2,
                    "instances": instances,
                    "hc_params": hc_params,
                    "circuits": circuits,
                }
            )
            new_path = path + "_metricrouting"
            with Path(new_path).open("wb") as f:
                pickle.dump(res_lst_routing, f)

    return res_lst


def plot_improvement_circuit_types(
    res_lst: list[dict[str, Any]],
    path: str = "./results",
    size: tuple[int, int] = (5, 4),
) -> None:
    """Based on a run of collect_data_space time with different circuit types and constant t, and constant factories.

    Plots the Improvement from hill climbing based on different circuit types.
    ONLY CNOTs without T gates.
    """
    instances = res_lst[0][
        "instances"
    ]  # index does not matter because accidentally stored redundantely.
    hc_params = res_lst[0]["hc_params"]

    # cut off instances at length of res_lst
    instances = instances[
        : len(res_lst)
    ]  # just in case there where more instacnes included but the run stopped earlier

    ratio = 1.0
    tgate = False

    # filter instances with desired values for
    idx_include = []
    for i, instance in enumerate(instances):
        if instance["ratio"] == ratio and instance["tgate"] == tgate:
            idx_include.append(i)

    # filter what range of t and ratio we get
    dct_mat = []  # gather for each included idx the value for t, ratio and improvement
    for i, instance in enumerate(instances):
        if i in idx_include:
            res = res_lst[i]
            num_init_lst = res["num_init_lst"]
            num_final_lst = res["num_final_lst"]
            improvements = []
            for ni, nf in zip(num_init_lst, num_final_lst):
                improvements.append((ni - nf) / ni)
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            dct_mat.append(
                {
                    "i": i,
                    "mean_final_layers": np.mean(num_final_lst),
                    "std_final_layers": np.std(num_final_lst),
                    "mean_improvement": mean_improvement,
                    "std_improvement": std_improvement,
                    "t": instance["t"],
                    "factory_locs": instance["factory_locs"],
                    "q": instance["q"],
                    "circuit_type": instance["circuit_type"],
                    "layout_name": instance["layout_name"],
                    "min_depth": instance["min_depth"],
                }
            )
    for _ in dct_mat:
        pass
    # reshape such that one gets lists with fixed layout_name and fixed q
    unique_q = {entry["q"] for entry in dct_mat}
    unique_circuit_types = {entry["circuit_type"] for entry in dct_mat}
    unique_layout_names = {entry["layout_name"] for entry in dct_mat}
    {entry["min_depth"] for entry in dct_mat}

    # define order of circuit_types
    circuit_types_ordered = ["sequential", "random", "parallelmax"]
    sorted_circuit_types = [
        el for el in circuit_types_ordered if el in unique_circuit_types
    ]
    dct_plot = {}
    dct_plot_abs = {}

    for layout_name, q in itertools.product(unique_layout_names, unique_q):
        key = (layout_name, q)
        # for layout_name, min_depth in itertools.product(unique_layout_names, unique_min_depths):
        # key = (layout_name, min_depth)
        lst_improvement = []
        lst_std = []
        lst_abslayers = []
        lst_std_abslayers = []
        for ckt_i in range(len(sorted_circuit_types)):
            for el in dct_mat:
                if (
                    el["q"] == q
                    and el["layout_name"] == layout_name
                    and el["circuit_type"] == sorted_circuit_types[ckt_i]
                ):
                    # if el["min_depth"] == min_depth and el["layout_name"] == layout_name and el["circuit_type"] == sorted_circuit_types[ckt_i]:
                    lst_improvement.append(el["mean_improvement"])
                    lst_std.append(el["std_improvement"])
                    lst_abslayers.append(el["mean_final_layers"])
                    lst_std_abslayers.append(el["std_final_layers"])
        dct_plot.update({key: [lst_improvement, lst_std]})
        dct_plot_abs.update({key: [lst_abslayers, lst_std_abslayers]})

    for _, _ in dct_plot.items():
        pass

    colors = plt.cm.rainbow(np.linspace(0, 1, 7))
    # Define marker and color for each layout type
    # !CHOOSE WHICH LAYOUT STYLE YOU WANT, is q varied or is the depth varied?
    layout_styles = {
        "hex24": {
            "color": colors[0],
            "marker": "o",
            "linestyle": "--",
            "label": "hex",
        },  # , q=24"},
        "hex42": {
            "color": colors[0],
            "marker": "x",
            "linestyle": "--",
            "label": "hex, q=42",
        },
        "hex60": {
            "color": colors[0],
            "marker": "v",
            "linestyle": "--",
            "label": "hex, q=60",
        },
        "row24": {
            "color": colors[1],
            "marker": "o",
            "linestyle": "--",
            "label": "row",
        },  # , q=24"},
        "row42": {
            "color": colors[1],
            "marker": "x",
            "linestyle": "--",
            "label": "row, q=42",
        },
        "row60": {
            "color": colors[1],
            "marker": "v",
            "linestyle": "--",
            "label": "row, q=60",
        },
        "pair24": {
            "color": colors[2],
            "marker": "o",
            "linestyle": "--",
            "label": "pair",
        },  # , q=24"},
        "pair42": {
            "color": colors[2],
            "marker": "x",
            "linestyle": "--",
            "label": "pair, q=42",
        },
        "pair60": {
            "color": colors[2],
            "marker": "v",
            "linestyle": "--",
            "label": "pair, q=60",
        },
    }
    # layout_styles = {
    #    "hex48": {"color": colors[0], "marker": "o", "linestyle": "--", "label": "hex, Num. gates = 48"},
    #    "row48": {"color": colors[1], "marker": "o", "linestyle": "--", "label": "row, Num. gates = 48"},
    #    "pair48": {"color": colors[2], "marker": "o", "linestyle": "--", "label": "pair, Num. gates = 48"},
    #    "hex96": {"color": colors[0], "marker": "x", "linestyle": "--", "label": "hex, Num. gates = 96"},
    #    "row96": {"color": colors[1], "marker": "x", "linestyle": "--", "label": "row, Num. gates = 96"},
    #    "pair96": {"color": colors[2], "marker": "x", "linestyle": "--", "label": "pair, Num. gates = 96"},
    #    "hex192": {"color": colors[0], "marker": "v", "linestyle": "--", "label": "hex, Num. gates = 192"},
    #    "row192": {"color": colors[1], "marker": "v", "linestyle": "--", "label": "row, Num. gates = 192"},
    #    "pair192": {"color": colors[2], "marker": "v", "linestyle": "--", "label": "pair, Num. gates = 192"},
    # }

    # plot
    _, ax = plt.subplots(figsize=size)
    legend_handles = {}
    for key, val in dct_plot.items():
        label = key[0] + str(key[1])
        lst_improvement = val[0]
        lst_std = val[1]
        style = layout_styles[label]
        ax.errorbar(
            range(len(sorted_circuit_types)),
            lst_improvement,
            yerr=lst_std,
            color=style["color"],
            fmt=style["marker"],
            linestyle=style["linestyle"],
        )
        # Add label only once per layout type
        legend_handles[label] = Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="None",
            label=style["label"],
        )

    # Add unique legend entries
    legend = ax.legend(
        handles=list(legend_handles.values()),
        loc="upper left",
        # bbox_to_anchor=(1.0, 0.5),  # Moves the legend above the plot, adapt this for other plots
        fontsize=10,
        ncol=1,  # Adjust the number of columns as needed
        fancybox=False,
        borderpad=0.2,
        handletextpad=0.2,  # Reduce space between legend markers and text
        columnspacing=0.5,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("black")

    ax.set_xticks(range(len(sorted_circuit_types)))
    sorted_circuit_types2 = ["seq.", "rand.", "max."]
    ax.set_xticklabels(sorted_circuit_types2, rotation=45)

    ax.set_ylabel(r"$\tilde{\Delta}$")  # ("Mean improvement $(n_i-n_f)/n_i$")
    ax.set_xlabel("Random Circuit type")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Create the filename based on hyperparameters
    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]
    file_path = (
        Path(path)
        / f"circuit_types_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q24_240321.pdf"
    )
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    plt.clf()
    # --------abs plot----------
    _, ax = plt.subplots(figsize=size)
    legend_handles = {}
    for key, val in dct_plot_abs.items():
        label = key[0] + str(key[1])
        lst_improvement = val[0]  # actually abslayer lst but too lazy to rename
        lst_std = val[1]
        style = layout_styles[label]
        ax.errorbar(
            range(len(sorted_circuit_types)),
            lst_improvement,
            yerr=lst_std,
            color=style["color"],
            fmt=style["marker"],
            linestyle=style["linestyle"],
        )
        # Add label only once per layout type
        legend_handles[label] = Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="None",
            label=style["label"],
        )

    # Add unique legend entries
    legend = ax.legend(
        handles=list(legend_handles.values()),
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.4,
        ),  # Moves the legend above the plot, adapt this for other plots
        fontsize=10,
        ncol=3,  # Adjust the number of columns as needed
        fancybox=False,
        borderpad=0.2,
        handletextpad=0.2,  # Reduce space between legend markers and text
        columnspacing=0.5,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("black")

    ax.set_xticks(range(len(sorted_circuit_types)))
    ax.set_xticklabels(sorted_circuit_types, rotation=45)

    ax.set_ylabel(r"$\Delta_f$")  # ("Mean improvement $(n_i-n_f)/n_i$")
    ax.set_xlabel("Random Circuit type")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Create the filename based on hyperparameters
    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]
    file_path = (
        Path(path)
        / f"circuit_types_abslayer_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q24_240321.pdf"
    )
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)


def plot_f_vs_t(
    res_lst: list[dict[str, Any]],
    q: int,
    ratio: float,
    layout_name: str,
    min_depth: int,
    graphtype: str,
    hc_params: dict[str, Any],
    path: str = "./results",
    size: tuple[int, int] = (5, 4),
) -> None:
    """Plots a Matrix Plot with variation in number of factories and t. Also plots std.

    Args:
        res_lst (list[Dict]): _description_
        q (int): _description_
        ratio (float): _description_
        layout_name (str): _description_
        min_depth (int): _description_
        graphtype (str): _description_
        hc_params (dict): _description
        path (str, optional): _description_. Defaults to "./results".
        size (tuple[int,int]): size of plot
    """
    # extract data and put into matrix
    instances = res_lst[0][
        "instances"
    ]  # index does not matter because accidentally stored redundantely.
    # hc_params = res_lst[0]["hc_params"]

    # cut off instances at length of res_lst
    instances = instances[
        : len(res_lst)
    ]  # just in case there where more instacnes included but the run stopped earlier

    # filter instances with desired values for
    idx_include = []
    for i, instance in enumerate(instances):
        if (
            instance["q"] == q
            and instance["ratio"] == ratio
            and instance["layout_name"] == layout_name
            and instance["min_depth"] == min_depth
        ):
            idx_include.append(i)

    # filter what range of t and ratio we get
    dct_mat = []  # gather for each included idx the value for t, ratio and improvement
    for i, instance in enumerate(instances):
        if i in idx_include:
            res = res_lst[i]
            num_init_lst = res["num_init_lst"]
            num_final_lst = res["num_final_lst"]
            improvements = []
            for ni, nf in zip(num_init_lst, num_final_lst):
                improvements.append((ni - nf) / ni)
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            dct_mat.append(
                {
                    "i": i,
                    "mean_final_layers": np.mean(num_final_lst),
                    "std_final_layers": np.std(num_final_lst),
                    "mean_improvement": mean_improvement,
                    "std_improvement": std_improvement,
                    "t": instance["t"],
                    "factory_locs": instance["factory_locs"],
                }
            )

    available_t = set()
    available_f = set()
    for el in dct_mat:
        available_t.add(el["t"])
        available_f.add(len(el["factory_locs"]))

    data = np.zeros((len(available_f), len(available_t)))
    data_std = np.zeros((len(available_f), len(available_t)))
    data_abs = np.zeros((len(available_f), len(available_t)))
    data_abs_std = np.zeros((len(available_f), len(available_t)))
    # available_f_dct = {f: i for i, f in enumerate(available_f)}
    # available_t_dct = {t: i for i, t in enumerate(available_t)}

    # order the entries
    available_t_sorted = sorted(available_t)
    available_f_sorted = sorted(available_f)

    available_f_dct = {f: i for i, f in enumerate(available_f_sorted)}
    available_t_dct = {t: i for i, t in enumerate(available_t_sorted)}

    for el in dct_mat:
        f_idx = available_f_dct[len(el["factory_locs"])]
        t_idx = available_t_dct[el["t"]]
        data[f_idx, t_idx] = el["mean_improvement"]
        data_std[f_idx, t_idx] = el["std_improvement"]
        data_abs[f_idx, t_idx] = el["mean_final_layers"]
        data_abs_std[f_idx, t_idx] = el["std_final_layers"]

    # ---------plot improvements-------------
    plt.figure(figsize=size)
    im = plt.imshow(data, cmap="viridis", aspect="auto")

    # add text std for each tile
    for i in range(data_std.shape[0]):  # Iterate rows
        for j in range(data_std.shape[1]):  # Iterate columns
            plt.text(
                j,
                i + 0.2,
                "std=" + str(round(data_std[i, j], 3)),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )
    for i in range(data.shape[0]):  # Iterate rows
        for j in range(data.shape[1]):  # Iterate columns
            plt.text(
                j,
                i,
                str(round(data[i, j], 3)),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )

    plt.xticks(
        ticks=list(available_t_dct.values()),
        labels=list(available_t_dct.keys()),
        rotation=45,
    )
    plt.yticks(
        ticks=list(available_f_dct.values()), labels=list(available_f_dct.keys())
    )

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(
        r"$\tilde{\Delta}$"
    )  # ("Mean Improvement $(n_i-n_f)/n_i$")  # Label for the colorbar

    plt.xlabel("Reset time $t$")
    plt.ylabel("Number of factories")

    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]

    plt.tight_layout()
    file_path = (
        Path(path)
        / f"f_vs_t_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_ratio{ratio}_layout{layout_name}_depth{min_depth}_graphtype{graphtype}_250321.pdf"
    )
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()
    plt.clf()

    """
    #--------plot improvements in 3d---------------
    X, Y = np.meshgrid(list(available_t_dct.values()), list(available_f_dct.values()))

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, data, cmap="viridis", edgecolor="k", linewidth=0.5, alpha=0.9)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    #cbar.set_label("Mean Layer Reduction $(n_i - n_f) / n_i$")

    ax.set_xlabel("Reset time $t$")
    ax.set_ylabel("Number of factories")
    ax.set_zlabel("Mean Improvement $(n_i - n_f) / n_i$")

    ax.legend()
    plt.show()

    plt.tight_layout()
    file_path = Path(path) / f"f_vs_t_3d_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_ratio{ratio}_layout{layout_name}_depth{min_depth}_2503011.pdf"
    plt.savefig(file_path)

    plt.clf()
    """

    # ---------plot abs layers-------------
    plt.figure(figsize=size)
    # need to adapt colorbar
    im = plt.imshow(data_abs, cmap="viridis", aspect="auto")

    # add text std for each tile
    for i in range(data_abs_std.shape[0]):  # Iterate rows
        for j in range(data_abs_std.shape[1]):  # Iterate columns
            plt.text(
                j,
                i,
                str(round(data_abs[i, j], 3)),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )
    for i in range(data_abs.shape[0]):  # Iterate rows
        for j in range(data_abs.shape[1]):  # Iterate columns
            plt.text(
                j,
                i + 0.2,
                "std=" + str(round(data_abs_std[i, j], 3)),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )

    plt.xticks(
        ticks=list(available_t_dct.values()),
        labels=list(available_t_dct.keys()),
        rotation=45,
    )
    plt.yticks(
        ticks=list(available_f_dct.values()), labels=list(available_f_dct.keys())
    )

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\Delta_f$")  # ("Number of Layers")  # Label for the colorbar

    plt.xlabel("Reset time $t$")
    plt.ylabel("Number of factories")

    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]

    plt.tight_layout()
    file_path = (
        Path(path)
        / f"f_vs_t_abslayers_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_ratio{ratio}_layout{layout_name}_depth{min_depth}_graphtype{graphtype}_250321.pdf"
    )
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)

    plt.show()

    plt.clf()

    """
    #------plot absolute layers in 3d---------

    X, Y = np.meshgrid(list(available_t_dct.values()), list(available_f_dct.values()))

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, data_abs, cmap="viridis", edgecolor="k", linewidth=0.5, alpha=0.9)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    #cbar.set_label("Number of Layers")

    ax.set_xlabel("Reset time $t$")
    ax.set_ylabel("Number of factories")
    ax.set_zlabel("Number of Layers")

    ax.legend()
    plt.show()

    plt.tight_layout()
    file_path = Path(path) / f"f_vs_t_abslayers_3d_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_ratio{ratio}_layout{layout_name}_depth{min_depth}_2503011.pdf"
    plt.savefig(file_path)
    """


def plot_ratio_vs_t(
    res_lst: list[dict[str, Any]],
    q: int,
    num_factories: int,
    layout_name: str,
    min_depth: int,
    path: str = "./results",
    size: tuple[int, int] = (5, 4),
) -> None:
    """Plots a Matrix Plot with variation in ratio and t. Also plots std.

    Args:
        res_lst (list[dict]): _description_
        q (int): _description_
        num_factories (int): _description_
        layout_name (str): _description_
        min_depth (int): _description_
        path (str, optional): _description_. Defaults to "./results".
        size (tuple[int,int]) : size of plot
    """
    # extract data and put into matrix
    instances = res_lst[0][
        "instances"
    ]  # index does not matter because accidentally stored redundantely.
    hc_params = res_lst[0]["hc_params"]

    # cut off instances at length of res_lst
    instances = instances[
        : len(res_lst)
    ]  # just in case there where more instacnes included but the run stopped earlier

    # filter instances with desired values for
    idx_include = []
    for i, instance in enumerate(instances):
        if (
            instance["q"] == q
            and len(instance["factory_locs"]) == num_factories
            and instance["layout_name"] == layout_name
            and instance["min_depth"] == min_depth
        ):
            idx_include.append(i)

    # filter what range of t and ratio we get
    dct_mat = []  # gather for each included idx the value for t, ratio and improvement
    for i, instance in enumerate(instances):
        if i in idx_include:
            res = res_lst[i]
            num_init_lst = res["num_init_lst"]
            num_final_lst = res["num_final_lst"]
            improvements = []
            for ni, nf in zip(num_init_lst, num_final_lst):
                improvements.append((ni - nf) / ni)
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            dct_mat.append(
                {
                    "i": i,
                    "mean_final_layers": np.mean(num_final_lst),
                    "std_final_layers": np.std(num_final_lst),
                    "mean_improvement": mean_improvement,
                    "std_improvement": std_improvement,
                    "t": instance["t"],
                    "ratio": instance["ratio"],
                }
            )

    available_t = set()
    available_ratio = set()
    for el in dct_mat:
        available_t.add(el["t"])
        available_ratio.add(el["ratio"])

    data = np.zeros((len(available_ratio), len(available_t)))
    data_std = np.zeros((len(available_ratio), len(available_t)))
    data_abs = np.zeros((len(available_ratio), len(available_t)))
    data_abs_std = np.zeros((len(available_ratio), len(available_t)))
    available_ratio_dct = {ratio: i for i, ratio in enumerate(available_ratio)}
    available_t_dct = {t: i for i, t in enumerate(available_t)}

    # order the entries
    # available_t = sorted(available_t)
    # available_ratio = sorted(available_ratio)

    for el in dct_mat:
        ratio_idx = available_ratio_dct[el["ratio"]]
        t_idx = available_t_dct[el["t"]]
        data[ratio_idx, t_idx] = el["mean_improvement"]
        data_std[ratio_idx, t_idx] = el["std_improvement"]
        data_abs[ratio_idx, t_idx] = el["mean_final_layers"]
        data_abs_std[ratio_idx, t_idx] = el["std_final_layers"]

    # -------------plot improvements------------------
    plt.figure(figsize=size)
    im = plt.imshow(data, cmap="viridis", aspect="auto")

    # add text std for each tile
    for i in range(data_std.shape[0]):  # Iterate rows
        for j in range(data_std.shape[1]):  # Iterate columns
            plt.text(
                j,
                i,
                str(round(data_std[i, j], 5)),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )

    plt.xticks(
        ticks=list(available_t_dct.values()),
        labels=list(available_t_dct.keys()),
        rotation=45,
    )
    plt.yticks(
        ticks=list(available_ratio_dct.values()),
        labels=list(available_ratio_dct.keys()),
    )

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Mean Improvement $(n_i-n_f)/n_i$")  # Label for the colorbar

    plt.xlabel("Reset time $t$")
    plt.ylabel(r"$\alpha = \frac{CNOTS}{all}$")

    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]

    plt.tight_layout()
    file_path = (
        Path(path)
        / f"ratio_vs_t_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_numfac{num_factories}_layout{layout_name}_depth{min_depth}.pdf"
    )
    plt.savefig(file_path)

    plt.show()

    plt.close()

    # ----------plot absolute layers-----------
    plt.figure(figsize=size)
    im = plt.imshow(data_abs, cmap="viridis", aspect="auto")

    # add text std for each tile
    for i in range(data_abs_std.shape[0]):  # Iterate rows
        for j in range(data_std.shape[1]):  # Iterate columns
            plt.text(
                j,
                i,
                str(round(data_std[i, j], 5)),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )

    plt.xticks(
        ticks=list(available_t_dct.values()),
        labels=list(available_t_dct.keys()),
        rotation=45,
    )
    plt.yticks(
        ticks=list(available_ratio_dct.values()),
        labels=list(available_ratio_dct.keys()),
    )

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Number of Layers")  # Label for the colorbar

    plt.xlabel("Reset time $t$")
    plt.ylabel(r"$\alpha = \frac{CNOTS}{all}$")

    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]

    plt.tight_layout()
    file_path = (
        Path(path)
        / f"ratio_vs_t_abslayers_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_numfac{num_factories}_layout{layout_name}_depth{min_depth}.pdf"
    )
    plt.savefig(file_path)

    plt.show()


def plot_space_time(
    instances: list[dict[str, Any]],
    hc_params: dict[str, Any],
    res_lst: list[dict[str, Any]],
    path: str = "./results",
    size: tuple[int, int] = (5, 4),
) -> None:
    """Plots the results from collect_data_space_time.

    Args:
        instances (list[dict]): List of instances containing various parameters.
        hc_params (dict): Hyperparameters used in the optimization.
        res_lst (list[dict]): Results containing space and time metrics.
        path (str, optional): Path to save the plot. Defaults to "./results".
        size (tuple[int,int]): Size of figure
    """
    assert len(instances) == len(
        res_lst
    ), "instances and res_lst do not have the same length."
    colors = plt.cm.rainbow(np.linspace(0, 1, 7))

    _, ax = plt.subplots(figsize=size)

    # Store unique legend entries
    legend_handles = {}

    for instance, result in zip(instances, res_lst):
        space = result["space"]
        time_mean = result["time_mean"]
        time_std = result["time_std"]

        q = instance["q"]
        # t = instance["t"]
        ratio = instance["ratio"]
        # num_factories = len(instance["factory_locs"])
        layout_name = instance["layout_name"]

        # Define marker and color for each layout type
        layout_styles = {
            "hex" + str(24): {"color": colors[0], "marker": "o", "label": "hex, q=24"},
            "hex" + str(42): {"color": colors[0], "marker": "x", "label": "hex, q=42"},
            "hex" + str(60): {"color": colors[0], "marker": "v", "label": "hex, q=60"},
            "row" + str(24): {"color": colors[1], "marker": "o", "label": "row, q=24"},
            "row" + str(42): {"color": colors[1], "marker": "x", "label": "row, q=42"},
            "row" + str(60): {"color": colors[1], "marker": "v", "label": "row, q=60"},
            "pair"
            + str(24): {"color": colors[2], "marker": "o", "label": "pair, q=24"},
            "pair"
            + str(42): {"color": colors[2], "marker": "x", "label": "pair, q=42"},
            "pair"
            + str(60): {"color": colors[2], "marker": "v", "label": "pair, q=60"},
            # do not plot sparse result
            # "sparse"+str(24): {"color": colors[3], "marker": "o", "label": "sparse, q=24"},
            # "row": {"color": colors[1], "marker": "x", "label": "row"},
            # "pair": {"color": colors[2], "marker": "*", "label": "pair"},
            # "sparse": {"color": colors[3], "marker": "v", "label": "sparse"},
        }

        layout_name_label = layout_name + str(q)

        if layout_name_label in layout_styles:
            style = layout_styles[layout_name_label]

            # Plot the point
            ax.errorbar(
                time_mean,
                space,
                xerr=time_std,
                color=style["color"],
                fmt=style["marker"],
            )

            # Add label only once per layout type
            legend_handles[layout_name_label] = Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="None",
                label=style["label"],
            )

            # Format the label with new lines
            # text_label = f"q={q}\n t={t}\n #f={num_factories} \n ratio={ratio}"
            text_label = f"r={ratio}"  # f"q={q} \n r={ratio}"
            ax.text(
                time_mean,
                space + 0.05,
                text_label,
                fontsize=8,
                ha="center",
                va="bottom",
            )

    # Ensure texts are not outside the plot
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.03 * y_max)

    # Add unique legend entries
    legend = ax.legend(
        handles=list(legend_handles.values()),
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.4,
        ),  # Moves the legend above the plot, adapt this for other plots
        fontsize=10,
        ncol=3,  # Adjust the number of columns as needed
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("black")

    ax.set_xlabel("Time (#Layers)")
    ax.set_ylabel("Space (# logical data and ancilla qubits)")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Create the filename based on hyperparameters
    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]
    file_path = (
        Path(path)
        / f"space_time_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_250308_new.pdf"
    )

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()


def plot_improvement_f_variation(
    res_lst_crossing: list[dict[str, Any]],
    res_lst_routing: list[dict[str, Any]],
    t: int,
    ratio: float,
    q: int,
    min_depth: int,
    path: str = "./results",
    size: tuple[int, int] = (5, 4),
) -> None:
    """Plots the hc improvement v.s. number of f for fixed t.

    Plot graphs for multiple layouts and metrics

    Args:
        res_lst_crossing (list[dict]): _description_
        res_lst_routing (list[dict]): _description_
        t (int): _
        ratio (float): _
        q (int): _description_
        layout_name (str): _description_
        min_depth (int): _description_
        path (str, optional): _description_. Defaults to "./results".
        size (tuple[int,int], optional): _description_. Defaults to (5,4).
    """
    # extract data for both the crossing and the routing metric run
    instances_crossing = res_lst_crossing[0][
        "instances"
    ]  # index does not matter because accidentally stored redundantely.
    hc_params_crossing = res_lst_crossing[0]["hc_params"]

    instances_routing = res_lst_routing[0][
        "instances"
    ]  # index does not matter because accidentally stored redundantely.
    hc_params_routing = res_lst_routing[0]["hc_params"]

    idx_include_c = []
    for i, instance in enumerate(instances_crossing):
        if (
            instance["q"] == q
            and instance["ratio"] == ratio
            and instance["min_depth"] == min_depth
            and instance["t"] == t
        ):
            idx_include_c.append(i)
    idx_include_r = []
    for i, instance in enumerate(instances_routing):
        if (
            instance["q"] == q
            and instance["ratio"] == ratio
            and instance["min_depth"] == min_depth
            and instance["t"] == t
        ):
            idx_include_r.append(i)

    dct_mat_c = []  # gather for each included idx the important outcomes
    for i, instance in enumerate(instances_crossing):
        if i in idx_include_c:
            res = res_lst_crossing[i]
            num_init_lst = res["num_init_lst"]
            num_final_lst = res["num_final_lst"]
            improvements = []
            for ni, nf in zip(num_init_lst, num_final_lst):
                improvements.append((ni - nf) / ni)
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            dct_mat_c.append(
                {
                    "i": i,
                    "mean_final_layers": np.mean(num_final_lst),
                    "std_final_layers": np.std(num_final_lst),
                    "mean_improvement": mean_improvement,
                    "std_improvement": std_improvement,
                    "t": instance["t"],
                    "factory_locs": instance["factory_locs"],
                    "layout_name": instance["layout_name"],
                }
            )

    dct_mat_r = []  # gather for each included idx the important outcomes
    for i, instance in enumerate(instances_routing):
        if i in idx_include_r:
            res = res_lst_routing[i]
            num_init_lst = res["num_init_lst"]
            num_final_lst = res["num_final_lst"]
            improvements = []
            for ni, nf in zip(num_init_lst, num_final_lst):
                improvements.append((ni - nf) / ni)
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            dct_mat_r.append(
                {
                    "i": i,
                    "mean_final_layers": np.mean(num_final_lst),
                    "std_final_layers": np.std(num_final_lst),
                    "mean_improvement": mean_improvement,
                    "std_improvement": std_improvement,
                    "t": instance["t"],
                    "factory_locs": instance["factory_locs"],
                    "layout_name": instance["layout_name"],
                }
            )

    # gather the lists where each list should form a graph i.e. a list for each metric / layout type and length should be number of different factoriy numbers.
    available_f_c = set()
    available_layout_c = set()
    for el in dct_mat_c:
        available_f_c.add(len(el["factory_locs"]))
        available_layout_c.add(el["layout_name"])

    available_f_r = set()
    available_layout_r = set()
    for el in dct_mat_r:
        available_f_r.add(len(el["factory_locs"]))
        available_layout_r.add(el["layout_name"])

    # assert available values should be the same for routing and crossing metric
    assert (
        available_f_c == available_f_r
    ), "Choose your data such that runs for both metrics provide same values for the number of factories"
    assert (
        available_layout_r == available_layout_c
    ), "Choose your data such that runs for both metrics provide same values for the layouts"

    # gather together lists for each graph
    data_c = np.zeros((len(available_f_c), len(available_layout_c)))
    data_std_c = data_c.copy()
    data_abs_c = data_c.copy()
    data_abs_std_c = data_c.copy()

    data_r = data_c.copy()
    data_std_r = data_c.copy()
    data_abs_r = data_c.copy()
    data_abs_std_r = data_c.copy()

    # available_f_c = sorted(available_f_c)

    available_f_dct = {f: i for i, f in enumerate(sorted(available_f_c))}
    available_layout_dct = {f: i for i, f in enumerate(available_layout_c)}

    for el in dct_mat_c:
        f_idx = available_f_dct[len(el["factory_locs"])]
        l_idx = available_layout_dct[el["layout_name"]]
        data_c[f_idx, l_idx] = el["mean_improvement"]
        data_std_c[f_idx, l_idx] = el["std_improvement"]
        data_abs_c[f_idx, l_idx] = el["mean_final_layers"]
        data_abs_std_c[f_idx, l_idx] = el["std_final_layers"]

    for el in dct_mat_r:
        f_idx = available_f_dct[len(el["factory_locs"])]
        l_idx = available_layout_dct[el["layout_name"]]
        data_r[f_idx, l_idx] = el["mean_improvement"]
        data_std_r[f_idx, l_idx] = el["std_improvement"]
        data_abs_r[f_idx, l_idx] = el["mean_final_layers"]
        data_abs_std_r[f_idx, l_idx] = el["std_final_layers"]

    # define colors for each layout and use different line types for the metrics. however you have two lists, so do it directly
    colors = plt.cm.rainbow(np.linspace(0, 1, 7))
    _, ax = plt.subplots(figsize=size)

    for k, lay in enumerate(available_layout_c):
        # routing metric
        ax.errorbar(
            list(available_f_c),
            data_c[:, k],
            yerr=data_std_c[:, k],
            color=colors[k],
            fmt="o",
            linestyle="--",
            label=f"{lay}, crossing",
        )
        # crossing metric
        ax.errorbar(
            list(available_f_r),
            data_r[:, k],
            yerr=data_std_r[:, k],
            color=colors[k],
            fmt="v",
            linestyle="--",
            label=f"{lay}, routing",
        )

    # plt.yticks(ticks=list(available_f_dct.values()), labels=list(available_f_dct.keys()), rotation=45)
    plt.legend()
    ax.set_ylabel("Mean improvement $(n_i-n_f)/n_i$")
    ax.set_xlabel("Number of factories")

    # only integer x
    ax.set_xticks(available_f_c)
    ax.set_xticklabels(available_f_c)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    max_restarts_c = hc_params_crossing["max_restarts"]
    max_iterations_c = hc_params_crossing["max_iterations"]
    max_restarts_r = hc_params_routing["max_restarts"]
    max_iterations_r = hc_params_routing["max_iterations"]
    assert max_iterations_r == max_iterations_c
    assert max_restarts_c == max_restarts_r
    assert len(instances_crossing) == len(instances_routing)
    filepath = (
        Path(path)
        / f"f_variation_t{t}_restarts{max_restarts_c}_it{max_iterations_c}_numinstances{len(instances_routing)}_q{q}_ratio{ratio}_depth{min_depth}.pdf"
    )
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()


def plot_f_vs_t_subfigs(
    res_lst1: list[dict[str, Any]],
    res_lst2: list[dict[str, Any]],
    q: int,
    ratio: float,
    layout_name: str,
    min_depth: int,
    graphtype: str,
    hc_params: dict[str, Any],
    path: str = "./results",
    size: tuple[int, int] = (5, 5),
) -> None:
    """Plots a Matrix Plot with variation in number of factories and t for two result lists in subplots."""

    def process_res_list(res_lst: list[dict[str, Any]]) -> tuple[Any, ...]:
        instances = res_lst[0]["instances"][: len(res_lst)]
        idx_include = [
            i
            for i, instance in enumerate(instances)
            if instance["q"] == q
            and instance["ratio"] == ratio
            and instance["layout_name"] == layout_name
            and instance["min_depth"] == min_depth
        ]
        dct_mat = []

        for i in idx_include:
            res = res_lst[i]
            num_init_lst: list[int] = res["num_init_lst"]
            num_final_lst: list[int] = res["num_final_lst"]
            improvements = [
                (ni - nf) / ni for ni, nf in zip(num_init_lst, num_final_lst)
            ]
            dct_mat.append(
                {
                    "mean_final_layers": np.mean(num_final_lst),
                    "std_final_layers": np.std(num_final_lst),
                    "mean_improvement": np.mean(improvements),
                    "std_improvement": np.std(improvements),
                    "t": instances[i]["t"],
                    "factory_locs": instances[i]["factory_locs"],
                }
            )

        available_t = sorted({el["t"] for el in dct_mat})
        available_f = sorted({len(el["factory_locs"]) for el in dct_mat})
        available_t_dct = {t: i for i, t in enumerate(available_t)}
        available_f_dct = {f: i for i, f in enumerate(available_f)}

        data = np.zeros((len(available_f), len(available_t)))
        data_std = np.zeros((len(available_f), len(available_t)))
        data_abs = np.zeros((len(available_f), len(available_t)))
        data_abs_std = np.zeros((len(available_f), len(available_t)))

        for el in dct_mat:
            f_idx = available_f_dct[len(el["factory_locs"])]
            t_idx = available_t_dct[el["t"]]
            data[f_idx, t_idx] = el["mean_improvement"]
            data_std[f_idx, t_idx] = el["std_improvement"]
            data_abs[f_idx, t_idx] = el["mean_final_layers"]
            data_abs_std[f_idx, t_idx] = el["std_final_layers"]

        return (
            data,
            data_std,
            data_abs,
            data_abs_std,
            available_t,
            available_f,
            available_t_dct,
            available_f_dct,
        )

    (
        data1,
        data_std1,
        data_abs1,
        data_abs_std1,
        _available_t1,
        _available_f1,
        available_t_dct,
        available_f_dct,
    ) = process_res_list(res_lst1)
    data2, data_std2, data_abs2, data_abs_std2, _available_t2, _available_f2, _, _ = (
        process_res_list(res_lst2)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=size,
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]},
    )

    def plot_with_text(
        ax: plt.axes.Axes, data: NDArray[Any], data_std: NDArray[Any], r: int
    ) -> plt.image.AxesImage:
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    str(round(data[i, j], r)),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground="black")
                    ],
                )
                ax.text(
                    j,
                    i + 0.25,
                    r"$\pm$" + str(round(data_std[i, j], r)),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground="black")
                    ],
                )
        ax.set_xticks(list(available_t_dct.values()))
        ax.set_xticklabels(list(available_t_dct.keys()), rotation=45)
        ax.set_yticks(list(available_f_dct.values()))
        ax.set_yticklabels(list(available_f_dct.keys()))
        return im

    def plot_with_text_int(
        ax: plt.axes.Axes, data: NDArray[Any], data_std: NDArray[Any]
    ) -> plt.image.AxesImage:
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        r = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(round(data[i, j], r))),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground="black")
                    ],
                )
                ax.text(
                    j,
                    i + 0.25,
                    r"$\pm$" + str(int(round(data_std[i, j], r))),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground="black")
                    ],
                )
        ax.set_xticks(list(available_t_dct.values()))
        ax.set_xticklabels(list(available_t_dct.keys()), rotation=45)
        ax.set_yticks(list(available_f_dct.values()))
        ax.set_yticklabels(list(available_f_dct.keys()))
        return im

    # Global color limits
    vmin1, vmax1 = min(data1.min(), data2.min()), max(data1.max(), data2.max())
    vmin2, vmax2 = min(data_abs1.min(), data_abs2.min()), max(
        data_abs1.max(), data_abs2.max()
    )

    # Plot data with consistent color limits
    r = 2
    im1 = plot_with_text(axes[0, 0], data1, data_std1, r)
    im2 = plot_with_text(axes[0, 1], data2, data_std2, r)
    im3 = plot_with_text_int(axes[1, 0], data_abs1, data_abs_std1)
    im4 = plot_with_text_int(axes[1, 1], data_abs2, data_abs_std2)

    # Apply the same color limits to ensure consistent colors across the plots
    im1.set_clim(vmin1, vmax1)
    im2.set_clim(vmin1, vmax1)
    im3.set_clim(vmin2, vmax2)
    im4.set_clim(vmin2, vmax2)

    # Add colorbars outside the plot using the `pad` parameter
    cbar_ax1 = fig.add_axes([0.92, 0.65, 0.02, 0.3])  # Position for the first colorbar
    cbar_ax2 = fig.add_axes([0.92, 0.2, 0.02, 0.3])  # Position for the second colorbar

    # Add colorbars and set the labels directly
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation="vertical")
    cbar1.set_label(r"$\tilde{\Delta}$")

    cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation="vertical")
    cbar2.set_label(r"$\Delta_f$")

    # Set axis labels for all subplots
    # axes[0, 0].set_xlabel('Reset time $t$')
    # axes[0, 0].set_ylabel('Number of factories')

    # axes[0, 1].set_xlabel('Reset time $t$')
    # axes[0, 1].set_ylabel('Number of factories')

    # axes[1, 0].set_xlabel('Reset time $t$')
    # axes[1, 0].set_ylabel('Number of factories')

    # axes[1, 1].set_xlabel('Reset time $t$')
    # axes[1, 1].set_ylabel('Number of factories')

    fig.supxlabel("Reset time $t$", fontsize=12)
    fig.supylabel("Number of factories", fontsize=12)
    plt.subplots_adjust(
        left=0.15, right=0.95, top=0.95, bottom=0.12
    )  # Fine-tune spacing

    # Adjust layout to give room for colorbars
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save the figure to the specified file path
    instances = res_lst1[0]["instances"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]
    file_path = (
        Path(path)
        / f"f_vs_t_ALL_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}_q{q}_ratio{ratio}_layout{layout_name}_depth{min_depth}_graphtype{graphtype}_250321.pdf"
    )
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)

    # Show the plot
    plt.show()
