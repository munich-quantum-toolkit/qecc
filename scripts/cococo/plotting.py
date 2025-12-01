# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import rainbow

if TYPE_CHECKING:
    from mqt.qecc.cococo.types import HistoryTemp


def plot_lattice(
    g: nx.Graph,
    size: tuple[float, float] = (3.5, 3.5),
    data_qubit_locs: list[tuple[int, int]] | None = None,
    factory_locs: list[tuple[int, int]] | None = None,
) -> None:
    """Plots the lattice G with networkx labels."""
    if data_qubit_locs is None:
        data_qubit_locs = []
    if factory_locs is None:
        factory_locs = []
    pos = nx.get_node_attributes(g, "pos")

    plt.figure(figsize=size)
    nx.draw(
        g,
        pos,
        with_labels=True,
        font_size=8,
        node_color="lightgray",
        edge_color="lightblue",
    )

    if len(data_qubit_locs) != 0:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=data_qubit_locs,
            node_color="orange",
        )

    if len(factory_locs) != 0:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=factory_locs,
            node_color="violet",
        )


def plot_history(
    hill_climber_obj: any,
    score_history: dict[int, HistoryTemp],
    filename: str = "./hc_history_plot.pdf",
    size: tuple[float, float] = (5, 5),
) -> None:
    """Plots the scores for each restart and iteration. from a hill climber run for initial mapping opt.

    Args:
        hill_climber_obj (any): current hill climber object.
        score_history (dict): Score history from HillClimber.run
        filename (str, optional): Path to store the plot. Defaults to "./hc_history_plot.pdf".
        size (tuple[float,float], optional): Size of the plot. Defaults to (3.5,3.5).
    """
    plt.figure(figsize=size)
    for rep, history in score_history.items():
        scores = history["scores"]
        plt.plot(
            range(len(scores)),
            scores,
            "x-",
            color=rainbow(rep / len(list(score_history.keys()))),
            label=f"Restart {rep}",
        )
    plt.legend()
    plt.ylabel(f"{hill_climber_obj.metric}")
    plt.xlabel("Hill Climbing Iteration")
    plt.title(f"$q=${hill_climber_obj.q}, Layout-Type = {hill_climber_obj.layout_type}, Num CNOTS = {len(hill_climber_obj.circuit)}")
    plt.savefig(filename)


def plot_lattice_paths(
    g: nx.Graph,
    vdp_dict: (
        dict[
            tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
            list[tuple[int, int]],
        ]
        | None
    ),
    steiner_dct: dict | None = None,
    layout: dict[tuple[int, int], int] | None = None,
    factory_locs: list[tuple[int, int]] | None = None,
    size: tuple[float, float] = (3.5, 3.5),
) -> None:
    """Plots the graph and the corresponding VDP of a layer.

    Args:
        g (nx.Graph): Graph on which we route
        vdp_dict (dict): Output of router.find_total_vdp_layers
        steiner_dct (dict): dct of trees.
        layout (dict): potentially also display the qubit labels. keys = qubit label, value = node label
        factory_locs (list[tuple[int,int]] | None): factory locations.
        size (tuple[float,float], optional): _description_. Size of the plot. Defaults to (3.5,3.5).
    """
    if layout is None:
        layout = {}
    if factory_locs is None:
        factory_locs = []
    pos = nx.get_node_attributes(g, "pos")

    plt.figure(figsize=size)
    nx.draw(g, pos, with_labels=True, node_color="gray", edge_color="lightblue", font_size=8)

    num_paths = len(vdp_dict.keys())
    num_trees = len(steiner_dct.keys()) if steiner_dct is not None else 0
    num = num_paths + num_trees
    colormap = plt.cm.get_cmap("rainbow", num)
    colors = [mcolors.to_hex(colormap(i)) for i in range(num)]

    offset = 0
    if vdp_dict is not None:
        for i, path in enumerate(vdp_dict.values()):
            if path:
                path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
                nx.draw_networkx_edges(g, pos, edgelist=path_edges, width=2, edge_color=colors[i])
                nx.draw_networkx_nodes(g, pos, nodelist=path, node_color=colors[i], label=f"Path {i + 1}")
                offset += 1

    if steiner_dct is not None:
        for i, (path1, path2) in enumerate(steiner_dct.values()):
            if path1 and path2:
                path_edges1 = [(path1[j], path1[j + 1]) for j in range(len(path1) - 1)]
                path_edges2 = [(path2[j], path2[j + 1]) for j in range(len(path2) - 1)]
                k = i + offset
                nx.draw_networkx_edges(g, pos, edgelist=path_edges1, width=2, edge_color=colors[k])
                nx.draw_networkx_nodes(g, pos, nodelist=path1, node_color=colors[k], label=f"Path {k + 1}")
                nx.draw_networkx_edges(g, pos, edgelist=path_edges2, width=2, edge_color=colors[k])
                nx.draw_networkx_nodes(g, pos, nodelist=path2, node_color=colors[k])

    if len(factory_locs) != 0:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=factory_locs,
            node_color="violet",
        )

    if len(list(layout.keys())) != 0:
        for key, value in layout.items():
            node_pos = pos[value]
            plt.text(
                node_pos[0],
                node_pos[1] - 0.1,
                str(key),
                fontsize=8,
                color="white",
                horizontalalignment="center",
            )
            # also highlight data qubits
            nx.draw_networkx_nodes(
                g,
                pos,
                nodelist=layout.values(),  # Nodes to highlight
                node_color="none",  # Unfilled circles
                edgecolors="lime",  # Neon green outline
                linewidths=1.5,  # Line width for the outline
            )

    plt.legend()
    plt.show()


def plot_schedule(g: nx.Graph, schedule: dict[any], factory_pos: list[tuple[int,int]], size: tuple[int,int]=(5, 5)) -> None:
    """Plots the layers of a whole schedule from TeleportationRouter."""
    for i, step in enumerate(schedule):
        print(f"Step {i + 1}: Move Type - {step['move_type']}, Idle Move - {step['idle_move_label']}")
        print("vdp dict", step["vdp_dict"].keys())
        plot_lattice_paths(
            g,
            vdp_dict=step["vdp_dict"],
            steiner_dct=step["steiner"],
            layout=step["layout"],
            factory_locs=factory_pos,
            size=size,
        )
