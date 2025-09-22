# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Plotting functions for the paper."""

from __future__ import annotations

import argparse
import json
import operator
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from matplotlib.axes import Axes


ler_k = "logical_error_rates"
ler_eb_k = "logical_error_rate_ebs"
min_wts_k = "min_wts_logical_err"


def plot_ler_vs_distance(code_dict: dict[float, Any], ax: Axes, pers: list[float], logical_idx: int = 0) -> None:
    """Plot the logical error rate vs distance for different err rates."""
    for p in pers:
        ds = []
        lers = []
        for d, ler in sorted(code_dict[p].items()):
            ds.append(int(d))
            lers.append(ler[logical_idx])
        ax.plot(ds, lers, label="p=" + str(round(p, 2)), marker="o", linestyle="-")
    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("Logical failure rate")
    ax.set_xlabel(r"Code distance $\it\{d\}$")


def threshold_fit(
    variables: tuple[float, float],
    b0: float,
    b1: float,
    b2: float,
    mu: float,
    pth: float,
) -> float:
    """Compute standard fit function for the threshold."""
    p, ell = variables
    expr = (p - pth) * (ell ** (1 / mu))
    return cast("float", b0 + b1 * expr + b2 * (expr**2))


def calculate_threshold(
    ax: Axes,
    code_dict: dict[int, Any],
    min_per: float = 0.06,
    max_per: float = 0.13,
    title: str | None = None,
    logical_idx: int = 0,
) -> None:
    """Calculate the threshold for the given results."""
    ler_data = []
    ler_eb_data = []
    per_data = []
    distance_data = []
    for distance in code_dict:
        per_array = []
        ler_array = []
        ler_eb = []

        for index, per in enumerate(code_dict[distance]["p"]):
            if min_per < per < max_per:
                per_array.append(per)
                ler_array.append(code_dict[distance][ler_k][index][logical_idx])
                ler_eb.append(code_dict[distance][ler_eb_k][index][logical_idx])
        per_data.extend(per_array)
        ler_data.extend(ler_array)
        ler_eb_data.extend(ler_eb)
        distance_data.extend([int(distance) for _ in range(len(per_array))])

    popt, _ = curve_fit(threshold_fit, (per_data, distance_data), ler_data, maxfev=10000)
    if ax is not None:
        ax.axvline(x=popt[-1], color="black", linestyle="dashed")

    distance_array = [int(distance) for distance in code_dict]
    distance_array.sort()
    for distance in distance_array:
        per_array = code_dict[distance]["p"]
        la = np.array(code_dict[distance][ler_k])
        ler_arr = la[:, logical_idx]
        ea = np.array(code_dict[distance][ler_eb_k])
        ler_eb_ar = ea[:, logical_idx]

        if per_array != [] and ax is not None:
            ax.errorbar(
                per_array,
                ler_arr,
                yerr=ler_eb_ar,
                label="d = " + str(distance),
                fmt="|",
            )

    ax.legend()
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlim(min_per, max_per)


def generate_plots(results_dir: Path, results_file: Path) -> None:
    """Generate the plots for the paper."""
    # read in all generated data
    data = []
    for file in results_dir.glob("*.json"):
        with file.open() as f:
            data.append(json.loads(f.read()))

    _, ax = plt.subplots(4, 4, figsize=(12, 12))
    metrics: dict[int, Any] = {}
    per_metrics: dict[float, Any] = {}

    for result in data:
        d = result["distance"]
        p = result["p"]

        if d not in metrics:
            metrics[d] = {
                "p": [],
                ler_k: [],
                ler_eb_k: [],
                "avg_total_time": [],
                min_wts_k: [],
            }
        if p not in per_metrics:
            per_metrics[p] = {}

        metrics[d]["p"].append(p)
        metrics[d][ler_k].append(result[ler_k])
        metrics[d][ler_eb_k].append(result[ler_eb_k])
        metrics[d]["avg_total_time"].append(result["avg_total_time"])
        metrics[d][min_wts_k].append(result[min_wts_k])

        per_metrics[p][d] = result[ler_k]
    for d, mdata in sorted(metrics.items()):
        (mdata["p"], mdata[ler_k], mdata["avg_total_time"], mdata[ler_eb_k]) = zip(
            *sorted(zip(mdata["p"], mdata[ler_k], mdata["avg_total_time"], mdata[ler_eb_k]))
        )

        # sum over all logical to get overall ler+ebs
        mdata["ler_sum"] = np.sum(mdata[ler_k], axis=1)
        mdata["ler_eb_sum"] = np.sum(mdata[ler_eb_k], axis=1)

        # plot lers for different logicals separately
        for logical_idx in range(len(mdata[ler_k][0])):
            y = np.array(mdata[ler_k])
            y = y[:, logical_idx]
            ebs = np.array(mdata[ler_eb_k])
            ebs = ebs[:, logical_idx]
            ax[0][logical_idx].errorbar(mdata["p"], y, ebs, label="d=" + str(d))
        ax[0][0].set_xlabel("Physical error rate")
        ax[0][0].set_ylabel("Logical error rate")
        ax[0][0].legend()

        ax[1][0].plot(mdata["p"], mdata["avg_total_time"], label="d=" + str(d))
        ax[1][0].set_xlabel("Physical error rate")
        ax[1][0].set_ylabel("Average time per run (micro seconds)")
        ax[1][0].legend()
        ax[1][0].set_ylim(0, 185000)

    # plot the average time per run over the distance
    ds = []
    p_data: dict[float, Any] = {}
    pers = [0.001, 0.02, 0.05, 0.08, 0.13]
    for d, mdata in sorted(metrics.items()):
        ds.append(d)
        for i, p in enumerate(mdata["p"]):
            if p in pers:
                if p not in p_data:
                    p_data[p] = {"d": [], "t": []}
                p_data[p]["d"].append(d)
                p_data[p]["t"].append(mdata["avg_total_time"][i])
    for p, pdata in sorted(p_data.items()):
        ax[2][1].plot(ds, pdata["t"], label="p=" + str(p))

    ax[1][1].set_xlabel("Distance")
    ax[1][1].set_ylabel("Average time per run (microseconds)")
    ax[1][1].legend()
    ax[1][1].set_xticks(ds)
    ax[1][1].set_ylim(0, 185000)
    for logical_idx in range(len(mdata[ler_k][0])):
        calculate_threshold(code_dict=metrics, ax=ax[1][logical_idx], title="Threshold")
    plot_ler_vs_distance(per_metrics, ax=ax[3][0], pers=[0.01])
    # save plot as vector graphic
    plt.savefig(results_file, bbox_inches="tight")


def generate_plots_tn(results_dir: Path, results_file: Path) -> None:
    """Generate plots for TN decoder."""
    # read in all generated data
    data = []
    for file in results_dir.glob("*.json"):
        with file.open() as f:
            data.append(json.loads(f.read()))

    # prepare code to per,ler map and print
    code_to_xys = {}  # type: ignore[var-annotated]
    for run in data:
        xys = code_to_xys.setdefault(run["n_k_d"][-1], [])
        xys.append((run["physical_error_rate"], run["logical_failure_rate"]))

    for xys in code_to_xys.values():
        xys.sort(key=operator.itemgetter(0))

    _fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    # add data
    for code, xys in sorted(code_to_xys.items()):
        ax[0][0].plot(*zip(*xys), "x-", label=f"d={code}")
    ax[0][0].set_xlabel("Physical error rate")
    ax[0][0].set_ylabel("Logical error rate")
    ax[0][0].legend()

    # prepare code to per,average-time-per-run map and print
    code_to_xys = {}
    for run in data:
        xys = code_to_xys.setdefault(run["n_k_d"][-1], [])
        # convert from seconds to microseconds
        xys.append((run["error_probability"], (run["wall_time"] / run["n_run"]) * 1e6))

    for xys in code_to_xys.values():
        xys.sort(key=operator.itemgetter(0))

    for code, xys in sorted(code_to_xys.items()):
        ax[1][0].plot(*zip(*xys), "x-", label=f"d={code}")
    ax[1][0].set_xlabel("Physical error rate")
    ax[1][0].set_ylabel("Average time per run (µs)")  # noqa: RUF001
    ax[1][0].legend()
    ax[1][0].set_ylim(0, 300000)

    ds = []
    p_data = {}  # type: ignore[var-annotated]
    pers = [0.051, 0.081, 0.111]  # 0.001, 0.021,
    for d, data in sorted(code_to_xys.items()):
        ds.append(d)
        for p, t in data:
            if p in pers:
                if p not in p_data:
                    p_data[p] = {"d": [], "t": []}
                p_data[p]["d"].append(d)
                p_data[p]["t"].append(t)

    for p, data in sorted(p_data.items()):
        ax[1][1].plot(ds, data["t"], label="p=" + str(p))  # type: ignore[call-overload]

    ax[1][1].set_xlabel("Distance")
    ax[1][1].set_ylabel("Average time per run (µs)")  # noqa: RUF001
    ax[1][1].legend()
    ax[1][1].set_yscale("log")
    ax[1][1].set_xticks(ds)
    ax[1][1].set_ylim(0, 300000)

    data = []
    for file in results_dir.glob("*.json"):
        with file.open() as f:
            data.append(json.loads(f.read()))
    metrics = {}  # type: ignore[var-annotated]
    per_metrics = {}  # type: ignore[var-annotated]

    # save plot as vector graphic
    for result in data:
        d = result["n_k_d"][2]
        p = result["physical_error_rate"]

        if d not in metrics:
            metrics[d] = {
                "p": [],
                "logical_error_rate": [],
            }
        if p not in per_metrics:
            per_metrics[p] = {}

        metrics[d]["p"].append(p)
        metrics[d]["logical_error_rate"].append(result["logical_failure_rate"])
    calculate_threshold(code_dict=metrics, ax=ax[0][1], title="Threshold")
    plt.savefig(results_file, bbox_inches="tight")


def generate_plots_comp(results_dir: Path, results_file: Path) -> None:
    """Generate plots for the comparison of the different solvers."""
    _, ax = plt.subplots(2, figsize=(12, 12))
    cols = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "cyan",
        "olive",
    ]
    idx = 0
    solver_to_col: dict[str, str] = {}
    p_to_col: dict[float, str] = {}
    for subdir, _, files in os.walk(results_dir):
        if not files:
            continue
        data = []
        solver = subdir.split("/")[-1]
        if solver not in solver_to_col:
            solver_to_col[solver] = cols[idx]
            idx += 1
        for f in files:
            fp = subdir + "/" + f
            with Path(fp).open(encoding="utf-8") as ff:
                data.append(json.loads(ff.read()))

        metrics: dict[int, dict[str, Any]] = {}
        for result in data:
            d = result["distance"]
            p = result["p"]
            if d not in metrics:
                metrics[d] = {"p": [], "avg_total_time": []}

            metrics[d]["p"].append(p)
            metrics[d]["avg_total_time"].append(result["avg_total_time"])

        for d, mdata in sorted(metrics.items()):
            if d in {17, 21}:
                (
                    mdata["p"],
                    mdata["avg_total_time"],
                ) = zip(*sorted(zip(mdata["p"], mdata["avg_total_time"])))
                ax[0].plot(
                    mdata["p"],
                    mdata["avg_total_time"],
                    label="solver=" + solver + ", d=" + str(d),
                    color=solver_to_col[solver],
                )
        ds: list[int] = []
        idx = 0
        p_data: dict[float, dict[str, list[Any]]] = {}
        pers = [0.001, 0.051, 0.131]
        for d, pdata in sorted(metrics.items()):
            ds.append(d)
            for i, p in enumerate(pdata["p"]):
                if p in pers:
                    if p not in p_data:
                        p_data[p] = {"d": [], "t": []}
                        p_to_col[p] = cols[idx]
                        idx += 1
                    p_data[p]["d"].append(d)
                    p_data[p]["t"].append(pdata["avg_total_time"][i])
        for p, ppdata in sorted(p_data.items()):
            ax[1].plot(
                ds,
                ppdata["t"],
                label="p=" + str(p) + ", " + (solver if solver == "z3" else "CASHWMaxSAT-CorePlus"),
                color=p_to_col[p],
                marker="x" if solver == "z3" else "o",
            )

        ax[1].set_xlabel("Distance")
        ax[1].set_ylabel("Average time per run (microseconds)")
        ax[1].legend()
        ax[1].set_xticks(ds)

        ax[0].set_xlabel("Physical error rate")
        ax[0].set_ylabel("Average time per run (microseconds)")
        ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # save plot as vector graphic
    plt.savefig(results_file, bbox_inches="tight")


def main() -> None:
    """Run main CLI function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the directory containing the results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="maxsat",
        help="Mode to use. Either 'maxsat' or 'tn'. Default: maxsat",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="./results.pdf",
        help="File where the results should be saved. Default: ./results.pdf",
    )
    args = parser.parse_args()

    if args.mode == "maxsat":
        generate_plots(Path(args.results_dir), Path(args.results_file))
    elif args.mode == "tn":
        generate_plots_tn(Path(args.results_dir), Path(args.results_file))
    elif args.mode == "comparison":
        generate_plots_comp(Path(args.results_dir), Path(args.results_file))
    else:
        msg = "Unknown mode: " + args.mode
        raise ValueError(msg)
