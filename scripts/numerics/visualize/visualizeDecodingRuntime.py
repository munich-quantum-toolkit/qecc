# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Plot the decoding runtime for different code lengths and different error probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
mpl.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
mpl.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"


def lin_fun(x: float, a: float, b: float) -> float:
    """Return the value of a*x + b."""
    return np.multiply(a, x) + b


def quad_fun(x: float, a: float, b: float) -> float:
    """Return the value of a*x^2 + b."""
    return np.multiply(a, np.power(x, 2)) + b


def pow_three_fun(x: float, a: float, b: float) -> float:
    """Return the value of a*x^3 + b."""
    return np.multiply(a, np.power(x, 3)) + b


def pow_four_fun(x: float, a: float, b: float) -> float:
    """Return the value of a*x^4 + b."""
    return np.multiply(a, np.power(x, 4)) + b


def runtime() -> None:
    """Plot the runtime."""
    plt.rcParams.update({"font.size": 15})
    input_filen = "/home/luca/Documents/codeRepos/qecc/scripts/numerics/data/runtime/rt-original-1k-01.json"
    x_data = []
    y_data = []
    pers = []
    orders = []

    with Path(input_filen).open as data_file:
        data = json.load(data_file)

    for per in data:
        per_x_data = []
        per_y_data = []

        for c in data[per]:
            per_x_data.append(float(c))
            per_y_data.append(float(data[per][c]) / 1000.0)
        pers.append(float(per))
        x_data.append(per_x_data)
        y_data.append(per_y_data)

    xfinal = []
    yfinal = []
    cols = []
    colors = mcolors.BASE_COLORS

    for i in range(len(x_data)):
        col, _val = colors.popitem()
        if col in {"w", "k"}:
            col, _val = colors.popitem()
            if col in {"w", "k"}:
                col, _val = colors.popitem()
                cols.append(col)
        label = f"{pers[i]: 6.3f}"
        orders.append(np.argsort(x_data[i]))
        xfinal.append(np.array(x_data[i])[orders[i]])
        yfinal.append(np.array(y_data[i])[orders[i]])
        plt.plot(xfinal[i], yfinal[i], "o", label="p=" + label, color=col)

    plt.legend()
    plt.ylabel("avg runtime (s) to decode 1k samples")
    plt.xlabel("code length n")
    plt.grid()
    plt.show()


def runtime_comparison() -> None:
    """Compare the runtime of the original and the heuristic decoding algorithm."""
    plt.rcParams.update({"font.size": 14})
    input_filen = ""
    input_filen2 = ""
    x_data = []
    y_data = []
    pers = []
    x_data2 = []
    y_data2 = []
    pers2 = 0.0
    orders = []

    with Path(input_filen).open(encoding="utf-8") as data_file:
        data = json.load(data_file)
    with Path(input_filen2).open(encoding="utf-8") as data_file2:
        data2 = json.load(data_file2)

    for per in data:
        per_x_data = []
        per_y_data = []

        for c in data[per]:
            per_x_data.append(float(c))
            per_y_data.append(float(data[per][c]) / 1000.0)
        pers.append(float(per))
        x_data.append(per_x_data)
        y_data.append(per_y_data)

    for per in data2:
        for c in data2[per]:
            x_data2.append(float(c))
            y_data2.append(float(data2[per][c]) / 1000.0)
        pers2 = float(per)

    xfinal = []
    yfinal = []
    cols = []
    colors = mcolors.BASE_COLORS

    for i in range(len(x_data)):
        col, _val = colors.popitem()
        if col in {"w", "k"}:
            col, _val = colors.popitem()
            if col in {"w", "k"}:
                col, _val = colors.popitem()
                cols.append(col)
        label = f"{pers[i]:2.2f}"
        orders.append(np.argsort(x_data[i]))
        xfinal.append(np.array(x_data[i])[orders[i]])
        yfinal.append(np.array(y_data[i])[orders[i]])
        plt.plot(xfinal[i], yfinal[i], "o", label="UFH, p=" + label, color=col)
        start = 0
        optimized_parameters, _pcov = opt.curve_fit(lin_fun, xfinal[i][start:], yfinal[i][start:])
        plt.plot(
            xfinal[i][start:],
            lin_fun(xfinal[i][start:], *optimized_parameters),
            "--",
            color=col,
            label=r"$O(n)$",
        )

    # general qlpd decoder data
    label = f"{pers2:2.2f}"
    orders2 = np.argsort(x_data2)
    xfinal2 = np.array(x_data2)[orders2]
    yfinal2 = np.array(y_data2)[orders2]
    plt.plot(xfinal2, yfinal2, "d", label="GD, p=" + label, color="green")
    start = 0
    optimized_parameters2, _pcov = opt.curve_fit(quad_fun, xfinal2[start:], yfinal2[start:])
    plt.plot(
        xfinal2[start:],
        quad_fun(xfinal2[start:], *optimized_parameters2),
        "--",
        color="green",
        label=r"$O(n^{2})$",
    )

    plt.legend()
    plt.ylabel("avg runtime (s) to decode $10^3$ samples")
    plt.xlabel("code length n")
    plt.grid()
    plt.show()
