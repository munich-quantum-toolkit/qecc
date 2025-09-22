# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Visualize the word error rate plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wer() -> None:
    """Plot the word error rate."""
    plt.rcParams.update({"font.size": 14})
    input_filename = ""
    input_filename2 = ""
    input_filename3 = ""
    _fig, ax = plt.subplots(1)

    with Path(input_filename).open(encoding="utf-8") as f:
        data = json.load(f)
    x_data = []
    y_data = []
    for key in data:
        x_data.append(float(key))
        y_data.append(data[key])

    order = np.argsort(x_data)
    x_dataf = np.array(x_data)[order]
    y_dataf = np.array(y_data)[order]

    with Path(input_filename2).open(encoding="utf-8") as f:
        data2 = json.load(f)
    x_data2 = []
    y_data2 = []
    for key2 in data2:
        x_data2.append(float(key2))
        y_data2.append(data2[key2])

    order2 = np.argsort(x_data2)
    x_data2f = np.array(x_data2)[order2]
    y_data2f = np.array(y_data2)[order2]

    with Path(input_filename3).open(encoding="utf-8") as f:
        data3 = json.load(f)
    x_data3 = []
    y_data3 = []
    for key3 in data3:
        x_data3.append(float(key3))
        y_data3.append(data3[key3])

    order3 = np.argsort(x_data3)
    x_data3f = np.array(x_data3)[order3]
    y_data3f = np.array(y_data3)[order3]

    ax.plot(x_dataf, y_dataf, "-x", label="heuristic SSG", color="b")
    ax.plot(x_data2f, y_data2f, "-o", label="GD", color="g")
    ax.plot(x_data3f, y_data3f, "-x", label="heuristic AG", color="r")

    ax.set_xlabel("physical X-error rate")
    ax.set_ylabel("WER")
    ax.legend()
    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()

    handles = [handles[1], handles[2], handles[0]]
    labels = [labels[1], labels[2], labels[0]]

    ax.legend(handles, labels, loc=4)
    plt.show()


def wer_comp() -> None:
    """Visualize WER comparison."""
    plt.rcParams.update({"font.size": 15})
    input_filename = ""
    input_filename2 = ""
    _fig, ax = plt.subplots(1)

    with Path(input_filename).open(encoding="utf-8") as f:
        data = json.load(f)
    x_data = []
    y_data = []
    for key in data:
        x_data.append(float(key))
        y_data.append(data[key])

    with Path(input_filename2).open(encoding="utf-8") as f:
        data2 = json.load(f)
    x_data2 = []
    y_data2 = []
    for key2 in data2:
        x_data2.append(float(key2))
        y_data2.append(data2[key2])

    ax.plot(x_data, y_data, "-d", label="heuristic", color="b")
    ax.plot(x_data2, y_data2, "-o", label="decoder", color="g")
    print(y_data)
    print(y_data2)

    ax.xlabel("physical X-error rate")
    ax.ylabel("WER")
    ax.legend()
    ax.grid()
    ax.xscale("log")
    ax.yscale("log")
    ax.gca().set_position([0, 0, 1, 1])
    ax.savefig("werComp.svg")
    handles, labels = ax.get_legend_handles_labels()

    handles = [handles[2], handles[1], handles[0]]
    labels = [labels[2], labels[1], labels[0]]

    ax.legend(handles, labels, loc=2)
    plt.show()
