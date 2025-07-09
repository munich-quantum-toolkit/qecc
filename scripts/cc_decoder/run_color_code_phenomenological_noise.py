# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This script is used to run the color code phenomenological noise simulation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sinter

from mqt.qecc.cc_decoder.stim_interface.color_code_stim import gen_pcm_and_logical, gen_stim_circuit_memory_experiment
from mqt.qecc.cc_decoder.stim_interface.max_sat_sinter_decoder import sinter_decoders


def generate_example_tasks() -> Any:  # noqa: ANN401
    """Generate example stim tasks."""
    for p in np.arange(0.001, 0.005, 0.002):
        for d in [3]:
            pcm, l_op = gen_pcm_and_logical(d)
            cc_circuit = gen_stim_circuit_memory_experiment(pcm, l_op, d, p)
            yield sinter.Task(
                circuit=cc_circuit,
                detector_error_model=cc_circuit.detector_error_model(decompose_errors=False),
                json_metadata={
                    "p": p,
                    "d": d,
                    "rounds": d,
                },
            )


def main() -> None:
    """Run the simulation."""
    samples = sinter.collect(
        num_workers=10,
        max_shots=10_000,
        max_errors=500,
        tasks=generate_example_tasks(),
        decoders=["maxsat"],
        custom_decoders=sinter_decoders(),
        print_progress=True,
        save_resume_filepath="pseudothreshold_plot.csv",
    )

    # Print samples as CSV data.
    for _sample in samples:
        pass

    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"Color Code d={stat.json_metadata['d']} dec={stat.decoder}",
        x_func=lambda stat: stat.json_metadata["p"],
        failure_units_per_shot_func=lambda stats: stats.json_metadata["rounds"],
        filter_func=lambda stat: stat.json_metadata["d"] > 2,
    )
    x_s = np.linspace(0.001, 0.029, 1000000)
    y_s = np.linspace(0.001, 0.029, 1000000)
    ax.set_yscale("log")
    ax.plot(x_s, y_s, "k-", alpha=0.75, zorder=0, label="x=y")

    ax.grid()
    ax.set_title("Phenomenological Noise")
    ax.set_ylabel("Logical Error Probability (per shot)")
    ax.set_xlabel("Physical Error Rate")
    ax.legend()

    # Save to file and also open in a window.
    fig.savefig("plot.png")
    plt.show()


if __name__ == "__main__":
    """ Run the main function """
    main()
