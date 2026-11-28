# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Command line interface for the CC decoder."""

from __future__ import annotations

import argparse

from ..codes.constructions.color_codes.color_code import LatticeType
from . import decoder
from .comparison import tn_decoder


def cli() -> None:
    """Run the CC decoder as cli."""
    parser = argparse.ArgumentParser()
    parser.add_argument("distance", type=int, help="the distance of the code")
    parser.add_argument("error_rate", type=float, help="the error rate")
    parser.add_argument(
        "--type",
        type=str,
        default=LatticeType.HEXAGON.value,
        choices=[lattice.value for lattice in LatticeType],
        help="type of the code lattice. The tn decoder supports only hexagon. Default: hexagon",
    )
    parser.add_argument(
        "--nr_sims",
        type=int,
        default=10000,
        help="the number of simulations to run. Default: 10000",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="the directory to save the results to. Default: ./results",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="maxsat",
        choices=["maxsat", "tn"],
        help="the decoder to use. Default: maxsat",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="z3",
        help=(
            "maxsat solver to use: z3 or the path to an executable. "
            "An executable only gets timed, so the logical error rates are NaN. Default: z3"
        ),
    )

    args = parser.parse_args()
    if args.decoder == "tn" and args.type != LatticeType.HEXAGON:
        parser.error("the tn decoder supports only --type hexagon")

    if args.decoder == "maxsat":
        decoder.run(
            args.type,
            args.distance,
            args.error_rate,
            args.nr_sims,
            args.results_dir,
            args.solver,
        )
    else:
        tn_decoder.run(args.distance, args.error_rate, args.nr_sims, args.results_dir)
