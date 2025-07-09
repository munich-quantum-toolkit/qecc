# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Command line interface for the CC decoder."""

from __future__ import annotations

import argparse

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
        default="hexagon",
        help="type of the code lattice (hexagon or square_octagon). Default: hexagon",
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
        help="the decoder to use (maxsat or tn). Default: maxsat",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="z3",
        help="maxsat solver to use (path to a executable). Default: z3",
    )

    args = parser.parse_args()

    if args.decoder == "maxsat":
        decoder.run(
            args.type,
            args.distance,
            args.error_rate,
            args.nr_sims,
            args.results_dir,
            args.solver,
        )
    elif args.decoder == "tn":
        tn_decoder.run(args.distance, args.error_rate, args.nr_sims, args.results_dir)
    else:
        msg = f"Unknown decoder {args.decoder}. Please choose either maxsat or tn."
        raise ValueError(msg)
