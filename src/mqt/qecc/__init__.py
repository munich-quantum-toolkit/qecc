# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT QECC library.

This file is part of the MQT QECC library released under the MIT license.
See README.md or go to https://github.com/cda-tum/qecc for more information.
"""

from __future__ import annotations

from ._version import __version__
from .analog_information_decoding.simulators.analog_tannergraph_decoding import AnalogTannergraphDecoder, AtdSimulator
from .analog_information_decoding.simulators.quasi_single_shot_v2 import QssSimulator
from .codes import CSSCode, StabilizerCode
from .equivalence_checking import (
    are_local_clifford_equivalent,
    are_permutation_equivalent,
    is_local_clifford_equivalent_to_css,
)

__all__ = [
    "AnalogTannergraphDecoder",
    "AtdSimulator",
    "CSSCode",
    "QssSimulator",
    "StabilizerCode",
    "__version__",
    "are_local_clifford_equivalent",
    "are_permutation_equivalent",
    "is_local_clifford_equivalent_to_css",
]
