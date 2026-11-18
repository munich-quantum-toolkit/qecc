# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Search for fault-tolerant CSS state preparation circuits over the trade-off of operations and depth.

:func:`~mqt.qecc.circuit_synthesis.ft_search.api.synthesize_ft_prep` runs the search on a code and returns its front of circuits.
"""

from __future__ import annotations

from .api import FrontCircuit, synthesize_ft_prep

__all__ = ["FrontCircuit", "synthesize_ft_prep"]
