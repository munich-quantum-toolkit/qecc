# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Toric code construction."""

# using tools from https://github.com/quantumgizmos/bp_osd
from __future__ import annotations

import numpy as np
from bposd.hgp import hgp
from ldpc.codes import ring_code

# toric code
for i in range(2, 100):
    h1 = ring_code(i)
    h2 = ring_code(i)

    qcode = hgp(h1, h2, compute_distance=True)
    if qcode.test:  # to make sure it is a valid css code
        np.savetxt(f"./toric_{qcode.code_params}_hz.txt", qcode.hz, fmt="%d", newline="\n")
