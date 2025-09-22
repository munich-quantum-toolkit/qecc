# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hypergraph product code construction from Roffe's LDPC library."""

from __future__ import annotations

import ldpc.protograph as pt
import numpy as np
from bposd.hgp import hgp

# med sized HGP code from https://github.com/quantumgizmos/bp_osd

h = np.loadtxt("examples/mkmn_24_6_10.txt").astype(int)
seed_code = np.loadtxt("examples/mkmn_24_6_10.txt").astype(int)
# print(seed_code)
qcode = hgp(seed_code, compute_distance=True)
qcode.canonical_logicals()
qcode.test()
print(qcode.code_params)
print(qcode.hx)
print("hx:")
print(qcode.hx)
np.savetxt(f"./hgp_{qcode.code_params}_hx.txt", qcode.hx, fmt="%d", newline="\n")

# larger code
a1 = pt.array([
    [(0), (11), (7), (12)],
    [(1), (8), (1), (8)],
    [(11), (0), (4), (8)],
    [(6), (2), (4), (12)],
])
H = a1.to_binary(lift_parameter=10)
qcode = hgp(H, H, compute_distance=True)
qcode.test()
np.savetxt(f"./hgp_{qcode.code_params}_hx.txt", qcode.hx, fmt="%d", newline="\n")
