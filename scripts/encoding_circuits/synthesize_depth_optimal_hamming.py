# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from qiskit import qasm2

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import depth_optimal_encoding_circuit

hamming = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
])

logicals = np.array([
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
])

hamming_code = CSSCode(3, hamming, hamming)
hamming_code.Lx = logicals
hamming_code.Lz = logicals

qc = depth_optimal_encoding_circuit(hamming_code, min_depth=5, max_depth=14, min_timeout=300, max_timeout=7200)

qc_str = qasm2.dumps(qc)

print(qc_str)
