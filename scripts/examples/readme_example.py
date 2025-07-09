# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example of using the UF heuristic decoder."""

from __future__ import annotations

import numpy as np

from mqt import qecc

code = qecc.Code("/path/to/Hx", "path/to/Hz")
decoder = qecc.UFHeuristic()
decoder.set_code(code)
x_err = qecc.sample_iid_pauli_err(code.N, 0.05)
decoder.decode(code.get_x_syndrome(x_err))
result = decoder.result
print(result)
residual_err = np.array(x_err) ^ np.array(result.estimate)
print(code.is_x_stabilizer(residual_err))
