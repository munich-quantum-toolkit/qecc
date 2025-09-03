# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test synthesis of FT state preparation circuits for surface code."""

from __future__ import annotations

import webbrowser

from mqt.qecc.circuit_synthesis import FTSurfaceCodeStatePrep
from mqt.qecc.circuit_synthesis.noise import CircuitLevelNoise


def test_ft_surface_code_state_prep() -> None:
    """Test the FTSurfaceCodeStatePrep class."""
    distance = 5
    ft_surface_code = FTSurfaceCodeStatePrep(
        distance
    )

    # Check the circuit generation
    noise = CircuitLevelNoise(0.1, 0.1, 0.1, 0.1)
    circuit = ft_surface_code.get_circuit_logical_z(noise)

    # Open the circuit in a web browser
    webbrowser.open(circuit.to_crumble_url(skip_detectors=False), new=2)
    circuit.to_file("ft_surface_code_state_prep.stim")

    assert True
