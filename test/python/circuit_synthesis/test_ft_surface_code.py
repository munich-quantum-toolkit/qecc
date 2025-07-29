"""Test synthesis of FT state preparation circuits for surface code."""

from __future__ import annotations

import webbrowser

from mqt.qecc.circuit_synthesis import FTSurfaceCodeStatePrep


def test_ft_surface_code_state_prep() -> None:
    """Test the FTSurfaceCodeStatePrep class."""
    distance = 5
    ft_surface_code = FTSurfaceCodeStatePrep(
        distance, zero_state=True, kwargs={"horizontal_cx_direction": "right", "vertical_cx_direction": "straight"}
    )

    # Check the circuit generation
    circuit = ft_surface_code.get_circuit()

    # Open the circuit in a web browser
    webbrowser.open(circuit.to_crumble_url(), new=2)

    assert True
