# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the decoder for the hexagonal color code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mqt.qecc.cc_decoder import HexagonalColorCode, code_from_string, decoder

from .test_utils import check_and_load_json

if TYPE_CHECKING:
    import pytest_mock
    from pytest_console_scripts import ScriptRunner


@pytest.fixture
def distance() -> int:
    """Return distance of the hexagonal color code."""
    return 3


@pytest.fixture
def p() -> float:
    """Return error rate."""
    return 0.1


@pytest.fixture
def nr_sims() -> int:
    """Return number of simulations."""
    return 1


@pytest.fixture
def results_dir() -> str:
    """Return directory to store results."""
    return "./results/test/"


@pytest.fixture
def code(distance: int) -> HexagonalColorCode:
    """Hexagonal color code."""
    return HexagonalColorCode(distance=distance)


@pytest.fixture
def d3_hexcode() -> HexagonalColorCode:
    """Distance of the hexagonal color code."""
    return HexagonalColorCode(distance=3)


def test_hex_layout(code: HexagonalColorCode, distance: int) -> None:
    """Test the construction of the hexagonal color code."""
    assert len(code.data_qubits) == 7
    code.compute_logical()
    assert len(code.L) == 1
    assert code.H.shape == (distance, 7)
    assert code.distance == distance


def test_lo(code: HexagonalColorCode, distance: int) -> None:
    """Test the construction of the lights out problem."""
    lo = decoder.LightsOut(code.faces_to_qubits, code.qubits_to_faces)

    assert len(lo.lights_to_switches) == distance
    assert len(lo.lights_to_switches[0]) == 4
    assert lo.optimizer is not None


def test_maxsat_decoder(code: HexagonalColorCode) -> None:
    """Test the MaxSAT decoder on a small example."""
    lo = decoder.LightsOut(code.faces_to_qubits, code.qubits_to_faces)
    lo.preconstruct_z3_instance()
    lights = [True, False, False]
    est, _, _ = lo.solve(lights=lights)
    assert len(est) == 7
    assert est[1] == 1


def test_simulate(d3_hexcode: HexagonalColorCode, p: float, nr_sims: int) -> None:
    """Test the general simulation function."""
    res = decoder.simulate_error_rate(d3_hexcode, p, nr_sims)
    assert res is not None
    assert res["distance"] == d3_hexcode.distance
    assert res["p"] == p


@pytest.mark.xdist_group(name="check_and_load_json_user")
def test_z3_solver(
    script_runner: ScriptRunner,
    d3_hexcode: HexagonalColorCode,
    p: float,
    nr_sims: int,
    results_dir: str,
) -> None:
    """Test the Z3 solver."""
    ret = script_runner.run([
        "mqt.qecc.cc-decoder",
        str(d3_hexcode.distance),
        str(p),
        "--nr_sims",
        str(nr_sims),
        "--results_dir",
        results_dir,
    ])
    assert ret.success
    assert not ret.stderr

    result = check_and_load_json(
        f"./code={d3_hexcode.lattice_type},distance={d3_hexcode.distance},p={round(p, 4)},solver=z3.json",
        results_dir,
    )
    assert result is not None
    assert result["distance"] == d3_hexcode.distance
    assert result["p"] == p
    assert result["logical_error_rates"] is not None
    assert result["logical_error_rate_ebs"] is not None
    assert result["min_wts_logical_err"] is not None
    assert result["preconstr_time"] > 0.0
    assert result["avg_constr_time"] > 0.0
    assert result["avg_solve_time"] > 0.0
    assert result["avg_total_time"] > 0.0


@pytest.mark.xdist_group(name="check_and_load_json_user")
def test_external_solver(
    d3_hexcode: HexagonalColorCode,
    script_runner: ScriptRunner,
    mocker: pytest_mock.MockerFixture,
    p: float,
    nr_sims: int,
    results_dir: str,
) -> None:
    """Mock the call to an external solver."""
    solver = "mock_solver"
    decoder = "maxsat"

    # mock the subprocess.run call in the `solve` method of the `LightsOut` class
    mocker.patch("subprocess.run", return_value=None)
    ret = script_runner.run([
        "mqt.qecc.cc-decoder",
        str(d3_hexcode.distance),
        str(p),
        "--nr_sims",
        str(nr_sims),
        "--results_dir",
        results_dir,
        "--solver",
        solver,
        "--decoder",
        decoder,
    ])
    assert ret.success
    assert not ret.stderr

    result = check_and_load_json(
        f"./code={d3_hexcode.lattice_type},distance={d3_hexcode.distance},p={round(p, 4)},solver={solver}.json",
        results_dir,
    )

    solver_output_file = Path(f"solver-out_{solver}.txt")
    if solver_output_file.exists():
        solver_output_file.unlink()

    assert result is not None
    assert result["distance"] == d3_hexcode.distance
    assert result["p"] == p
    assert result["logical_error_rates"] == [0.0]
    assert result["logical_error_rate_ebs"] == [0.0]
    assert result["min_wts_logical_err"] == [-1]
    assert result["preconstr_time"] > 0.0
    assert result["avg_constr_time"] > 0.0
    assert result["avg_solve_time"] > 0.0
    assert result["avg_total_time"] > 0.0


@pytest.mark.xdist_group(name="check_and_load_json_user")
def test_tn_decoder(script_runner: ScriptRunner, distance: int, p: float, nr_sims: int, results_dir: str) -> None:
    """Test the TN decoder."""
    decoder = "tn"

    ret = script_runner.run([
        "mqt.qecc.cc-decoder",
        str(distance),
        str(p),
        "--nr_sims",
        str(nr_sims),
        "--results_dir",
        results_dir,
        "--decoder",
        decoder,
    ])
    assert ret.success
    assert not ret.stderr

    result = check_and_load_json(f"distance={distance},p={round(p, 4)}.json", results_dir)
    assert result is not None
    assert result["n_k_d"][-1] == distance
    assert result["error_probability"] == p
    assert result["n_run"] == nr_sims
    assert result["wall_time"] > 0.0


def test_get_code_from_str() -> None:
    """Test the construction of a color code from a string."""
    assert code_from_string(lattice_type="hexagon", distance=3) == HexagonalColorCode(distance=3)


@pytest.mark.xdist_group(name="check_and_load_json_user")
def test_scenario_with_logical_errors(
    script_runner: ScriptRunner,
    d3_hexcode: HexagonalColorCode,
    results_dir: str,
) -> None:
    """Test the Z3 solver."""
    ret = script_runner.run([
        "mqt.qecc.cc-decoder",
        str(d3_hexcode.distance),
        "0.2",
        "--nr_sims",
        "50",
        "--results_dir",
        results_dir,
    ])
    assert ret.success
    assert not ret.stderr

    result = check_and_load_json(
        f"./code={d3_hexcode.lattice_type},distance={d3_hexcode.distance},p=0.2,solver=z3.json",
        results_dir,
    )
    assert result is not None
    assert result["distance"] == d3_hexcode.distance
    assert result["logical_error_rates"] is not None
    assert any(result["logical_error_rates"])
    assert result["logical_error_rate_ebs"] is not None
    assert result["min_wts_logical_err"] is not None
    assert result["preconstr_time"] > 0.0
    assert result["avg_constr_time"] > 0.0
    assert result["avg_solve_time"] > 0.0
    assert result["avg_total_time"] > 0.0
