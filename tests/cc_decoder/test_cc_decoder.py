# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the decoder for the hexagonal color code."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mqt.qecc.cc_decoder import HexagonalColorCode, cli, code_from_string, decoder

from .test_utils import check_and_load_json

if TYPE_CHECKING:
    from collections.abc import Generator

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
def results_dir() -> Generator[str]:
    """Return directory to store results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


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


def test_maxsat_decoder_light_without_switches() -> None:
    """A light that no switch controls must be off for the problem to have a solution."""
    lo = decoder.LightsOut(lights_to_switches={0: [], 1: [0]}, switches_to_lights={0: [1]})
    lo.preconstruct_z3_instance()
    est, converged, _ = lo.solve(lights=[False, True])
    assert converged
    assert est == [1]
    _, converged, _ = lo.solve(lights=[True, True])
    assert not converged


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the call to an external solver."""
    solver = "mock_solver"
    decoder = "maxsat"
    # The solver output goes to the working directory.
    monkeypatch.chdir(tmp_path)

    wcnf_headers: list[str] = []

    def solver_stub(args: list[str], **_: object) -> None:
        # The solver gets the path of a WCNF file, not the problem text.
        wcnf_headers.append(Path(args[1]).read_text(encoding="utf-8").split("\n", maxsplit=1)[0])

    # mock the subprocess.run call in the `solve` method of the `LightsOut` class
    mocker.patch("subprocess.run", side_effect=solver_stub)
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

    assert result is not None
    assert result["distance"] == d3_hexcode.distance
    assert result["p"] == p
    assert len(wcnf_headers) == nr_sims
    assert all(header.startswith("p wcnf") for header in wcnf_headers)
    assert len(result["logical_error_rates"]) == 1
    assert math.isnan(result["logical_error_rates"][0])
    assert math.isnan(result["logical_error_rate_ebs"][0])
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


def test_solve_keeps_lights(code: HexagonalColorCode) -> None:
    """LightsOut.solve does not change the light pattern of the caller."""
    lo = decoder.LightsOut(code.faces_to_qubits, code.qubits_to_faces)
    lo.preconstruct_z3_instance()
    lights = [True, False, False]
    _, converged, _ = lo.solve(lights=lights)
    assert converged
    assert lights == [True, False, False]


def test_simulate_reports_construction_time(d3_hexcode: HexagonalColorCode) -> None:
    """The average construction time is a time, not the success flag of LightsOut.solve."""
    result = decoder.simulate_error_rate(d3_hexcode, 0.05, 20)
    # Building the constraints of one sample of the d=3 code takes far less than one second.
    assert 0.0 < result["avg_constr_time"] < 0.5
    assert result["avg_total_time"] == pytest.approx(result["avg_constr_time"] + result["avg_solve_time"])


def test_simulate_counts_unsolved_samples(d3_hexcode: HexagonalColorCode, mocker: pytest_mock.MockerFixture) -> None:
    """Samples without a valid solution are counted."""
    mocker.patch.object(decoder.LightsOut, "solve", return_value=([0] * d3_hexcode.n, False, 0.0))
    result = decoder.simulate_error_rate(d3_hexcode, 0.05, 5)
    assert result["nr_unsolved"] == 5


@pytest.mark.parametrize(
    "args",
    [
        ["--decoder", "tn", "--type", "square_octagon"],
        ["--decoder", "unknown"],
        ["--type", "unknown"],
    ],
)
def test_cli_rejects_invalid_options(
    args: list[str], monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
) -> None:
    """The CLI stops with a usage error for unknown or unsupported options and runs no decoder."""
    maxsat_run = mocker.patch.object(cli.decoder, "run")
    tn_run = mocker.patch.object(cli.tn_decoder, "run")
    monkeypatch.setattr(sys, "argv", ["mqt.qecc.cc-decoder", "3", "0.1", *args])
    with pytest.raises(SystemExit) as exc_info:
        cli.cli()
    assert exc_info.value.code == 2
    maxsat_run.assert_not_called()
    tn_run.assert_not_called()
