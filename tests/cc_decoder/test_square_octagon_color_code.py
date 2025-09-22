# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the square octagon color code. Created by Peter-Jan Derks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc.cc_decoder import SquareOctagonColorCode

from .test_utils import check_and_load_json

if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner


@pytest.fixture
def d3_so_code() -> SquareOctagonColorCode:
    """Distance of the hexagonal color code."""
    return SquareOctagonColorCode(distance=3)


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


@pytest.mark.parametrize("code", [SquareOctagonColorCode(distance=d) for d in range(3, 23, 2)])
def test_number_of_qubits(code: SquareOctagonColorCode) -> None:
    """Test the number of qubits for larger distances."""
    assert len(code.data_qubits) == 1 / 2 * code.distance**2 + code.distance - 1 / 2
    assert len(code.ancilla_qubits) == (1 / 2 * code.distance**2 + code.distance - 1 / 2) // 2


def test_d3(d3_so_code: SquareOctagonColorCode) -> None:
    """Test coordinates of qubits for distance 3."""
    assert d3_so_code.data_qubits == {
        (0, 1),
        (2, 1),
        (6, 1),
        (3, 2),
        (5, 2),
        (3, 4),
        (5, 4),
    }
    assert d3_so_code.ancilla_qubits == {(4, 0), (1, 3), (4, 3)}


def test_h(d3_so_code: SquareOctagonColorCode) -> None:
    """Test the parity check matrix for distance 3."""
    assert np.array_equal(
        d3_so_code.H,
        np.array([[0, 0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 1, 0]]),
    )


@pytest.mark.xdist_group(name="check_and_load_json_user")
def test_z3_solver(
    script_runner: ScriptRunner,
    d3_so_code: SquareOctagonColorCode,
    p: float,
    nr_sims: int,
    results_dir: str,
) -> None:
    """Test the Z3 solver."""
    d = d3_so_code.distance
    ret = script_runner.run([
        "mqt.qecc.cc-decoder",
        "--type",
        "square_octagon",
        str(d),
        str(p),
        "--nr_sims",
        str(nr_sims),
        "--results_dir",
        results_dir,
    ])
    assert ret.success
    assert not ret.stderr

    result = check_and_load_json(
        f"./code={d3_so_code.lattice_type},distance={d3_so_code.distance},p={round(p, 4)},solver=z3.json",
        results_dir,
    )
    assert result is not None
    assert result["distance"] == d3_so_code.distance
    assert result["p"] == p
    assert result["logical_error_rates"] is not None
    assert result["logical_error_rate_ebs"] is not None
    assert result["min_wts_logical_err"] is not None
    assert result["preconstr_time"] > 0.0
    assert result["avg_constr_time"] > 0.0
    assert result["avg_solve_time"] > 0.0
    assert result["avg_total_time"] > 0.0
