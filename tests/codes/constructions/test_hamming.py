# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the CSSCode class."""

from __future__ import annotations

from mqt.qecc.codes import (
    construct_iceberg_code,
    construct_many_hypercube_code,
    construct_quantum_hamming_code,
)


def test_hamming_code() -> None:
    """Test that the Hamming code is constructed as a valid CSS code."""
    code = construct_quantum_hamming_code(3)
    assert code.n == 7
    assert code.k == 1
    assert code.distance == 3


def test_many_hypercube_code_level_1() -> None:
    """Test that the many-hypercube code."""
    code = construct_many_hypercube_code(1)
    assert code.n == 6
    assert code.k == 4
    assert code.distance == 2
    iceberg = construct_iceberg_code(3)
    assert code == iceberg


def test_many_hypercube_code_level_2() -> None:
    """Test that the many-hypercube code."""
    code = construct_many_hypercube_code(2)
    assert code.n == 36
    assert code.k == 16
    assert code.distance == 4
