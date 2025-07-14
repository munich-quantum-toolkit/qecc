# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the functions in misc."""

from __future__ import annotations

import mqt.qecc.co3 as co


def _split_layers_cnot(circuit: list[tuple[int, int] | int]) -> list[list[tuple[int, int] | int]]:
    """Split the circuit into initial layers.

    The input can also have mere ints, but this is only for mypy. This helper function only assumes tuple[int,int] in the circuit.
    """
    result = []
    current_group = []
    seen = set()

    for a, b in circuit:
        if a in seen or b in seen:
            result.append(current_group)
            current_group = []
            seen = set()
        current_group.append((a, b))
        seen.update([a, b])

    if current_group:
        result.append(current_group)

    return result


def test_generate_max_parallel_circuit():
    """Tests generate_max_parallel_circuit from misc."""
    q = 4
    min_depth = q * 2
    circuit = co.generate_max_parallel_circuit(q, min_depth)
    split = _split_layers_cnot(circuit)
    for el in split:
        assert len(el) == q // 2

    q = 10
    min_depth = q * 4
    circuit = co.generate_max_parallel_circuit(q, min_depth)
    split = _split_layers_cnot(circuit)
    for el in split:
        assert len(el) == q // 2

    q = 22
    min_depth = q * 4
    circuit = co.generate_max_parallel_circuit(q, min_depth)
    split = _split_layers_cnot(circuit)
    for el in split:
        assert len(el) == q // 2


def test_generate_min_parallel_circuit():
    """Tests generate_min_parallel_circuit from misc."""
    q = 10
    min_depth = q * 3

    for layer_size in range(2, 6):
        circuit = co.generate_min_parallel_circuit(q, min_depth, layer_size)
        split = _split_layers_cnot(circuit)
        for el in split:
            assert len(el) == layer_size
