# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the circuits that NoisyDFTStatePrepSimulator builds.

qsample is an optional dependency, so these tests replace it with a small module that records the circuits.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections import UserList
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import DeterministicVerification

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType


class _Circuit(UserList[dict[str, Any]]):
    """Record the operations of a qsample circuit."""

    def __init__(self, ops: list[dict[str, Any]], noisy: bool = True) -> None:
        super().__init__(ops)
        self.noisy = noisy


class _Protocol:
    """Record the nodes of a qsample protocol."""

    def __init__(self, **_kwargs: object) -> None:
        self.nodes: dict[str, _Circuit] = {}
        self.check_functions: dict[str, object] = {}

    def add_node(self, name: str, circuit: _Circuit) -> None:
        self.nodes[name] = circuit

    def add_edge(self, *_args: object, **_kwargs: object) -> None:
        pass


@pytest.fixture
def simulation_det(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import simulation_det with a stand-in for qsample.

    After the test, the fixture removes the imported module, so that later imports do not use the stand-in.
    """
    qsample = types.ModuleType("qsample")
    qsample.__dict__.update(
        Circuit=_Circuit, Protocol=_Protocol, ErrorModel=object, noise=types.SimpleNamespace(E1_1=object)
    )
    monkeypatch.setitem(sys.modules, "qsample", qsample)
    module_name = "mqt.qecc.circuit_synthesis.simulation_det"
    original_module = sys.modules.pop(module_name, None)
    package = importlib.import_module("mqt.qecc.circuit_synthesis")
    monkeypatch.setattr(package, "simulation_det", original_module, raising=False)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


@pytest.mark.parametrize("zero_state", [True, False])
def test_decoding_circuit_measures_every_stabilizer(simulation_det: ModuleType, zero_state: bool) -> None:
    """The decoding circuit measures each stabilizer of the check matrix with its own ancilla."""
    code = CSSCode.from_code_name("Steane")
    no_verification = DeterministicVerification([], {})
    simulator = simulation_det.NoisyDFTStatePrepSimulator(
        QuantumCircuit(code.n), (no_verification, no_verification), code, zero_state=zero_state
    )
    decoding_circuit = simulator.protocol.nodes["DEC"]
    matrix = code.Hz if zero_state else code.Hx
    ancillas = list(range(code.n, code.n + matrix.shape[0]))

    measured = [qubit for op in decoding_circuit if "measure" in op for qubit in sorted(op["measure"])]
    assert measured == ancillas + list(range(code.n))

    cnots = [cnot for op in decoding_circuit if "CNOT" in op for cnot in op["CNOT"]]
    for ancilla, stabilizer in zip(ancillas, matrix, strict=True):
        support = [int(q) for q in np.flatnonzero(stabilizer)]
        if zero_state:
            assert sorted(q for q, a in cnots if a == ancilla) == support
        else:
            assert sorted(q for a, q in cnots if a == ancilla) == support
