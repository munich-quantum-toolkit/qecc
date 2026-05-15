# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for synthesizing circuits."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import multiprocess
import numpy as np
import z3
from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumCircuit

from .circuits import CNOTCircuit

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    from qiskit.circuit import AncillaQubit, Clbit, Qubit

    from ..codes.pauli import CheckMatrix


logger = logging.getLogger(__name__)


def run_with_timeout(func: Callable[[Any], Any], *args: Any, timeout: int = 10) -> Any | str | None:  # noqa: ANN401
    """Run a function with a timeout.

    If the function does not complete within the timeout, return None.

    Args:
        func: The function to run.
        args: The arguments to pass to the function.
        timeout: The maximum time to allow the function to run for in seconds.
    """
    manager = multiprocess.Manager()
    return_list = manager.list()
    p = multiprocess.Process(target=lambda: return_list.append(func(*args)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return "timeout"
    return return_list[0]


def iterative_search_with_timeout(
    fun: Callable[[int], QuantumCircuit],
    min_param: int,
    max_param: int,
    min_timeout: int,
    max_timeout: int,
    param_factor: float = 2,
    timeout_factor: float = 2,
) -> tuple[QuantumCircuit | None, int] | None:
    """Geometrically increases the parameter and timeout until a result is found or the maximum timeout is reached.

    Args:
        fun: function to run with increasing parameters and timeouts
        min_param: minimum parameter to start with
        max_param: maximum parameter to reach
        min_timeout: minimum timeout to start with
        max_timeout: maximum timeout to reach
        param_factor: factor to increase the parameter by at each iteration
        timeout_factor: factor to increase the timeout by at each iteration
    """
    curr_timeout = min_timeout
    curr_param = min_param
    while curr_timeout <= max_timeout:
        while curr_param <= max_param:
            logger.info(f"Running iterative search with param={curr_param} and timeout={curr_timeout}")
            res = run_with_timeout(fun, curr_param, timeout=curr_timeout)
            if res is not None and (not isinstance(res, str) or res != "timeout"):
                return res, curr_param
            if curr_param == max_param:
                break

            curr_param = int(curr_param * param_factor)
            curr_param = min(curr_param, max_param)

        curr_timeout = int(curr_timeout * timeout_factor)
        curr_param = min_param
    return None, max_param


def build_css_encoder_from_cnot_list(
    checks: CheckMatrix, logicals: CheckMatrix, cnots: list[tuple[int, int]]
) -> CNOTCircuit:
    """Build a CSS encoding circuit from a list of CNOTs, given the stabilizers and logicals.

    Args:
        checks: The stabilizer check matrix of the CSS code.
        logicals: The logical operator matrix of the CSS code.
        cnots: The list of CNOT operations to apply.

    Returns:
        The synthesized encoding circuit.
    """
    if checks.type != logicals.type:
        msg = "Checks and logicals must be of the same type."
        raise ValueError(msg)

    check_matrix = checks.matrix
    logical_matrix = logicals.matrix
    n = checks.num_qubits()
    encoding_qubits = np.where(logical_matrix.sum(axis=0) != 0)[0].tolist()
    if checks.type == "X":
        hadamards = np.where(check_matrix.sum(axis=0) != 0)[0]
    else:
        hadamards = np.where(check_matrix.sum(axis=0) == 0)[0]

    hadamards = np.setdiff1d(hadamards, encoding_qubits)
    non_hadamards = [i for i in range(n) if i not in hadamards and i not in encoding_qubits]
    return CNOTCircuit.from_cnot_list(cnots, initialize_z=non_hadamards, initialize_x=hadamards, inputs=encoding_qubits)


def build_css_circuit_from_cnot_list(n: int, cnots: list[tuple[int, int]], hadamards: list[int]) -> QuantumCircuit:
    """Build a quantum circuit consisting of Hadamards followed by a layer of CNOTs from a list of CNOTs and a list of checks.

    Args:
        n: Number of qubits in the circuit.
        cnots: List of CNOTs to apply. Each CNOT is a tuple of the form (control, target).
        hadamards: List of qubits to apply Hadamards to.

    Returns:
        The quantum circuit.
    """
    circ = QuantumCircuit(n)
    circ.h(hadamards)
    for i, j in cnots:
        circ.cx(i, j)
    return circ


def symbolic_vector_eq(v1: npt.NDArray[np.bool_] | list[z3.BoolRef], v2: npt.NDArray[np.bool_]) -> z3.BoolRef:
    """Return assertion that two symbolic vectors should be equal."""
    if len(v1) != len(v2):
        msg = "Vectors must have the same length for equality check."
        raise ValueError(msg)

    # map all numpy bools to Python bools, otherwise z3 will not be able to handle them
    v1 = np.array([bool(v) if isinstance(v, (bool, np.bool_)) else v for v in v1], dtype=object)
    v2 = np.array([bool(v) if isinstance(v, (bool, np.bool_)) else v for v in v2], dtype=object)

    constraints = [False for _ in v1]
    for i in range(len(v1)):
        # If one of the elements is a bool, we can simplify the expression
        v1_i_is_bool = isinstance(v1[i], (bool, np.bool_))
        v2_i_is_bool = isinstance(v2[i], (bool, np.bool_))
        if v1_i_is_bool:
            v1[i] = bool(v1[i])
            if v1[i]:
                constraints[i] = v2[i]
            else:
                constraints[i] = z3.Not(v2[i]) if not v2_i_is_bool else not v2[i]

        elif v2_i_is_bool:
            v2[i] = bool(v2[i])
            if v2[i]:
                constraints[i] = v1[i]
            else:
                constraints[i] = z3.Not(v1[i])
        else:
            constraints[i] = v1[i] == v2[i]
    return z3.And(constraints)


def odd_overlap(v_sym: npt.NDArray[np.bool_], v_con: npt.NDArray[np.int8]) -> z3.BoolRef:
    """Return True if the overlap of symbolic vector with constant vector is odd."""
    if np.array_equal(v_con, np.zeros(len(v_con), dtype=np.int8)):
        return z3.BoolVal(False)

    constraint = False
    for i, c in enumerate(v_con):
        if c != 1:
            continue
        constraint = z3.Xor(constraint, v_sym[i])
    return constraint


def symbolic_scalar_mult(v: npt.NDArray[np.int8], a: z3.BoolRef | bool) -> npt.NDArray[np.bool_]:
    """Multiply a concrete vector by a symbolic scalar."""
    return np.array([a if s == 1 else False for s in v])


def symbolic_vector_add(v1: npt.NDArray[np.bool_], v2: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Add two symbolic vectors."""
    v_new = [False for _ in range(len(v1))]
    for i in range(len(v1)):
        # If one of the elements is a bool, we can simplify the expression
        v1_i_is_bool = isinstance(v1[i], (bool, np.bool_))
        v2_i_is_bool = isinstance(v2[i], (bool, np.bool_))
        if v1_i_is_bool:
            v1[i] = bool(v1[i])
            if v1[i]:
                v_new[i] = z3.Not(v2[i]) if not v2_i_is_bool else not v2[i]
            else:
                v_new[i] = v2[i]

        elif v2_i_is_bool:
            v2[i] = bool(v2[i])
            if v2[i]:
                v_new[i] = z3.Not(v1[i])
            else:
                v_new[i] = v1[i]

        elif bool(v1[i] == v2[i]):
            v_new[i] = False
        else:
            v_new[i] = z3.Xor(v1[i], v2[i])

    return np.array(v_new)


def _ancilla_cnot(qc: QuantumCircuit, qubit: Qubit | AncillaQubit, ancilla: AncillaQubit, z_measurement: bool) -> None:
    if z_measurement:
        qc.cx(qubit, ancilla)
    else:
        qc.cx(ancilla, qubit)


def _flag_measure(qc: QuantumCircuit, flag: AncillaQubit, meas_bit: Clbit, z_measurement: bool) -> None:
    if z_measurement:
        qc.h(flag)
    qc.measure(flag, meas_bit)


def _flag_reset(qc: QuantumCircuit, flag: AncillaQubit, z_measurement: bool) -> None:
    qc.reset(flag)
    if z_measurement:
        qc.h(flag)


def _flag_init(qc: QuantumCircuit, flag: AncillaQubit, z_measurement: bool) -> None:
    if z_measurement:
        qc.h(flag)


def measure_stab_unflagged(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a stabilizer without flags. The measurement is done in place.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: The qubits to measure.
        ancilla: The ancilla qubit to use for the measurement.
        measurement_bit: The classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    if not z_measurement:
        qc.h(ancilla)
        qc.cx([ancilla] * len(stab), stab)
        qc.h(ancilla)
    else:
        qc.cx(stab, [ancilla] * len(stab))
    qc.measure(ancilla, measurement_bit)


def measure_flagged(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    t: int,
    z_measurement: bool = True,
) -> None:
    """Measure a w-flagged stabilizer.

    The measurement is done in place.

    Args:
        Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        t: The number of errors to protect against.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    w = len(stab)
    if w < 3:
        measure_stab_unflagged(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    if t == 1:
        measure_one_flagged(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    if w == 4 and t >= 2:
        measure_two_flagged_4(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    if w in {5, 6}:
        weight_5 = w == 5
        if t == 2:
            measure_two_flagged_5_or_6(qc, stab, ancilla, measurement_bit, z_measurement, weight_5)
            return
        measure_w_flagged_5_or_6(qc, stab, ancilla, measurement_bit, z_measurement, weight_5)
        return

    if w in {7, 8}:
        weight_7 = w == 7
        if t == 2:
            measure_two_flagged_7_or_8(qc, stab, ancilla, measurement_bit, z_measurement, weight_7)
            return
        if t == 3:
            measure_three_flagged_7_or_8(qc, stab, ancilla, measurement_bit, z_measurement, weight_7)
            return

    if w in {11, 12}:
        weight_11 = w == 11
        if t == 2:
            measure_two_flagged_11_or_12(qc, stab, ancilla, measurement_bit, z_measurement, weight_11)
        if t == 3:
            measure_three_flagged_12(qc, stab, ancilla, measurement_bit, z_measurement, weight_11)
        return

    if t == 2:
        measure_two_flagged_general(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    msg = f"Flagged measurement for w={w} and t={t} not implemented."
    raise NotImplementedError(msg)


def measure_one_flagged(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a 1-flagged stabilizer.

    In this case only one flag is required.
    """
    flag_reg = AncillaRegister(1)
    meas_reg = ClassicalRegister(1)
    qc.add_register(flag_reg)
    qc.add_register(meas_reg)
    flag = flag_reg[0]
    flag_meas = meas_reg[0]
    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)
    _flag_init(qc, flag, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)

    for q in stab[1:-1]:
        _ancilla_cnot(qc, q, ancilla, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)
    _flag_measure(qc, flag, flag_meas, z_measurement)

    _ancilla_cnot(qc, stab[-1], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_general(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a 2-flagged stabilizer using the scheme of https://arxiv.org/abs/1708.02246 (page 13).

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    n_flags = (len(stab) + 1) // 2 - 1
    flag_reg = AncillaRegister(n_flags)
    meas_reg = ClassicalRegister(n_flags)

    qc.add_register(flag_reg)
    qc.add_register(meas_reg)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag_reg[0], z_measurement)
    _ancilla_cnot(qc, flag_reg[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)
    _flag_init(qc, flag_reg[1], z_measurement)
    _ancilla_cnot(qc, flag_reg[1], ancilla, z_measurement)

    cnots = 2
    flags = 2
    for q in stab[2:-2]:
        _ancilla_cnot(qc, q, ancilla, z_measurement)
        cnots += 1
        if cnots % 2 == 0 and cnots < len(stab) - 2:
            _flag_init(qc, flag_reg[flags], z_measurement)
            _ancilla_cnot(qc, flag_reg[flags], ancilla, z_measurement)
        if cnots >= 7 and cnots % 2 == 1:
            _ancilla_cnot(qc, flag_reg[flags - 2], ancilla, z_measurement)
            _flag_measure(qc, flag_reg[flags - 2], meas_reg[flags - 2], z_measurement)
        if cnots % 2 == 0 and cnots < len(stab) - 2:
            flags += 1

    _ancilla_cnot(qc, flag_reg[0], ancilla, z_measurement)
    _flag_measure(qc, flag_reg[0], meas_reg[0], z_measurement)

    _ancilla_cnot(qc, stab[-2], ancilla, z_measurement)

    cnots += 1
    if cnots >= 7 and cnots % 2 == 1:
        _ancilla_cnot(qc, flag_reg[flags - 1], ancilla, z_measurement)
        _flag_measure(qc, flag_reg[flags - 1], meas_reg[flags - 1], z_measurement)

    _ancilla_cnot(qc, flag_reg[1], ancilla, z_measurement)
    _flag_measure(qc, flag_reg[1], meas_reg[1], z_measurement)

    _ancilla_cnot(qc, stab[-1], ancilla, z_measurement)

    cnots += 1
    if cnots >= 7 and cnots % 2 == 1:
        _ancilla_cnot(qc, flag_reg[flags - 1], ancilla, z_measurement)
        _flag_measure(qc, flag_reg[flags - 1], meas_reg[flags - 1], z_measurement)
    if not z_measurement:
        qc.h(ancilla)

    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_4(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a 2-flagged weight 4 stabilizer. In this case only one flag is required.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    assert len(stab) == 4
    flag_reg = AncillaRegister(1)
    meas_reg = ClassicalRegister(1)
    qc.add_register(flag_reg)
    qc.add_register(meas_reg)
    flag = flag_reg[0]
    flag_meas = meas_reg[0]

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)
    _flag_init(qc, flag, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)
    _flag_measure(qc, flag, flag_meas, z_measurement)

    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_5_or_6(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_5: bool = False,
) -> None:
    """Measure a two-flagged weight 6 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_5: Whether the stabilizer has weight 5.
    """
    assert len(stab) == 6 or (len(stab) == 5 and weight_5)
    flag = AncillaRegister(2)
    meas = ClassicalRegister(2)

    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_5:
        _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_w_flagged_5_or_6(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_5: bool = False,
) -> None:
    """Measure a w-flagged weight 6 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_5: Whether the stabilizer has weight 5.
    """
    assert len(stab) == 6 or (len(stab) == 5 and weight_5)
    flag = AncillaRegister(3)
    meas = ClassicalRegister(3)

    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_5:
        _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_7_or_8(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_7: bool = False,
) -> None:
    """Measure a two-flagged weight 8 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_7: Whether the stabilizer has weight 7.
    """
    assert len(stab) == 8 or (len(stab) == 7 and weight_7)
    flag = AncillaRegister(3)
    meas = ClassicalRegister(3)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_7:
        _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_three_flagged_7_or_8(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_7: bool = False,
) -> None:
    """Measure a three-flagged weight 8 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_7: Whether the stabilizer has weight 7.
    """
    assert len(stab) == 8 or (len(stab) == 7 and weight_7)
    flag = AncillaRegister(4)
    meas = ClassicalRegister(4)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[3], z_measurement)
    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)
    _flag_measure(qc, flag[3], meas[3], z_measurement)

    if not weight_7:
        _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_11_or_12(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_11: bool = False,
) -> None:
    """Measure a two-flagged weight 12 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_11: Whether the stabilizer has weight 11.
    """
    assert len(stab) == 12 or (len(stab) == 11 and weight_11)
    flag = AncillaRegister(5)
    meas = ClassicalRegister(5)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    _flag_init(qc, flag[3], z_measurement)
    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    _flag_init(qc, flag[4], z_measurement)
    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[8], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)
    _flag_measure(qc, flag[3], meas[3], z_measurement)

    _ancilla_cnot(qc, stab[9], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[10], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)
    _flag_measure(qc, flag[4], meas[4], z_measurement)

    if not weight_11:
        _ancilla_cnot(qc, stab[11], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_three_flagged_12(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_11: bool = False,
) -> None:
    """Measure a three-flagged weight 12 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_11: Whether the stabilizer has weight 11.
    """
    assert len(stab) == 12 or (len(stab) == 11 and weight_11)
    flag = AncillaRegister(6)
    meas = ClassicalRegister(6)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[5], z_measurement)
    _ancilla_cnot(qc, flag[5], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[5], ancilla, z_measurement)
    _flag_measure(qc, flag[5], meas[5], z_measurement)

    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    _flag_init(qc, flag[3], z_measurement)
    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    _flag_init(qc, flag[4], z_measurement)
    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[8], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)
    _flag_measure(qc, flag[3], meas[3], z_measurement)

    _ancilla_cnot(qc, stab[9], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[10], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)
    _flag_measure(qc, flag[4], meas[4], z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_11:
        _ancilla_cnot(qc, stab[11], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def vars_to_stab(measurement: list[z3.BoolRef | bool], generators: npt.NDArray[np.int8]) -> npt.NDArray[np.bool_]:
    """Compute the stabilizer measured giving the generators and the measurement variables."""
    if not measurement:
        msg = "Measurement must not be empty"
        raise ValueError(msg)

    if len(generators) != len(measurement):
        msg = "Generators and measurement must have the same length"
        raise ValueError(msg)

    measurement_stab = symbolic_scalar_mult(generators[0], measurement[0])
    for i, scalar in enumerate(measurement[1:]):
        measurement_stab = symbolic_vector_add(measurement_stab, symbolic_scalar_mult(generators[i + 1], scalar))
    return measurement_stab
