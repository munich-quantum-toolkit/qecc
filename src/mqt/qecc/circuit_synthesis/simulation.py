# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulation of Non-deterministic fault tolerant state preparation."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stim
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ..codes import InvalidCSSCodeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from qiskit.circuit import QuantumCircuit

    from ..codes.css_code import CSSCode

logger = logging.getLogger(__name__)


class NoisyNDFTStatePrepSimulator:
    """Class for simulating noisy state preparation circuit using a depolarizing noise model."""

    def __init__(
        self,
        state_prep_circ: QuantumCircuit,
        code: CSSCode,
        p: float = 0.0,
        p_idle: float | None = None,
        zero_state: bool = True,
        parallel_gates: bool = True,
        decoder: LutDecoder | None = None,
    ) -> None:
        """Initialize the simulator.

        Args:
            state_prep_circ: The state preparation circuit.
            code: The code to simulate.
            p: The error rate.
            p_idle: Idling error rate. If None, it is set to p.
            zero_state: Whether thezero state is prepared or nor.
            parallel_gates: Whether to allow for parallel execution of gates.
            decoder: The decoder to use.
        """
        if code.Hx is None or code.Hz is None:
            msg = "The code must have both X and Z checks."
            raise InvalidCSSCodeError(msg)

        self.circ = state_prep_circ
        self.num_qubits = state_prep_circ.num_qubits
        self.code = code
        self.p = p
        self.p_idle = p if p_idle is None else p_idle
        self.zero_state = zero_state
        # Store which measurements are X, Z or data measurements.
        # The indices refer to the indices of the measurements in the stim circuit.
        self.x_verification_measurements: list[int] = []
        self.z_verification_measurements: list[int] = []
        self.x_measurements: list[int] = []
        self.z_measurements: list[int] = []
        self.data_measurements: list[int] = []
        self.parallel_gates = parallel_gates
        self.n_measurements = 0
        self.stim_circ = stim.Circuit()
        if decoder is None:
            self.decoder = LutDecoder(code)
        else:
            self.decoder = decoder
        self.set_p(p, p_idle)

    def set_p(self, p: float, p_idle: float | None = None) -> None:
        """Set the error rate.

        This reinitializes the stim circuit.

        Args:
        p: The error rate.
        p_idle: Idling error rate. If None, it is set to p.
        """
        self.x_verification_measurements = []
        self.z_verification_measurements = []
        self.x_measurements = []
        self.z_measurements = []
        self.data_measurements = []
        self.n_measurements = 0
        self.p = p
        self._reused_qubits = 0
        self.p_idle = p if p_idle is None else p_idle
        self.stim_circ = self.to_stim_circ()
        self.num_qubits = (
            self.stim_circ.num_qubits
            - (len(self.x_verification_measurements) + len(self.z_verification_measurements))
            + self._reused_qubits
        )
        self.measure_stabilizers()
        if self.zero_state:
            self.measure_z()
        else:
            self.measure_x()

    def to_stim_circ(self) -> stim.Circuit:
        """Convert a QuantumCircuit to a noisy STIM circuit.

        A depolarizing error model is used:
        - Single-qubit gates and idling qubits are followed by a single-qubit Pauli error with probability 2/9 p. This reflects the fact that two-qubit gates are more likely to fail.
        - Two-qubit gates are followed by a two-qubit Pauli error with probability p/15.
        - Measurements flip with a probability of 2/3 p.
        - Qubit are initialized in the -1 Eigenstate with probability 2/3 p.
        """
        initialized = [False for _ in self.circ.qubits]
        stim_circuit = stim.Circuit()
        ctrls = []

        def idle_error(used_qubits: list[int]) -> None:
            for q in self.circ.qubits:
                qubit = self.circ.find_bit(q)[0]
                if initialized[qubit] and qubit not in used_qubits:
                    stim_circuit.append_operation("DEPOLARIZE1", [self.circ.find_bit(q)[0]], [self.p_idle])

        dag = circuit_to_dag(self.circ)
        layers = dag.layers()
        used_qubits: list[int] = []
        targets = set()
        measured: defaultdict[int, int] = defaultdict(int)
        for layer in layers:
            layer_circ = dag_to_circuit(layer["graph"])

            # Apply idling errors to all qubits that were unused in the previous layer
            if len(used_qubits) > 0:
                idle_error(used_qubits)

            used_qubits = []
            for gate in layer_circ.data:
                if gate.operation.name == "h":
                    qubit = self.circ.find_bit(gate.qubits[0])[0]
                    ctrls.append(qubit)
                    if initialized[qubit]:
                        stim_circuit.append_operation("H", [qubit])
                        # stim_circuit.append_operation("DEPOLARIZE1", [qubit], [self.p])
                        if not self.parallel_gates:
                            idle_error([qubit])
                        else:
                            used_qubits.append(qubit)

                elif gate.operation.name == "cx":
                    ctrl = self.circ.find_bit(gate.qubits[0])[0]
                    target = self.circ.find_bit(gate.qubits[1])[0]
                    targets.add(target)
                    if not initialized[ctrl]:
                        if ctrl in ctrls:
                            stim_circuit.append_operation("H", [ctrl])
                            stim_circuit.append_operation("Z_ERROR", [ctrl], [2 * self.p / 3])  # Wrong initialization
                        else:
                            stim_circuit.append_operation("X_ERROR", [ctrl], [2 * self.p / 3])  # Wrong initialization
                        initialized[ctrl] = True
                    if not initialized[target]:
                        stim_circuit.append_operation("X_ERROR", [target], [2 * self.p / 3])  # Wrong initialization
                        initialized[target] = True

                    stim_circuit.append_operation("CX", [ctrl, target])
                    stim_circuit.append_operation("DEPOLARIZE2", [ctrl, target], [self.p])
                    if not self.parallel_gates:
                        idle_error([ctrl, target])
                    else:
                        used_qubits.extend([ctrl, target])

                elif gate.operation.name == "measure":
                    anc = self.circ.find_bit(gate.qubits[0])[0]
                    stim_circuit.append_operation("X_ERROR", [anc], [2 * self.p / 3])
                    stim_circuit.append_operation("MR", [anc])
                    if anc in targets:
                        self.z_verification_measurements.append(self.n_measurements)
                    else:
                        self.x_verification_measurements.append(self.n_measurements)
                    self.n_measurements += 1
                    if not self.parallel_gates:
                        idle_error([anc])
                    else:
                        used_qubits.append(anc)
                    initialized[anc] = False
                    measured[anc] += 1
                    if measured[anc] == 2:
                        self._reused_qubits += 1

        return stim_circuit

    def measure_stabilizers(self) -> None:
        """Measure the stabilizers of the code.

        An ancilla is used for each measurement.
        """
        assert self.code.Hx is not None
        assert self.code.Hz is not None

        for check in self.code.Hx:
            supp = _support(check)
            anc = self.stim_circ.num_qubits
            self.stim_circ.append_operation("H", [anc])
            for q in supp:
                self.stim_circ.append_operation("CX", [anc, q])
            self.stim_circ.append_operation("MRX", [anc])
            self.x_measurements.append(self.n_measurements)
            self.n_measurements += 1

        for check in self.code.Hz:
            supp = _support(check)
            anc = self.stim_circ.num_qubits
            for q in supp:
                self.stim_circ.append_operation("CX", [q, anc])
            self.stim_circ.append_operation("MRZ", [anc])
            self.z_measurements.append(self.n_measurements)
            self.n_measurements += 1

    def measure_z(self) -> None:
        """Measure all data qubits in the Z basis."""
        self.data_measurements = [self.n_measurements + i for i in range(self.num_qubits)]
        self.n_measurements += self.num_qubits
        self.stim_circ.append_operation("MRZ", list(range(self.num_qubits)))

    def measure_x(self) -> None:
        """Measure all data qubits in the X basis."""
        self.data_measurements = [self.n_measurements + i for i in range(self.num_qubits)]
        self.n_measurements += self.num_qubits
        self.stim_circ.append_operation("MRX", list(range(self.num_qubits)))

    def logical_error_rate(
        self,
        shots: int = 100000,
        shots_per_batch: int = 100000,
        at_least_min_errors: bool = True,
        min_errors: int = 500,
    ) -> tuple[float, float, int, int]:
        """Estimate the logical error rate of the code.

        Args:
            shots: The number of shots to use.
            shots_per_batch: The number of shots per batch.
            at_least_min_errors: Whether to continue simulating until at least min_errors are found.
            min_errors: The minimum number of errors to find before stopping.
        """
        batch = min(shots_per_batch, shots)
        p_l = 0.0
        r_a = 0.0

        num_logical_errors = 0

        if self.zero_state:
            self.decoder.generate_x_lut()
        else:
            self.decoder.generate_z_lut()

        i = 1
        while i <= int(np.ceil(shots / batch)) or at_least_min_errors:
            num_logical_errors_batch, discarded_batch = self._simulate_batch(batch)

            logger.info(
                f"Batch {i}: {num_logical_errors_batch} logical errors and {discarded_batch} discarded shots. {batch - discarded_batch} shots used.",
            )
            p_l_batch = num_logical_errors_batch / (batch - discarded_batch) if discarded_batch != batch else 0.0
            p_l = ((i - 1) * p_l + p_l_batch) / i

            r_a_batch = 1 - discarded_batch / batch

            # Update statistics
            num_logical_errors += num_logical_errors_batch
            r_a = ((i - 1) * r_a + r_a_batch) / i

            if at_least_min_errors and num_logical_errors >= min_errors:
                break
            i += 1

        return p_l / self.code.k, r_a, num_logical_errors, i * batch

    def _simulate_batch(self, shots: int = 1024) -> tuple[int, int]:
        sampler = self.stim_circ.compile_sampler()
        detection_events = sampler.sample(shots)

        # Filter events where the verification circuit flagged
        verification_measurements = self.x_verification_measurements + self.z_verification_measurements
        index_array = np.where(np.all(detection_events[:, verification_measurements] == 0, axis=1))[0]
        filtered_events = detection_events[index_array].astype(np.int8)

        if len(filtered_events) == 0:  # All events were discarded
            return 0, shots

        state = filtered_events[:, self.data_measurements]

        if self.zero_state:
            checks = filtered_events[:, self.z_measurements]
            observables = self.code.Lz
            estimates = self.decoder.batch_decode_x(checks)
        else:
            checks = filtered_events[:, self.x_measurements]
            observables = self.code.Lx
            estimates = self.decoder.batch_decode_z(checks)

        corrected = state + estimates

        num_discarded = detection_events.shape[0] - filtered_events.shape[0]
        num_logical_errors: int = np.sum(
            np.any(corrected @ observables.T % 2 != 0, axis=1)
        )  # number of non-commuting corrected states
        return num_logical_errors, num_discarded

    def plot_state_prep(
        self,
        ps: list[float],
        min_errors: int = 500,
        name: str | None = None,
        p_idle_factor: float = 1.0,
    ) -> None:
        """Plot the logical error rate and accaptence rate as a function of the physical error rate.

        Args:
            ps: The physical error rates to plot.
            min_errors: The minimum number of errors to find before stopping.
            name: The name of the plot.
            p_idle_factor: Factor to scale the idling error rate depending on ps.
        """
        p_ls = []
        r_as = []
        for p in ps:
            self.set_p(p, p_idle_factor * p)
            p_l, r_a, _num_logical_errors, _num_shots = self.logical_error_rate(min_errors=min_errors)
            p_ls.append(p_l)
            r_as.append(r_a)

        plt.subplot(1, 2, 1)
        plt.plot(ps, p_ls, marker="o", label=name)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Physical error rate")
        plt.ylabel("Logical error rate")

        if name is not None:
            plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(ps, r_as, marker="o", label=name)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Physical error rate")
        plt.ylabel("Acceptance rate")
        if name is not None:
            plt.legend()
        plt.tight_layout()


class LutDecoder:
    """Lookup table decoder for a CSSCode."""

    def __init__(self, code: CSSCode, init_luts: bool = True) -> None:
        """Initialize the decoder.

        Args:
            code: The code to decode.
            init_luts: Whether to initialize the lookup tables at object creation.
        """
        self.code = code
        self.x_lut: dict[bytes, npt.NDArray[np.int8]] = {}
        self.z_lut: dict[bytes, npt.NDArray[np.int8]] = {}
        if init_luts:
            self.generate_x_lut()
            self.generate_z_lut()

    def batch_decode_x(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int8]:
        """Decode the X errors given a batch of syndromes."""
        return np.apply_along_axis(self.decode_x, 1, syndromes)

    def batch_decode_z(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int8]:
        """Decode the Z errors given a batch of syndromes."""
        return np.apply_along_axis(self.decode_z, 1, syndromes)

    def decode_x(self, syndrome: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Decode the X errors given a syndrome."""
        if len(self.x_lut) == 0:
            self.generate_x_lut()
        return self.x_lut[syndrome.tobytes()]

    def decode_z(self, syndrome: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Decode the Z errors given a syndrome."""
        if len(self.z_lut) == 0:
            self.generate_z_lut()
        return self.z_lut[syndrome.tobytes()]

    def generate_x_lut(self) -> None:
        """Generate the lookup table for the X errors."""
        if len(self.x_lut) != 0:
            return

        assert self.code.Hz is not None, "The code does not have a Z stabilizer matrix."
        self.x_lut = LutDecoder._generate_lut(self.code.Hz)
        if self.code.is_self_dual():
            self.z_lut = self.x_lut

    def generate_z_lut(self) -> None:
        """Generate the lookup table for the Z errors."""
        if len(self.z_lut) != 0:
            return

        assert self.code.Hx is not None, "The code does not have an X stabilizer matrix."
        self.z_lut = LutDecoder._generate_lut(self.code.Hx)
        if self.code.is_self_dual():
            self.z_lut = self.x_lut

    @staticmethod
    def _generate_lut(checks: npt.NDArray[np.int_]) -> dict[bytes, npt.NDArray[np.int_]]:
        """Generate a lookup table for the stabilizer state.

        The lookup table maps error syndromes to their errors.
        """
        n_qubits = checks.shape[1]

        lut: dict[bytes, npt.NDArray[np.int8]] = {}
        syndrome_weights: dict[bytes, tuple[npt.NDArray[np.int8_], int]] = {}

        for i in range(2**n_qubits):
            state: npt.NDArray[np.int_] = np.array(list(np.binary_repr(i).zfill(n_qubits))).astype(np.int8)
            syndrome = checks @ state % 2
            syndrome_bytes = syndrome.astype(np.int8).tobytes()
            val = syndrome_weights.get(syndrome_bytes)
            weight = state.sum()
            if val is None:
                syndrome_weights[syndrome_bytes] = (state, weight)
                continue
            _, w = val

            if weight < w:
                syndrome_weights[syndrome_bytes] = (state, weight)

        for key, v in syndrome_weights.items():
            lut[key] = np.array(v[0])

        return lut


def _support(v: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Return the support of a vector."""
    return np.where(v != 0)[0]
