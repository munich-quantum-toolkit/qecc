# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulation of Non-deterministic fault tolerant state preparation."""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stim
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from tqdm import tqdm

from ..codes import InvalidCSSCodeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ..codes.css_code import CSSCode

logger = logging.getLogger(__name__)


class NoisyNDFTStatePrepSimulator(ABC):
    """Base class for simulating noisy quantum state preparation circuits with a depolarizing noise model and support for error correction using CSS codes."""

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
            zero_state: Whether the zero state is prepared or nor.
            parallel_gates: Whether to allow for parallel execution of gates.
            decoder: The decoder to use.
        """
        if code.Hx is None or code.Hz is None:
            msg = "The code must have both X and Z checks."
            raise InvalidCSSCodeError(msg)

        self.circ = state_prep_circ
        self.num_verification_qubits = 0
        self.code = code
        self.p = p
        self.p_idle = p if p_idle is None else p_idle
        self.zero_state = zero_state
        self.parallel_gates = parallel_gates
        # Store which measurements are X, Z or data measurements.
        # The indices refer to the indices of the measurements in the stim circuit.
        self.stim_circ = stim.Circuit()
        self.data_measurements: list[int] = []
        self.x_measurements: list[int] = []
        self.z_measurements: list[int] = []
        self.parallel_gates = parallel_gates
        self.n_measurements = 0
        self.stim_circ = stim.Circuit()
        if decoder is None:
            self.decoder = LutDecoder(code)
        else:
            self.decoder = decoder

        self.set_p(p, p_idle)

    def set_p(self, p: float, p_idle: float | None = None, error_free_qubits=None) -> int:
        """Set the error rate and initialize the stim circuit.

        This overwrites the previous stim circuit.

        Args:
        p: The error rate.
        p_idle: Idling error rate. If None, it is set to p.
        """
        if error_free_qubits is None:
            error_free_qubits = []
        self.n_measurements = 0
        self.p = p
        self.p_idle = p if p_idle is None else p_idle
        self.stim_circ = self.to_stim_circ(self.circ, error_free_qubits=error_free_qubits)
        n_measurements = self._compute_postselection_indices()
        if self.zero_state:
            self.data_measurements = self.measure_z(self.stim_circ, n_measurements)
        else:
            self.data_measurements = self.measure_x(self.stim_circ, n_measurements)
        n_measurements += self.code.n

    @abstractmethod
    def _compute_postselection_indices(self) -> int:
        """Compute indices of measurements for postselection.

        Returns:
            int: The number of measurements.
        """

    def to_stim_circ(self, circ: QuantumCircuit, error_free_qubits=None) -> stim.Circuit:
        """Convert a QuantumCircuit to a noisy STIM circuit.

        A depolarizing error model is used:
        - Single-qubit gates and idling qubits are followed by a single-qubit Pauli error with probability 2/9 p. This reflects the fact that two-qubit gates are more likely to fail.
        - Two-qubit gates are followed by a two-qubit Pauli error with probability p/15.
        - Measurements flip with a probability of 2/3 p.
        - Qubit are initialized in the -1 Eigenstate with probability 2/3 p.
        """
        if error_free_qubits is None:
            error_free_qubits = []
        initialized = [False for _ in circ.qubits]
        stim_circuit = stim.Circuit()
        ctrls = []

        def idle_error(used_qubits: list[int]) -> None:
            for q in circ.qubits:
                qubit = circ.find_bit(q)[0]
                if initialized[qubit] and qubit not in used_qubits and qubit not in error_free_qubits:
                    stim_circuit.append_operation("DEPOLARIZE1", [circ.find_bit(q)[0]], [self.p_idle])

        dag = circuit_to_dag(circ)
        layers = dag.layers()
        used_qubits: list[int] = []
        targets = set()
        defaultdict(int)
        [False for _ in circ.qubits]
        for layer in layers:
            layer_circ = dag_to_circuit(layer["graph"])

            # Apply idling errors to all qubits that were unused in the previous layer
            if len(used_qubits) > 0:
                idle_error(used_qubits)

            used_qubits = []
            for gate in layer_circ.data:
                if gate.operation.name == "h":
                    qubit = circ.find_bit(gate.qubits[0])[0]
                    ctrls.append(qubit)
                    if initialized[qubit]:
                        stim_circuit.append_operation("H", [qubit])
                        if not self.parallel_gates:
                            idle_error([qubit])
                        else:
                            used_qubits.append(qubit)

                elif gate.operation.name == "cx":
                    ctrl = circ.find_bit(gate.qubits[0])[0]
                    target = circ.find_bit(gate.qubits[1])[0]
                    targets.add(target)
                    if not initialized[ctrl]:
                        if ctrl in ctrls:
                            stim_circuit.append_operation("H", [ctrl])
                            if ctrl not in error_free_qubits:
                                stim_circuit.append_operation(
                                    "Z_ERROR", [ctrl], [2 * self.p / 3]
                                )  # Wrong initialization
                        elif ctrl not in error_free_qubits:
                            stim_circuit.append_operation("X_ERROR", [ctrl], [2 * self.p / 3])  # Wrong initialization
                        initialized[ctrl] = True
                    if not initialized[target]:
                        if target not in error_free_qubits:
                            stim_circuit.append_operation("X_ERROR", [target], [2 * self.p / 3])  # Wrong initialization
                        if target in ctrls:
                            stim_circuit.append_operation("H", [target])
                        initialized[target] = True

                    stim_circuit.append_operation("CX", [ctrl, target])
                    if ctrl not in error_free_qubits and target not in error_free_qubits:
                        stim_circuit.append_operation("DEPOLARIZE2", [ctrl, target], [self.p])
                    if not self.parallel_gates:
                        idle_error([ctrl, target])
                    else:
                        used_qubits.extend([ctrl, target])

                elif gate.operation.name == "measure":
                    anc = circ.find_bit(gate.qubits[0])[0]
                    if anc not in error_free_qubits:
                        stim_circuit.append_operation("X_ERROR", [anc], [2 * self.p / 3])
                    stim_circuit.append_operation("MR", [anc])
                    if not self.parallel_gates:
                        idle_error([anc])
                    else:
                        used_qubits.append(anc)
                    initialized[anc] = False

        return stim_circuit

    def measure_z(self, circ: stim.Circuit, measurement_index: int, data_index: int = 0) -> list[int]:
        """Measure all data qubits in the Z basis."""
        data_measurements = [measurement_index + i for i in range(self.code.n)]
        circ.append_operation("MRZ", list(range(data_index, data_index + self.code.n)))
        return data_measurements

    def measure_x(self, circ: stim.Circuit, measurement_index: int, data_index: int = 0) -> list[int]:
        """Measure all data qubits in the X basis."""
        data_measurements = [measurement_index + i for i in range(self.code.n)]
        circ.append_operation("MRX", list(range(data_index, data_index + self.code.n)))
        return data_measurements

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

        if self.decoder is None:
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

    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """

    def _simulate_batch(self, shots: int = 1024) -> tuple[int, int]:
        sampler = self.stim_circ.compile_sampler()
        detection_events = sampler.sample(shots).astype(np.int8)

        filtered_events = self._filter_runs(detection_events)

        if len(filtered_events) == 0:  # All events were discarded
            return 0, shots

        state = filtered_events[:, self.data_measurements]

        if self.zero_state:
            checks = ((state @ self.code.Hx.T) % 2).astype(np.int8)
            observables = self.code.Lz % 2
            estimates = self.decoder.batch_decode_x(checks)
        else:
            checks = ((state @ self.code.Hz.T) % 2).astype(np.int8)
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


class VerificationNDFTStatePrepSimulator(NoisyNDFTStatePrepSimulator):
    """Class for simulating noisy state preparation circuit using stabilizer measurements to detect in a verification circuit."""

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
            zero_state: Whether the zero state is prepared or nor.
            parallel_gates: Whether to allow for parallel execution of gates.
            decoder: The decoder to use.
        """
        self.z_verification_measurements: list[int] = []
        self.x_verification_measurements: list[int] = []
        super().__init__(state_prep_circ, code, p, p_idle, zero_state, parallel_gates, decoder)

    def _compute_postselection_indices(self) -> int:
        """Compute the indices of the verification measurements."""
        dag = circuit_to_dag(self.circ)
        layers = dag.layers()
        targets = set()
        n_measurements = 0
        for layer in layers:
            layer_circ = dag_to_circuit(layer["graph"])

            for gate in layer_circ.data:
                if gate.operation.name == "cx":
                    target = self.circ.find_bit(gate.qubits[1])[0]
                    targets.add(target)

                elif gate.operation.name == "measure":
                    anc = self.circ.find_bit(gate.qubits[0])[0]
                    if anc in targets:
                        self.z_verification_measurements.append(n_measurements)
                    else:
                        self.x_verification_measurements.append(n_measurements)
                    n_measurements += 1
        return n_measurements

    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """
        verification_measurements = self.x_verification_measurements + self.z_verification_measurements
        index_array = np.where(np.all(samples[:, verification_measurements] == 0, axis=1))[0]
        return samples[index_array].astype(np.int8)


class SteaneNDFTStatePrepSimulator(NoisyNDFTStatePrepSimulator):
    """Class for simulating Steane-type noisy state preparation circuit.

    A state is checked using multiple copies of the state preparation circuit, which are connected using transversal CNOTs.
    """

    def __init__(
        self,
        circ1: QuantumCircuit,
        circ2: QuantumCircuit,
        code: CSSCode,
        circ3: QuantumCircuit | None = None,
        circ4: QuantumCircuit | None = None,
        p: float = 0.0,
        p_idle: float | None = None,
        zero_state: bool = True,
        parallel_gates: bool = True,
        decoder: LutDecoder | None = None,
        check_circuit: QuantumCircuit | None = None,
    ) -> None:
        """Initialize the simulator.

        Builds the circuit for the Steane-type preparation protocol by connecting the four state preparation circuits using transversal CNOTs.


        If only two circuits are given, than a single transversal CNOT and check are performed.

        Args:
            circ1: The first, state preparation circuit.
            circ2: The second, state preparation circuit.
            circ3: The third, state preparation circuit.
            circ4: The fourth, state preparation circuit
            code: The code to simulate.
            p: The error rate.
            p_idle: Idling error rate. If None, it is set to p.
            zero_state: Whether the zero state is prepared or nor.
            parallel_gates: Whether to allow for parallel execution of gates.
            decoder: The decoder to use.
            check_circuit: Circuit used for checking error rates for the error type that cannot form a logical error on the synthesized state.
        """
        if (circ3 is None and circ4 is not None) or (circ3 is not None and circ4 is None):
            msg = "Only two or four circuits are supported."
            raise ValueError(msg)

        self.has_one_ancilla = circ3 is None

        circ1 = circ1.copy()
        circ2 = circ2.copy()
        if self.has_one_ancilla:
            circ3 = QuantumCircuit()
            circ4 = QuantumCircuit()
        else:
            assert circ3 is not None
            assert circ4 is not None
            circ3 = circ3.copy()
            circ4 = circ4.copy()

        circ2.remove_final_measurements()
        circ3.remove_final_measurements()
        circ4.remove_final_measurements()

        combined = circ4.tensor(circ3).tensor(circ2).tensor(circ1)

        combined.barrier()  # need the barrier to retain order of measurements
        # transversal cnots

        # Define ranges for better readability
        self._data_range = range(code.n)
        self._first_ancilla_range = range(code.n, 2 * code.n)
        self._second_ancilla_range = range(2 * code.n, 3 * code.n)
        self._third_ancilla_range = range(3 * code.n, 4 * code.n)

        if self.has_one_ancilla:
            if zero_state:
                combined.cx(self._data_range, self._first_ancilla_range)
            else:
                combined.cx(self._first_ancilla_range, self._data_range)
        else:
            combined.cx(self._data_range, self._first_ancilla_range)
            combined.cx(self._second_ancilla_range, self._third_ancilla_range)
            combined.cx(self._second_ancilla_range, self._data_range)
            combined.h(self._second_ancilla_range)  # second ancilla is measured in X basis

        combined.barrier()  # need the barrier to retain order of measurements

        n_measured = 3 * code.n if not self.has_one_ancilla else code.n
        cr = ClassicalRegister(n_measured, "c")
        combined.add_register(cr)

        measure_range = self._first_ancilla_range if self.has_one_ancilla else range(code.n, 4 * code.n)
        combined.measure(measure_range, cr)

        self.anc_1: list[int] = []
        self.anc_2: list[int] = []
        self.anc_3: list[int] = []

        self.x_checks = code.Hx if zero_state else np.vstack((code.Hx, code.Lx))
        self.z_checks = code.Hz if not zero_state else np.vstack((code.Hz, code.Lz))
        self.secondary_error_gadget = None
        super().__init__(combined, code, p, p_idle, zero_state, parallel_gates, decoder)

        if check_circuit is None:
            return

        # Estimate error rate using Steane-type error correction
        secondary_error_gadget = combined.copy()
        secondary_error_gadget.barrier()
        secondary_error_gadget = check_circuit.tensor(secondary_error_gadget)
        self._measurement_ancilla_range = range(4 * code.n, 5 * code.n)
        if zero_state:
            secondary_error_gadget.cx(self._data_range, self._measurement_ancilla_range)
        else:
            secondary_error_gadget.cx(self._measurement_ancilla_range, self._data_range)
        if self.zero_state:
            secondary_error_gadget.h(self._data_range)
        secondary_error_gadget.barrier()
        new_cr = ClassicalRegister(code.n, "new_c")
        secondary_error_gadget.add_register(new_cr)
        secondary_error_gadget.measure(self._data_range, new_cr)
        self.secondary_error_gadget = secondary_error_gadget

    def _compute_postselection_indices(self) -> int:
        """Compute indices of measurements for postselection.

        Returns:
                int: The number of measurements.
        """
        self.anc_1 = list(self._first_ancilla_range)
        if not self.has_one_ancilla:
            self.anc_2 = list(self._second_ancilla_range)
            self.anc_3 = list(self._third_ancilla_range)
            return 3 * self.code.n
        return self.code.n

    def set_p(self, p: float, p_idle: float | None = None, error_free_qubits=None) -> None:
        """Set the error rate and initialize the stim circuit.

        This overwrites the previous stim circuit.

        Args:
        p: The error rate.
        p_idle: Idling error rate. If None, it is set to p.
        """
        if error_free_qubits is None:
            error_free_qubits = []
        super().set_p(p, p_idle, error_free_qubits)
        if self.secondary_error_gadget is None:
            return
        self.secondary_stim_circ = self.to_stim_circ(
            self.secondary_error_gadget,
            error_free_qubits=list(self._) + error_free_qubits,
        )
        self.secondary_stim_circ.append("DEPOLARIZE1", list(self._measurement_ancilla_range), [self.p])
        n_measurements = self._compute_postselection_indices()

        self.secondary_ancilla_measurements = list(
            range(n_measurements, n_measurements + self.code.n)
        )  # add measurements of the initial data qubit
        n_measurements += self.code.n

        if self.zero_state:
            self.secondary_data_measurements = self.measure_x(
                self.secondary_stim_circ, n_measurements, data_index=4 * self.code.n
            )
        else:
            self.secondary_data_measurements = self.measure_z(
                self.secondary_stim_circ, n_measurements, data_index=4 * self.code.n
            )

    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """
        anc_1 = samples[:, self.anc_1]
        check_anc_1 = (anc_1 @ self.z_checks.T) % 2

        if not self.has_one_ancilla:
            anc_2 = samples[:, self.anc_2]
            anc_3 = samples[:, self.anc_3]

            check_anc_2 = (anc_2 @ self.x_checks.T) % 2
            check_anc_3 = (anc_3 @ self.z_checks.T) % 2
            index_array = np.where(np.all(np.hstack((check_anc_1, check_anc_2, check_anc_3)) == 0, axis=1))[0]
        else:
            index_array = np.where(np.all(check_anc_1 == 0, axis=1))[0]
        return samples[index_array].astype(np.int8)

    def _simulate_secondary_batch(self, shots: int = 1024) -> tuple[int, int]:
        sampler = self.secondary_stim_circ.compile_sampler()
        detection_events = sampler.sample(shots).astype(np.int8)

        filtered_events = self._filter_runs(detection_events)

        if len(filtered_events) == 0:  # All events were discarded
            return 0, shots

        secondary_state = filtered_events[:, -self.code.n :]

        state = filtered_events[:, self.secondary_ancilla_measurements]

        if self.zero_state:
            observables = self.code.Lx
            estimate_1 = self.decoder.batch_decode_z((state @ self.code.Hx.T % 2).astype(np.int8))
            secondary_state = (secondary_state + estimate_1) % 2
            estimates = self.decoder.batch_decode_z((secondary_state @ self.code.Hx.T % 2).astype(np.int8))
        else:
            estimate_1 = self.decoder.batch_decode_x((state @ self.code.Hz.T % 2).astype(np.int8))
            secondary_state = (secondary_state + estimate_1) % 2
            observables = self.code.Lz
            estimates = self.decoder.batch_decode_x((secondary_state @ self.code.H.T % 2).astype(np.int8))
        corrected = secondary_state + estimates

        num_discarded = detection_events.shape[0] - filtered_events.shape[0]
        num_logical_errors: int = np.sum(
            np.any(corrected @ observables.T % 2 != 0, axis=1)
        )  # number of non-commuting corrected states
        return num_logical_errors, num_discarded

    def logical_error_rate(
        self,
        shots: int = 500000,
        shots_per_batch: int = 500000,
        at_least_min_errors: bool = True,
        min_errors: int = 250,
    ) -> tuple[float, float, int, int, float, float]:
        """Estimate the logical error rate of the code.

        Args:
            shots: The number of shots to use.
            shots_per_batch: The number of shots per batch.
            at_least_min_errors: Whether to continue simulating until at least min_errors are found.
            min_errors: The minimum number of errors to find before stopping.
        """
        p_l, r_a, num_logical_errors, total_shots = super().logical_error_rate(
            shots, shots_per_batch, at_least_min_errors, min_errors
        )

        p_l_error = np.sqrt(p_l * (1 - p_l) / (r_a * total_shots))
        r_a_error = np.sqrt(r_a * (1 - r_a) / total_shots)

        return p_l, r_a, num_logical_errors, total_shots, p_l_error, r_a_error

    def secondary_logical_error_rate(
        self,
        shots: int = 500000,
        shots_per_batch: int = 500000,
        at_least_min_errors: bool = True,
        min_errors: int = 250,
    ) -> tuple[float, float, int, int, float, float]:
        """Estimate the logical error rate of the code with regards to the secondary error type.

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
            num_logical_errors_batch, discarded_batch = self._simulate_secondary_batch(batch)
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

        p_l /= self.code.k
        total_shots = i * batch
        # p_l, r_a, num_logical_errors, total_shots = super().logical_error_rate(
        #     shots, shots_per_batch, at_least_min_errors, min_errors
        # )

        p_l_error = np.sqrt(p_l * (1 - p_l) / (r_a * total_shots))
        r_a_error = np.sqrt(r_a * (1 - r_a) / total_shots)

        return p_l, r_a, num_logical_errors, total_shots, p_l_error, r_a_error


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
    def _generate_lut(
        checks: np.ndarray, chunk_size: int = 2**20, num_workers: int = 8, print_progress: bool = False
    ) -> dict[bytes, np.ndarray]:
        """Generate a lookup table (LUT) for error correction by processing the state space in chunks,
        in parallel, and displaying a progress bar.

        Parameters:
            checks (np.ndarray): The stabilizer check matrix (binary).
            chunk_size (int): Number of states processed per chunk.
            num_workers (int): Number of parallel worker processes (default: use available cores).
            print_progress (bool): Whether to print progress information.

        Returns:
            dict[bytes, np.ndarray]: A LUT mapping syndrome bytes to error state arrays.
        """
        n_qubits = checks.shape[1]
        global_lut = {}

        # Process weights in increasing order so that lower-weight errors take precedence.
        for weight in range(n_qubits):
            total_combinations = math.comb(n_qubits, weight)
            if total_combinations == 0:
                continue
            if print_progress:
                print(f"Processing weight {weight} with {total_combinations} combinations.")

            # Create a generator of all combinations for this weight.
            comb_iter = itertools.combinations(range(n_qubits), weight)
            # Split the combinations into chunks.
            chunks = _chunked_iterable(comb_iter, chunk_size)

            weight_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                d2 = weight_dict.copy()
                futures = [
                    executor.submit(_process_combinations_chunk, chunk, checks, n_qubits, d2) for chunk in chunks
                ]
                if print_progress:
                    for future in tqdm(
                        concurrent.futures.as_completed(futures), total=len(futures), desc=f"Weight {weight}"
                    ):
                        _merge_into(weight_dict, future.result())

                else:
                    for future in concurrent.futures.as_completed(futures):
                        _merge_into(weight_dict, future.result())

            _merge_into(global_lut, weight_dict)

            if len(global_lut) == 2 ** checks.shape[0]:
                if print_progress:
                    print("All syndromes found.")
                break

        return global_lut


def _chunked_iterable(iterable, chunk_size):
    """Yield lists of items from the given iterable, each of size at most chunk_size."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _process_combinations_chunk(chunk, checks, n_qubits, weight_map):
    """Process a chunk of combinations. For each combination, construct the binary state,
    compute its syndrome, and add it to a dictionary if not already present.

    Returns:
        dict: mapping syndrome (bytes) -> error state (numpy array)
    """
    chunk_dict = {}
    for comb in chunk:
        # Create an error state with 1's in positions given by comb.
        state = np.zeros(n_qubits, dtype=np.int8)
        state[list(comb)] = 1
        # Compute the syndrome and cast to int8 for consistency.
        syndrome = ((checks @ state) % 2).astype(np.int8)
        syndrome_bytes = syndrome.tobytes()
        # Since all states here have the same weight,
        # we keep the first encountered state for a given syndrome.
        if syndrome_bytes not in weight_map and syndrome_bytes not in chunk_dict:
            chunk_dict[syndrome_bytes] = state.copy()
    return chunk_dict


def _merge_dicts(dict_list):
    """Merge a list of dictionaries. In case of key conflicts,
    the first encountered value is kept.
    """
    merged = {}
    for d in dict_list:
        for key, state in d.items():
            if key not in merged:
                merged[key] = state
    return merged


def _merge_into(target, source) -> None:
    """Merge source dictionary into target dictionary. In case of key conflicts,
    keep the existing value in the target.
    """
    for key, state in source.items():
        if key not in target:
            target[key] = state
