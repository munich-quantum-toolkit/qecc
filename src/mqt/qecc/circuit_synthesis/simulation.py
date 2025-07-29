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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from tqdm import tqdm

from ..codes import InvalidCSSCodeError
from .circuit_utils import measured_qubits, qiskit_to_stim_circuit, relabel_qubits, unmeasured_qubits
from .circuits import CNOTCircuit
from .noise import CircuitLevelNoiseIdlingParallel
from .state_prep import heuristic_prep_circuit

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Generator, Iterator

    import numpy.typing as npt
    import stim

    from ..codes.css_code import CSSCode
    from .noise import NoiseModel


logger = logging.getLogger(__name__)


class NoisyNDFTStatePrepSimulator(ABC):
    """Base class for simulating noisy quantum state preparation circuits with a depolarizing noise model and support for error correction using CSS codes."""

    def __init__(
        self,
        state_prep_circ: QuantumCircuit | stim.Circuit | CNOTCircuit,
        code: CSSCode,
        zero_state: bool = True,
        decoder: LutDecoder | None = None,
    ) -> None:
        """Initialize the simulator.

        Args:
            state_prep_circ: The state preparation circuit.
            code: The code to simulate.
            zero_state: Whether the zero state is prepared or nor.
            decoder: The decoder to use.
        """
        if code.Hx is None or code.Hz is None:
            msg = "The code must have both X and Z checks."
            raise InvalidCSSCodeError(msg)

        if isinstance(state_prep_circ, QuantumCircuit):
            self.circ = qiskit_to_stim_circuit(state_prep_circ)
        elif isinstance(state_prep_circ, CNOTCircuit):
            self.circ = state_prep_circ.to_stim_circuit()
        else:
            self.circ = state_prep_circ.copy()

        self.code = code
        self.zero_state = zero_state
        self.data_measurements: list[int] = []
        self.n_measurements = 0
        if decoder is None:
            self.decoder = LutDecoder(code)
        else:
            self.decoder = decoder
        self.data_qubits = sorted(unmeasured_qubits(self.circ))
        self.data_measurements = list(range(self.circ.num_measurements, self.circ.num_measurements + code.n))

    def _build_noisy_circuit(self, noise: NoiseModel) -> stim.Circuit:
        """Set the error rate and initialize the noisy stim circuit.

        Args:
            noise: The noise model to apply.

        Returns:
            The noisy stim circuit used for the protocol.
        """
        noisy_circ = noise.apply(self.circ)

        if self.zero_state:
            noisy_circ.append("MR", self.data_qubits)
        else:
            noisy_circ.append("MRX", self.data_qubits)
        self._noisy_circ = noisy_circ
        return noisy_circ

    def _build_noisy_gadget(self, noise: NoiseModel, p: float) -> stim.Circuit:
        anc = heuristic_prep_circuit(self.code, zero_state=not self.zero_state).circ.to_stim_circuit()
        noisy_circ = noise.apply(self.circ)

        anc_qubits = list(range(noisy_circ.num_qubits, noisy_circ.num_qubits + anc.num_qubits))
        anc = relabel_qubits(anc, noisy_circ.num_qubits)
        anc.append_operation("DEPOLARIZE1", range(anc.num_qubits), p)
        noisy_circ += anc

        ctrls = self.data_qubits if self.zero_state else anc_qubits
        trgts = anc_qubits if self.zero_state else self.data_qubits
        noisy_circ.append("CX", [item for pair in zip(ctrls, trgts) for item in pair])
        if self.zero_state:
            noisy_circ.append("MRX", self.data_qubits)
            noisy_circ.append("MRX", anc_qubits)
        else:
            noisy_circ.append("MR", self.data_qubits)
            noisy_circ.append("MR", anc_qubits)
        self._noisy_circ = noisy_circ
        return noisy_circ

    def _batched_logical_error_rate(
        self,
        sampler: stim.CompiledMeasurementSampler,
        processing_fun: Callable[[stim.CompiledMeasurementSampler, int], tuple[int, int]],
        shots: int = 100000,
        shots_per_batch: int = 100000,
        at_least_min_errors: bool = True,
        min_errors: int = 500,
    ) -> tuple[float, float, int, int]:
        batch = min(shots_per_batch, shots)
        p_l = 0.0
        r_a = 0.0

        num_logical_errors = 0

        i = 1
        total_batches = int(np.ceil(shots / batch))
        while i <= total_batches or at_least_min_errors:
            num_logical_errors_batch, discarded_batch = processing_fun(sampler, batch)

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

    def logical_error_rate(
        self,
        noise: NoiseModel,
        shots: int = 100000,
        shots_per_batch: int = 100000,
        at_least_min_errors: bool = True,
        min_errors: int = 500,
    ) -> tuple[float, float, int, int]:
        """Estimate the logical error rate of the code.

        Args:
            noise: The noise model to apply.
            shots: The number of shots to use.
            shots_per_batch: The number of shots per batch.
            at_least_min_errors: Whether to continue simulating until at least min_errors are found.
            min_errors: The minimum number of errors to find before stopping.
        """
        noisy_circ = self._build_noisy_circuit(noise)
        sampler = noisy_circ.compile_sampler()
        return self._batched_logical_error_rate(
            sampler, self._simulate_batch, shots, shots_per_batch, at_least_min_errors, min_errors
        )

    def secondary_logical_error_rate(
        self,
        noise: NoiseModel,
        p: float,
        shots: int = 100000,
        shots_per_batch: int = 100000,
        at_least_min_errors: bool = True,
        min_errors: int = 500,
    ) -> tuple[float, float, int, int]:
        """Estimate the secondary logical error rate of the code.

        For a zero (plus) state, we cannot directly estimate whether the circuit is strictly fault-tolerant with respect to Z (X) errors because a logical error of that kind cannot happen.
        This method is used to estimate the logical error rate for a zero (plus) state using a secondary error gadget.
        The prepared qubit is used as the ancilla in Steane-type QEC of an ancillary state prepared in the opposite basis (for a zero state, the plus state is used and vice-versa).
        The prepared qubit is assumed to be subject to uniform depolarizing noise of strength p.

        Args:
            noise: The noise model to apply.
            p: Noise to apply to the gadget.
            shots: The number of shots to use.
            shots_per_batch: The number of shots per batch.
            at_least_min_errors: Whether to continue simulating until at least min_errors are found.
            min_errors: The minimum number of errors to find before stopping.

        Returns:
            The logical error rate and the acceptance rate of the protocol.
        """
        noisy_circ = self._build_noisy_gadget(noise, p)
        sampler = noisy_circ.compile_sampler()
        return self._batched_logical_error_rate(
            sampler, self._simulate_secondary_batch, shots, shots_per_batch, at_least_min_errors, min_errors
        )

    @abstractmethod
    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """

    def _simulate_batch(self, sampler: stim.CompiledMeasurementSampler, shots: int = 1024) -> tuple[int, int]:
        detection_events = sampler.sample(shots).astype(np.int8)

        filtered_events = self._filter_runs(detection_events)

        if len(filtered_events) == 0:  # All events were discarded
            return 0, shots

        state = filtered_events[:, self.data_measurements]

        if self.zero_state:
            checks = ((state @ self.code.Hz.T) % 2).astype(np.int8)
            observables = self.code.Lz
            estimates = self.decoder.batch_decode_x(checks)
        else:
            checks = ((state @ self.code.Hx.T) % 2).astype(np.int8)
            observables = self.code.Lx
            estimates = self.decoder.batch_decode_z(checks)

        corrected = state + estimates

        num_discarded = detection_events.shape[0] - filtered_events.shape[0]
        num_logical_errors: int = np.sum(
            np.any(corrected @ observables.T % 2 != 0, axis=1)
        )  # number of non-commuting corrected states
        return num_logical_errors, num_discarded

    def _simulate_secondary_batch(self, sampler: stim.CompiledMeasurementSampler, shots: int = 1024) -> tuple[int, int]:
        detection_events = sampler.sample(shots).astype(np.int8)

        filtered_events = self._filter_runs(detection_events)

        if len(filtered_events) == 0:  # All events were discarded
            return 0, shots

        state = filtered_events[:, self.data_measurements]

        ancilla_state = filtered_events[:, -self.code.n :]  # last n measurements are the gadget ancilla

        if self.zero_state:
            checks = ((state @ self.code.Hx.T) % 2).astype(np.int8)
            observables = self.code.Lx
            estimates = self.decoder.batch_decode_z(checks)
        else:
            checks = ((state @ self.code.Hz.T) % 2).astype(np.int8)
            observables = self.code.Lz
            estimates = self.decoder.batch_decode_x(checks)

        # Steane-type QEC: apply estimate to the ancilla state
        corrected_anc = ancilla_state ^ estimates

        if self.zero_state:
            checks = ((corrected_anc @ self.code.Hx.T) % 2).astype(np.int8)
            estimates = self.decoder.batch_decode_z(checks)
        else:
            checks = ((corrected_anc @ self.code.Hz.T) % 2).astype(np.int8)
            estimates = self.decoder.batch_decode_x(checks)

        # apply new estimates

        corrected_anc ^= estimates

        num_discarded = detection_events.shape[0] - filtered_events.shape[0]
        num_logical_errors: int = np.sum(
            np.any(corrected_anc @ observables.T % 2 != 0, axis=1)
        )  # number of non-commuting corrected states
        return num_logical_errors, num_discarded

    def plot_state_prep(  # pragma: no cover
        self,
        ps: list[float],
        min_errors: int = 500,
        p_idle_factor: float = 1.0,
        kind: str = "primary",
    ) -> None:
        """Plot the logical error rate and acceptance rate as a function of the physical error rate.

        Args:
            ps: The physical error rates to plot.
            min_errors: The minimum number of errors to find before stopping.
            p_idle_factor: Factor to scale the idling error rate depending on ps.
            kind: Which error rates to plot. Can be "primary", "secondary", or "all".
        """
        if kind not in {"primary", "secondary", "all"}:
            msg = 'kind must be either "primary", "secondary", or "all".'
            raise ValueError(msg)
        plot_primary = kind in {"primary", "all"}
        plot_secondary = kind in {"secondary", "all"}

        if plot_primary:
            results = [
                self.logical_error_rate(
                    CircuitLevelNoiseIdlingParallel(p, p, p, p, p * p_idle_factor, True),
                    min_errors=min_errors,
                )
                for p in ps
            ]
            p_ls, r_as = zip(*[(p_l, r_a) for p_l, r_a, _, _ in results])

        if plot_secondary:
            results_secondary = [
                self.secondary_logical_error_rate(
                    CircuitLevelNoiseIdlingParallel(p, p, p, p, p * p_idle_factor, True),
                    p,
                    min_errors=min_errors,
                )
                for p in ps
            ]
            p_ls_secondary, r_as = zip(*[(p_l, r_a) for p_l, r_a, _, _ in results_secondary])

        # Create a figure with a consistent size
        plt.figure(figsize=(12, 6))

        # Plot logical error rate
        plt.subplot(1, 2, 1)
        if plot_primary:
            plt.plot(
                ps,
                p_ls,
                marker="o",
                linestyle="-",
                color="blue",
                label="Primary Logical Error Rate",
            )
        if plot_secondary:
            plt.plot(
                ps,
                p_ls_secondary,
                marker="^",
                linestyle="-",
                color="red",
                label="Secondary Logical Error Rate",
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Physical Error Rate", fontsize=12)
        plt.ylabel("Logical Error Rate", fontsize=12)
        plt.title("Logical Error Rate vs Physical Error Rate", fontsize=14, fontweight="bold")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=10)

        # Plot acceptance rate
        plt.subplot(1, 2, 2)

        # acceptance rate is the same for both protocols
        plt.plot(
            ps,
            r_as,
            marker="d",
            linestyle="--",
            color="orange",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Physical Error Rate", fontsize=12)
        plt.ylabel("Acceptance Rate", fontsize=12)
        plt.title("Acceptance Rate vs Physical Error Rate", fontsize=14, fontweight="bold")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()


class VerificationNDFTStatePrepSimulator(NoisyNDFTStatePrepSimulator):
    """Class for simulating noisy state preparation circuit using stabilizer measurements to detect in a verification circuit."""

    def __init__(
        self,
        state_prep_circ: QuantumCircuit | stim.Circuit | CNOTCircuit,
        code: CSSCode,
        zero_state: bool = True,
        decoder: LutDecoder | None = None,
    ) -> None:
        """Initialize the simulator.

        Args:
            state_prep_circ: The state preparation circuit.
            code: The code to simulate.
            zero_state: Whether the zero state is prepared or nor.
            decoder: The decoder to use.
        """
        super().__init__(state_prep_circ, code, zero_state, decoder)

    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """
        index_array = np.where(np.all(samples[:, : self.circ.num_measurements] == 0, axis=1))[0]
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
        zero_state: bool = True,
        decoder: LutDecoder | None = None,
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
            zero_state: Whether the zero state is prepared or nor.
            decoder: The decoder to use.
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
        super().__init__(combined, code, zero_state, decoder)

        self.anc_1 = list(self._first_ancilla_range)
        if not self.has_one_ancilla:
            self.anc_2 = list(self._second_ancilla_range)
            self.anc_3 = list(self._third_ancilla_range)

    def _filter_runs(self, samples: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Filter samples based on measurement outcomes.

        Args:
            samples: The samples to filter.

        Returns:
            npt.NDArray[np.int8]: The filtered samples.
        """
        measured = measured_qubits(self._noisy_circ)

        qubit_to_meas = {q: i for i, q in enumerate(measured)}
        idx1 = [qubit_to_meas[q] for q in self.anc_1]
        anc_1 = samples[:, idx1]
        check_anc_1 = (anc_1 @ self.z_checks.T) % 2

        if not self.has_one_ancilla:
            idx2 = [qubit_to_meas[q] for q in self.anc_2]
            idx3 = [qubit_to_meas[q] for q in self.anc_3]
            anc_2 = samples[:, idx2]
            anc_3 = samples[:, idx3]

            check_anc_2 = (anc_2 @ self.x_checks.T) % 2
            check_anc_3 = (anc_3 @ self.z_checks.T) % 2
            index_array = np.where(np.all(np.hstack((check_anc_1, check_anc_2, check_anc_3)) == 0, axis=1))[0]
        else:
            index_array = np.where(np.all(check_anc_1 == 0, axis=1))[0]
        return samples[index_array].astype(np.int8)


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
        """Generate a lookup table (LUT) for error correction by processing the state space in chunks, in parallel, and displaying a progress bar.

        Parameters:
            checks (np.ndarray): The stabilizer check matrix (binary).
            chunk_size (int): Number of states processed per chunk.
            num_workers (int): Number of parallel worker processes (default: use available cores).
            print_progress (bool): Whether to print progress information.

        Returns:
            dict[bytes, np.ndarray]: A LUT mapping syndrome bytes to error state arrays.
        """
        n_qubits = checks.shape[1]
        global_lut: dict[bytes, np.ndarray] = {}

        # Process weights in increasing order so that lower-weight errors take precedence.
        for weight in range(n_qubits):
            total_combinations = math.comb(n_qubits, weight)
            if total_combinations == 0:
                continue
            if print_progress:
                pass

            # Create a generator of all combinations for this weight.
            comb_iter = itertools.combinations(range(n_qubits), weight)
            # Split the combinations into chunks.
            chunks = _chunked_iterable(comb_iter, chunk_size)

            weight_dict: dict[bytes, int] = {}
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
                    pass
                break

        return global_lut


def _chunked_iterable(iterable: Iterator[tuple[int, ...]], chunk_size: int) -> Generator[list[tuple[int, ...]]]:
    """Yield lists of items from the given iterable, each of size at most chunk_size."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _process_combinations_chunk(
    chunk: list[tuple[int, ...]], checks: npt.NDArray[np.int8], n_qubits: int, weight_map: dict[bytes, int]
) -> dict[bytes, npt.NDArray[np.int8]]:
    """Process a chunk of combinations.

    For each combination, construct the binary state, compute its syndrome, and add it to a dictionary if not already present.

    Returns:
        dict: mapping syndrome (bytes) -> error state (numpy array)
    """
    chunk_dict: dict[bytes, npt.NDArray[np.int8]] = {}
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


def _merge_dicts(dict_list: list[dict[bytes, npt.NDArray[np.int8]]]) -> dict[bytes, npt.NDArray[np.int8]]:
    """Merge a list of dictionaries.

    In case of key conflicts, the first encountered value is kept.
    """
    merged = {}
    for d in dict_list:
        for key, state in d.items():
            if key not in merged:
                merged[key] = state
    return merged


def _merge_into(target: dict[bytes, npt.NDArray[np.int8]], source: dict[bytes, npt.NDArray[np.int8]]) -> None:
    """Merge source dictionary into target dictionary.

    In case of key conflicts, keep the existing value in the target.
    """
    for key, state in source.items():
        if key not in target:
            target[key] = state
