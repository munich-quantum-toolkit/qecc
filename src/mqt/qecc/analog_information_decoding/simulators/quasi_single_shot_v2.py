# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Quasi single shot simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from pymatching import Matching

from mqt.qecc.noise import (
    BitFlipChannel,
    GaussianReadoutChannel,
    PhenomenologicalNoiseModel,
    PhenomenologicalNoiseSampler,
    ReadoutChannel,
)

from ..utils.data_utils import _check_convergence
from ..utils.simulation_utils import (
    error_channel_setup,
    get_binary_from_analog,
    is_logical_err,
    save_results,
)
from .memory_experiment_v2 import (
    build_multiround_pcm,
    decode_multiround,
    move_syndrome,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..utils.data_utils import BpParams


class QssSimulator:
    """Quasi single shot simulator."""

    def __init__(
        self,
        pcm: NDArray[np.int32],
        per: float,
        ser: float,
        logicals: NDArray[np.int32],
        bias: NDArray[np.float64],
        codename: str,
        bp_params: BpParams,
        code_params: dict[str, int],
        decoding_method: str = "bposd",  # bposd or matching
        check_side: str = "X",
        seed: int = 666,
        analog_tg: bool = False,
        repetitions: int = 0,
        rounds: int = 0,
        experiment: str = "qss",
        outpath: str = "/",
        **kwargs: Any,  # ruff:ignore[any-type]
    ) -> None:
        """Initialize QSS Simulator.

        :param pcm: parity-check matrix of code.

        :param per: physical data error rate
        :param ser: syndrome error rate
        :param logicals: logical matrix
        :param bias: bias array
        :param codename: name of the code
        :param bp_params: BP decoder parameters
        :param check_side: side of the check (X or Z)
        :param seed: random seed
        :param analog_tg: switch analog decoding on/off
        :param repetitions: number of total syndrome measurements, i.e., total time steps. Must be even.
        :param rounds: number of decoding runs, i.e., number of times we slide the window - 1
        :param experiment: name of experiment, for outpath creation
        :param kwargs:
        """
        self.H = pcm
        self.data_err_rate = per
        self.syndr_err_rate = ser
        self.check_side = check_side
        self.L = logicals
        self.bias = bias
        self.codename = codename
        self.bp_params = bp_params
        self.decoding_method = decoding_method
        self.save_interval = kwargs.get("save_interval", 50)
        self.eb_precision = kwargs.get("eb_precision", 1e-2)
        self.analog_tg = analog_tg
        self.repetitions = repetitions
        if repetitions % 2 != 0:
            msg = "repetitions must be even"
            raise ValueError(msg)

        if self.decoding_method not in {"bposd", "matching"}:
            msg = "Decoding method must be either bposd or matching"
            raise ValueError(msg)

        if self.repetitions % 2 != 0:
            msg = "Repetitions must be even!"
            raise ValueError(msg)

        self.rounds = rounds
        self.experiment = experiment
        self.code_params = code_params
        self.input_values = self.__dict__.copy()
        self.rng = np.random.default_rng(seed)

        self.outfile = outpath

        # Remove Arrays
        del self.input_values["H"]
        del self.input_values["L"]

        self.num_checks, self.num_qubits = self.H.shape

        data_channel = error_channel_setup(error_rate=self.data_err_rate, xyz_error_bias=bias)
        syndrome_pauli_channel = error_channel_setup(error_rate=self.syndr_err_rate, xyz_error_bias=bias)

        if self.check_side == "X":
            self.err_idx = 1
            # Z bit/syndrome errors
            data_err_rate = data_channel.z_marginal
            syndr_err_rate = syndrome_pauli_channel.z_marginal
        else:
            # we have X errors on qubits
            self.err_idx = 0
            # X bit/syndrome errors
            data_err_rate = data_channel.x_marginal
            syndr_err_rate = syndrome_pauli_channel.x_marginal
        self.data_err_channel = np.full(self.num_qubits, data_err_rate)
        self.syndr_err_channel = np.full(self.num_checks, syndr_err_rate)

        # initialize the multiround parity-check matrix as described in the paper
        self.H3D = build_multiround_pcm(
            self.H,
            self.repetitions - 1,
        )

        # the number of columns of the diagonal check matrix of the H3D matrix
        self.check_block_size = self.num_qubits * (self.repetitions)

        channel_probs: NDArray[np.float64] = np.zeros(self.H3D.shape[1]).astype(np.float64)
        # The bits corresponding to the columns of the diagonal H-block of H3D are initialized with the bit channel
        channel_probs[: self.check_block_size] = data_err_rate

        # The remaining bits (corresponding to the identity block of H3D)
        # are initialized with the syndrome error channel
        channel_probs[self.check_block_size :] = syndr_err_rate

        # If we do ATG decoding, initialize sigma (syndrome noise strength)
        if self.analog_tg:
            self.sigma = GaussianReadoutChannel.from_bit_error_probability(syndr_err_rate).sigma  # x/z + y
            syndrome_channel: ReadoutChannel = GaussianReadoutChannel(self.sigma)
        else:
            syndrome_channel = BitFlipChannel(syndr_err_rate)
        self.noise_model = PhenomenologicalNoiseModel(
            data=data_channel,
            x_syndrome=syndrome_channel,
            z_syndrome=syndrome_channel,
        )
        self.noise_sampler = PhenomenologicalNoiseSampler(self.noise_model, rng=self.rng)
        self.bp_iterations = 0
        if self.decoding_method == "bposd":
            self.decoder = BpOsdDecoder(
                pcm=self.H3D.astype(np.int_),
                channel_probs=channel_probs,
                max_iter=self.bp_params.max_bp_iter,
                bp_method=self.bp_params.bp_method,
                osd_order=self.bp_params.osd_order,
                osd_method=self.bp_params.osd_method,
                ms_scaling_factor=self.bp_params.ms_scaling_factor,
            )
        elif self.decoding_method == "matching":
            weights = np.log((1 - channel_probs) / channel_probs)
            self.decoder = Matching(self.H3D, weights=weights)
        self.channel_probs = channel_probs

    def _decode_multiround(
        self,
        syndrome_mat: NDArray[np.int32],
        analog_syndr_mat: NDArray[np.float64],
        last_round: bool = False,
    ) -> tuple[Any, NDArray[np.int32], NDArray[np.float64], int]:
        decoded, syndrome, analog_syndr, bp_iter = decode_multiround(
            syndrome=syndrome_mat,
            pcm=self.H,
            decoder=self.decoder,
            repetitions=self.repetitions,
            last_round=last_round,
            analog_syndr=analog_syndr_mat,
            check_block_size=self.check_block_size,
            sigma=self.sigma,
            h3d=self.H3D if self.decoding_method == "matching" else None,  # avoid passing matrix in case not needed
            channel_probs=self.channel_probs,
            decoding_method=self.decoding_method,
        )
        assert analog_syndr is not None
        return decoded, syndrome, analog_syndr, bp_iter

    def _single_sample(self) -> int:
        # prepare fresh syndrome matrix and error vector
        # each column == measurement result of a single timestep
        syndrome_mat: NDArray[np.int32] = np.zeros((self.num_checks, self.repetitions), dtype=np.int32)

        if self.analog_tg:
            analog_syndr_mat: NDArray[np.float64] = np.zeros((self.num_checks, self.repetitions), dtype=np.float64)

        err: NDArray[np.int32] = np.zeros(self.num_qubits, dtype=np.int32)
        cnt = 0  # counter for syndrome_mat

        for rnd in range(self.rounds):
            residual_err = [np.copy(err), np.copy(err)]
            err = self.noise_sampler.sample_data(self.num_qubits, (residual_err[0], residual_err[1]))[
                self.err_idx
            ]  # only first or last vector needed, depending on side (X or Z)
            noiseless_syndrome = (self.H @ err) % 2

            # add syndrome error
            if rnd != (self.rounds - 1):
                if self.analog_tg:
                    analog_syndrome = np.asarray(
                        self.noise_sampler.sample_syndrome(noiseless_syndrome, self.noise_model.x_syndrome),
                        dtype=np.float64,
                    )
                    syndrome = get_binary_from_analog(analog_syndrome)
                else:
                    syndrome = np.asarray(
                        self.noise_sampler.sample_syndrome(noiseless_syndrome, self.noise_model.x_syndrome),
                        dtype=np.int32,
                    )
            else:  # last round is perfect
                syndrome = np.copy(noiseless_syndrome)
                analog_syndrome = np.asarray(
                    self.noise_sampler.sample_syndrome(noiseless_syndrome, GaussianReadoutChannel(0.0)),
                    dtype=np.float64,
                )  # no noise

            # fill the corresponding column of the syndrome/analog syndrome matrix
            syndrome_mat[:, cnt] += syndrome
            if self.analog_tg:
                analog_syndr_mat[:, cnt] += analog_syndrome

            cnt += 1  # move to next column of syndrome matrix

            if cnt == self.repetitions:  # if we have filled the syndrome matrix, decode
                if rnd != (self.rounds - 1):  # if not last round, decode and move syndrome
                    cnt = self.repetitions // 2  # reset counter to start of tentative region

                    # the correction is only the correction of the commit region
                    (corr, syndrome_mat, analog_syndr_mat, bp_iters) = self._decode_multiround(
                        syndrome_mat,
                        analog_syndr_mat,
                        last_round=False,
                    )
                    # we compute the average for all rounds since this equals a single sample
                    self.bp_iterations += int(bp_iters / self.rounds)
                    err = (err + corr) % 2
                    syndrome_mat = move_syndrome(syndrome_mat)
                    if self.analog_tg:
                        analog_syndr_mat = move_syndrome(analog_syndr_mat, data_type=np.float64)

                else:  # if we are in the last round, decode and stop
                    # the correction is the correction of the commit and tentative region
                    (corr, syndrome_mat, analog_syndr_mat, bp_iters) = self._decode_multiround(
                        syndrome_mat,
                        analog_syndr_mat,
                        last_round=True,
                    )
                    self.bp_iterations += int(bp_iters / self.rounds)
                    err = (err + corr) % 2
        return int(not is_logical_err(self.L, err))

    def _save_results(self, success_cnt: int, samples: int) -> dict[str, Any]:
        return save_results(
            success_cnt=success_cnt,
            nr_runs=samples,
            p=self.data_err_rate,
            s=self.syndr_err_rate,
            input_vals=self.input_values,
            outfile=self.outfile,
            code_params=self.code_params,
            err_side="z" if self.check_side == "X" else "x",
            bp_iterations=self.bp_iterations,
            bp_params=self.bp_params,
        )

    def run(self, samples: int = 1) -> dict[str, Any]:
        """Returns single data point."""
        success_cnt = 0
        for run in range(1, samples + 1):
            success_cnt += self._single_sample()
            if run % self.save_interval == 1:
                self._save_results(success_cnt, run)
                if _check_convergence(success_cnt, run, self.code_params, self.eb_precision):
                    print("Converged")  # ruff:ignore[print]
                    break
        return self._save_results(success_cnt, run)
