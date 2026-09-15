# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analog Tannergraph Decoding Simulator."""

from __future__ import annotations

import json
import locale
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder

from mqt.qecc.noise import (
    GaussianReadoutChannel,
    PhenomenologicalNoiseModel,
    PhenomenologicalNoiseSampler,
)

from ..utils import simulation_utils
from ..utils.data_utils import calculate_error_rates, is_converged

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..utils.data_utils import BpParams


class AnalogTannergraphDecoder:
    """Analog Tannergraph decoder.

    Builds the analog tanner graph and uses this as decoding graph.
    """

    def __init__(
        self,
        pcm: NDArray[np.int32],
        bp_params: BpParams,
        error_channel: NDArray[np.float64],
        sigma: float = 0.0,
        ser: float | None = None,
    ) -> None:
        """Initialize the decoder."""
        self.m, self.n = pcm.shape
        self.sigma = sigma
        self.H = pcm
        self.bp_params = bp_params
        self.atg = get_analog_pcm(pcm)
        self.syndr_err_rate = ser
        self.error_channel = error_channel

        if self.sigma <= 0.0:
            if self.syndr_err_rate is None:
                msg = "Either sigma or ser must be specified"
                raise ValueError(msg)
            self.sigma = GaussianReadoutChannel.from_bit_error_probability(self.syndr_err_rate).sigma
        elif self.syndr_err_rate is not None:
            msg = "Only one of sigma or ser must be specified"
            raise ValueError(msg)

        self.bposd_decoder = BpOsdDecoder(
            pcm=self.atg.astype(np.int_),
            channel_probs=np.hstack((self.error_channel, np.zeros(self.m))),  # initd as dummy for now
            max_iter=self.bp_params.max_bp_iter,
            bp_method=self.bp_params.bp_method,
            osd_order=self.bp_params.osd_order,
            osd_method=self.bp_params.osd_method,
            ms_scaling_factor=self.bp_params.ms_scaling_factor,
            schedule=self.bp_params.schedule,
            omp_thread_count=self.bp_params.omp_thread_count,
            serial_schedule_order=self.bp_params.serial_schedule_order,
        )

        # self.bp_decoder = bp_decoder(
        #     parity_check_matrix=self.atg,
        #     channel_probs=np.hstack((self.error_channel, np.zeros(self.m))),  # initd as dummy for now
        #     max_iter=self.bp_params.max_bp_iter,
        #     bp_method=self.bp_params.bp_method,
        #     ms_scaling_factor=self.bp_params.ms_scaling_factor,
        # )

    def _set_analog_syndrome(self, analog_syndrome: NDArray[np.float64]) -> None:
        """Initializes the error channel of the BP decoder.

        Ensures that the virtual nodes are initialized with the analog syndrome LLRs on decoding initialization.
        :param analog_syndrome: the analog syndrome values to initialize the virtual nodes with.
        """
        new_channel = np.hstack((
            self.bposd_decoder.channel_probs[: self.n],
            simulation_utils.get_virtual_check_init_vals(analog_syndrome, self.sigma),
        ))
        self.bposd_decoder.update_channel_probs(new_channel)

    def decode(self, analog_syndrome: NDArray[np.float64]) -> NDArray[np.int32]:
        """Decode a given analog syndrome."""
        self._set_analog_syndrome(analog_syndrome)
        return np.asarray(
            self.bposd_decoder.decode(simulation_utils.get_binary_from_analog(analog_syndrome)), dtype=np.int32
        )


class AtdSimulator:
    """Analog Tanner graph Decoding Simulator."""

    def __init__(
        self,
        hx: NDArray[np.int32],
        lx: NDArray[np.int32],
        hz: NDArray[np.int32],
        lz: NDArray[np.int32],
        codename: str,
        seed: int,
        bp_params: BpParams,
        data_err_rate: float,
        code_params: dict[str, Any],
        syndr_err_rate: float | None = None,
        sigma: float | None = None,
        bias: NDArray[np.float64] | None = None,
        experiment: str = "atd",
        decoding_method: str = "atd",
        output_path: str = "./",
        **kwargs: Any,  # ruff:ignore[any-type]
    ) -> None:
        """Initialize the simulator."""
        if bias is None:
            bias = np.array([1.0, 1.0, 1.0])
        self.Hx = hx
        self.Lx = lx
        self.Hz = hz
        self.Lz = lz
        self.codename = codename
        self.data_err_rate = data_err_rate
        self.bias = bias
        self.experiment = experiment
        self.decoding_method = decoding_method
        self.sigma = sigma
        self.outfile = output_path
        if sigma is None:
            if syndr_err_rate is None:
                msg = "Either sigma or ser must be specified"
                raise ValueError(msg)

            self.syndr_err_rate = syndr_err_rate

        else:
            if syndr_err_rate is not None:
                msg = "Only one of sigma or ser must be specified"
                raise ValueError(msg)

            self.syndr_err_rate = GaussianReadoutChannel(sigma).bit_error_probability

        synd_err_channel = simulation_utils.error_channel_setup(
            error_rate=self.syndr_err_rate,
            xyz_error_bias=self.bias,
        )
        self.x_sigma = GaussianReadoutChannel.from_bit_error_probability(synd_err_channel.x_marginal).sigma
        self.z_sigma = GaussianReadoutChannel.from_bit_error_probability(synd_err_channel.z_marginal).sigma

        self.bp_params = bp_params
        self.save_interval = kwargs.get("save_interval", 1_000)
        self.eb_precision = kwargs.get("eb_precision", 1e-1)
        self.input_values = self.__dict__.copy()
        self.rng = np.random.default_rng(seed)

        self.n = hx.shape[1]
        self.code_params = code_params
        del self.input_values["Hx"]
        del self.input_values["Lx"]
        del self.input_values["Hz"]
        del self.input_values["Lz"]

        # setup decoders
        if self.decoding_method == "atd":
            Decoder = AnalogTannergraphDecoder  # ruff:ignore[non-lowercase-variable-in-function]

        # single-sided error only, no bias
        self.noise_model = PhenomenologicalNoiseModel(
            data=simulation_utils.error_channel_setup(
                error_rate=self.data_err_rate,
                xyz_error_bias=self.bias,
            ),
            x_syndrome=GaussianReadoutChannel(self.x_sigma),
            z_syndrome=GaussianReadoutChannel(self.z_sigma),
        )
        self.noise_sampler = PhenomenologicalNoiseSampler(self.noise_model, rng=self.rng)
        self.x_decoder = Decoder(
            error_channel=self.noise_model.x_marginals(self.n),
            pcm=self.Hz,
            sigma=self.x_sigma,
            bp_params=self.bp_params,
        )
        self.z_decoder = Decoder(
            error_channel=self.noise_model.z_marginals(self.n),
            pcm=self.Hx,
            sigma=self.z_sigma,
            bp_params=self.bp_params,
        )
        self.x_bp_iterations = 0
        self.z_bp_iterations = 0

    def single_sample(self) -> tuple[bool, bool]:
        """Samples and error and decodes once. Returns if the decoding round was successful separately for each side."""
        residual_err: list[NDArray[np.int32]] = [
            np.zeros(self.n).astype(np.int32),
            np.zeros(self.n).astype(np.int32),
        ]  # no residual error
        x_err, z_err = self.noise_sampler.sample_data(self.n, (residual_err[0], residual_err[1]))  # no residual error

        x_perf_syndr = (self.Hz @ x_err) % 2
        x_noisy_syndr = np.asarray(self.noise_sampler.sample_x_syndrome(x_perf_syndr), dtype=np.float64)
        x_decoding = self.x_decoder.decode(x_noisy_syndr)[: self.n]
        self.x_bp_iterations += self.x_decoder.bposd_decoder.iter
        x_residual = (x_err + x_decoding) % 2

        z_perf_syndr = (self.Hx @ z_err) % 2
        z_noisy_syndr = np.asarray(self.noise_sampler.sample_z_syndrome(z_perf_syndr), dtype=np.float64)
        z_decoding = self.z_decoder.decode(z_noisy_syndr)[: self.n]
        self.z_bp_iterations += self.z_decoder.bposd_decoder.iter
        z_residual = (z_err + z_decoding) % 2

        return not simulation_utils.is_logical_err(self.Lz, x_residual), not simulation_utils.is_logical_err(
            self.Lx, z_residual
        )

    def run(self, samples: int) -> dict[str, Any]:
        """Run the simulation."""
        x_success_cnt = 0
        z_success_cnt = 0
        for runs in range(1, samples + 1):
            out = self.single_sample()

            x_success_cnt += int(out[0])
            z_success_cnt += int(out[1])
            if runs % self.save_interval == 1:
                self.save_results(x_success_cnt, z_success_cnt, runs)

                # check convergence only once during each save interval
                if is_converged(
                    x_success_cnt,
                    z_success_cnt,
                    runs,
                    self.code_params,
                    self.eb_precision,
                ):
                    print("Result has converged.")  # ruff:ignore[print]
                    break

        x_ler, x_ler_eb, x_wer, x_wer_eb = calculate_error_rates(x_success_cnt, runs, self.code_params)
        z_ler, z_ler_eb, z_wer, z_wer_eb = calculate_error_rates(z_success_cnt, runs, self.code_params)
        output = {
            "code_K": self.code_params["k"],
            "code_N": self.code_params["n"],
            "nr_runs": runs,
            "pers": self.data_err_rate,
            "x_sigma": self.x_sigma,
            "z_sigma": self.z_sigma,
            "x_ler": x_ler,
            "x_ler_eb": x_ler_eb,
            "x_wer": x_wer,
            "x_wer_eb": x_wer_eb,
            "x_success_cnt": x_success_cnt,
            "z_ler": z_ler,
            "z_ler_eb": z_ler_eb,
            "z_wer": z_wer,
            "z_wer_eb": z_wer_eb,
            "z_success_cnt": z_success_cnt,
            "bp_params": self.bp_params,
            "x_bp_iterations": self.x_bp_iterations / runs,
            "z_bp_iterations": self.z_bp_iterations / runs,
        }

        self.save_results(x_success_cnt, z_success_cnt, runs)  # save final results

        return output

    def save_results(self, x_success_cnt: int, z_success_cnt: int, runs: int) -> dict[str, Any]:
        """Compute error rates and error bars and save output dict."""
        x_ler, x_ler_eb, x_wer, x_wer_eb = calculate_error_rates(x_success_cnt, runs, self.code_params)
        z_ler, z_ler_eb, z_wer, z_wer_eb = calculate_error_rates(z_success_cnt, runs, self.code_params)
        output = {
            "code_K": self.code_params["k"],
            "code_N": self.code_params["n"],
            "nr_runs": runs,
            "pers": self.data_err_rate,
            "x_sigma": self.x_sigma,
            "z_sigma": self.z_sigma,
            "x_ler": x_ler,
            "x_ler_eb": x_ler_eb,
            "x_wer": x_wer,
            "x_wer_eb": x_wer_eb,
            "x_success_cnt": x_success_cnt,
            "z_ler": z_ler,
            "z_ler_eb": z_ler_eb,
            "z_wer": z_wer,
            "z_wer_eb": z_wer_eb,
            "z_success_cnt": z_success_cnt,
            "bp_params": self.bp_params,
            "x_bp_iterations": self.x_bp_iterations / runs,
            "z_bp_iterations": self.z_bp_iterations / runs,
        }

        output.update(self.input_values)
        with Path(self.outfile).open(mode="w", encoding=locale.getpreferredencoding(False)) as f:
            json.dump(output, f, ensure_ascii=False, indent=4, default=lambda o: o.__dict__)
        return output


def get_analog_pcm(pcm: NDArray[np.int32]) -> NDArray[np.int32]:
    """Constructs apcm = [H | I_m] where I_m is the m x m identity matrix."""
    return np.hstack((pcm, np.identity(pcm.shape[0], dtype=np.int32)))
