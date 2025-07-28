# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Single shot simulations."""

from __future__ import annotations

import json
import locale
from pathlib import Path
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any

import numpy as np
from ldpc.osd import bposd_decoder

from ..utils.data_utils import (
    BpParams,
    calculate_error_rates,
    is_converged,
    replace_inf,
)
from ..utils.data_utils import create_outpath as get_outpath
from ..utils.simulation_utils import (
    build_single_stage_pcm,
    check_logical_err_h,
    error_channel_setup,
    generate_err,
    generate_syndr_err,
    get_binary_from_analog,
    get_noisy_analog_syndrome,
    get_sigma_from_syndr_er,
    get_signed_from_binary,
    get_virtual_check_init_vals,
    is_logical_err,
    set_seed,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SingleShotSimulator:
    """Single shot decoding simulator."""

    def __init__(
        self,
        codename: str,
        per: float,
        ser: float,
        single_stage: bool,
        seed: int,
        bias: NDArray[np.float64],
        x_meta: bool,
        z_meta: bool,
        sus_th_depth: int,
        bp_params: BpParams,
        code_params: dict[str, int],
        hx: NDArray[np.int32],
        hz: NDArray[np.int32],
        mx: NDArray[np.int32] | None,
        mz: NDArray[np.int32] | None,
        lz: NDArray[np.int32] | None,
        lx: NDArray[np.int32] | None,
        analog_info: bool = False,
        cutoff: int = 0,
        analog_tg: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize simulator."""
        set_seed(seed)
        self.codename = codename
        self.data_err_rate = per
        self.syndr_err_rate = ser
        self.bias = bias
        self.x_meta = x_meta
        self.z_meta = z_meta
        self.sus_th_depth = sus_th_depth
        self.bp_params = bp_params
        self.seed = seed
        self.single_stage = single_stage
        self.analog_info = analog_info
        self.cutoff = cutoff
        self.save_interval = kwargs.get("save_interval", 50)
        self.eb_precision = kwargs.get("eb_precision", 1e-1)
        self.analog_tg = analog_tg
        self.x_bp_iters = 0
        self.z_bp_iters = 0
        # self.code_path = f"generated_codes/{codename}"
        self.code_path = f"/codes/generated_codes/{codename}"
        # Load code params

        self.code_params = code_params

        self.input_values = self.__dict__.copy()
        self.outfile = get_outpath(**self.input_values)

        # Set parity check matrices
        self.Hx = hx
        self.Hz = hz
        if self.x_meta:
            self.Mx = mx
        if self.z_meta:
            self.Mz = mz

        self.n = self.Hx.shape[1]  # m==shape[0] ambiguous for Hx, Hz, better use their shape directly
        self.check_input()
        # load logicals if possible
        if lx is not None and lz is not None:
            self.lx = lx
            self.lz = lz
            self._check_logicals = True
        else:
            self._check_logicals = False

        (
            self.x_bit_err_channel,
            self.y_bit_err_channel,
            self.z_bit_err_channel,
        ) = error_channel_setup(self.data_err_rate, self.bias, self.n)
        # encapsulates all bit error channels
        self.data_error_channel = (
            self.x_bit_err_channel,
            self.y_bit_err_channel,
            self.z_bit_err_channel,
        )

        # now setup syndrome error channels.
        # This needs two calls since the syndromes may have different lengths
        # first vector ([0]) is x channel probability vector
        # by our convention the syndrome is named after the error.
        full_x_syndr_err_chnl = error_channel_setup(self.syndr_err_rate, self.bias, self.Hz.shape[0])
        full_z_syndr_err_chnl = error_channel_setup(self.syndr_err_rate, self.bias, self.Hx.shape[0])
        self.x_syndr_error_channel = full_x_syndr_err_chnl[0] + full_x_syndr_err_chnl[1]  # x+y
        self.z_syndr_error_channel = full_z_syndr_err_chnl[2] + full_z_syndr_err_chnl[1]  # z+y

        # if we want to decode with analog syndrome noise and an analog decoder
        if self.analog_info or self.analog_tg:
            self.sigma_x = get_sigma_from_syndr_er(self.x_syndr_error_channel[0])
            self.sigma_z = get_sigma_from_syndr_er(self.z_syndr_error_channel[0])

        # if we want to decode with the analog tanner graph method construct respective matrices.
        # These are assumed to exist in the *_setup() methods
        if self.analog_tg:
            self.x_apcm, self.z_apcm = self.construct_analog_pcms()

        if self.single_stage:
            self._single_stage_setup()
        else:
            self._two_stage_setup()

        self._total_decoding_time = 0.0

    def check_input(self) -> None:
        """Check initialization parameters for consistency."""
        if self.analog_tg is True and self.analog_info is True:
            msg = "analog_tg and analog_info cannot be both True"
            raise ValueError(msg)

    def _single_sample(
        self,
    ) -> tuple[bool, bool]:
        """Simulates a single sample for a given sustainable threshold depth."""
        residual_err: list[NDArray[np.int32]] = [
            np.zeros(self.n).astype(np.int32),  # X-residual error part
            np.zeros(self.n).astype(np.int32),  # Z-residual error part
        ]

        # for single shot simulation we have sus_th_depth number of 'noisy' simulations (residual error carried over)
        # followed by a single round of perfect syndrome extraction after the sustainable threshold loop
        for _round in range(self.sus_th_depth):
            x_err, z_err = generate_err(
                nr_qubits=self.n,
                channel_probs=self.data_error_channel,
                residual_err=residual_err,
            )
            # by our convention, we call the syndrome after the error that is occurred
            # however, the check_error_rate depends on which check errors,
            # hence is named after the check matrix.
            x_syndrome = self.Hz @ x_err % 2
            z_syndrome = self.Hx @ z_err % 2

            x_syndrome_w_err, z_syndrome_w_err = self._get_noisy_syndrome(x_syndrome, z_syndrome)

            # collect total decoding time
            start = timer()
            if self.single_stage:
                x_decoded, z_decoded = self._single_stage_decoding(x_syndrome_w_err, z_syndrome_w_err)
            else:
                x_decoded, z_decoded = self._two_stage_decoding(x_syndrome_w_err, z_syndrome_w_err)
            end = timer()
            self._total_decoding_time += end - start

            residual_err = [
                np.array((x_err + x_decoded) % 2, dtype=np.int32),  # np conversion needed to avoid rt error
                np.array((z_err + z_decoded) % 2, dtype=np.int32),
            ]

        # perfect measurement round at the end
        x_err, z_err = generate_err(
            nr_qubits=self.n,
            channel_probs=self.data_error_channel,
            residual_err=residual_err,
        )

        # X-syndrome: sx = Hz * ex
        # Z-syndrome: sz = Hx * ez
        x_syndrome = self.Hz @ x_err % 2
        z_syndrome = self.Hx @ z_err % 2

        start = timer()
        x_decoded = self.x_bpd.decode(x_syndrome)
        z_decoded = self.z_bpd.decode(z_syndrome)
        end = timer()
        self._total_decoding_time += end - start

        # residual[0]: X-residual error, residual[1]: Z-residual error
        residual_err = [(x_err + x_decoded) % 2, (z_err + z_decoded) % 2]

        x_residual_err = residual_err[0]
        z_residual_err = residual_err[1]

        # check for logical errors
        # check if residual X-error commutes with Z-logical and vice versa
        # equivalently, check if residual X-error is in rowspace of Hz and vice versa
        if self._check_logicals:
            is_x_logical_error = is_logical_err(self.lz, x_residual_err)
            is_z_logical_error = is_logical_err(self.lx, z_residual_err)
        else:
            is_x_logical_error = check_logical_err_h(self.Hz, x_err, x_residual_err)
            is_z_logical_error = check_logical_err_h(self.Hx, z_err, z_residual_err)

        return is_x_logical_error, is_z_logical_error

    def _get_noisy_syndrome(
        self, x_syndrome: NDArray[np.int32], z_syndrome: NDArray[np.int32]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        if self.syndr_err_rate != 0.0:
            if self.analog_info or self.analog_tg:  # analog syndrome error with converted sigma
                x_syndrome_w_err = get_noisy_analog_syndrome(perfect_syndr=x_syndrome, sigma=self.sigma_x)
                z_syndrome_w_err = get_noisy_analog_syndrome(perfect_syndr=z_syndrome, sigma=self.sigma_z)
            else:  # usual pauli error channel syndrome error
                x_syndrome_err = generate_syndr_err(channel_probs=self.x_syndr_error_channel)
                x_syndrome_w_err = (x_syndrome + x_syndrome_err) % 2
                z_syndrome_err = generate_syndr_err(channel_probs=self.z_syndr_error_channel)
                z_syndrome_w_err = (z_syndrome + z_syndrome_err) % 2
        else:
            x_syndrome_w_err = np.copy(x_syndrome)
            z_syndrome_w_err = np.copy(z_syndrome)

        return x_syndrome_w_err, z_syndrome_w_err

    def _single_stage_decoding(
        self, x_syndrome_w_err: NDArray[np.float64], z_syndrome_w_err: NDArray[np.float64]
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Single stage decoding of the given syndromes.

        If meta checks are activated, we apply the single-stage decoding method.
        This is complemented by either analog_tg decoding or analog_info decoding (or standard decoding) of the
        single stage matrix.

        Otherwise, we apply the decoding methods to the standard check matrices, hence the meta code is not used.
        """
        # for convenience
        x_err_channel = self.x_bit_err_channel + self.y_bit_err_channel
        z_err_channel = self.z_bit_err_channel + self.y_bit_err_channel

        if self.x_meta and self.Mx is not None:
            z_decoded, self.z_bp_iters = self._decode_ss_with_meta(
                syndrome_w_err=z_syndrome_w_err,
                meta_pcm=self.Mx,
                ss_bpd=self.ss_z_bpd,
                bit_err_channel=z_err_channel,
                sigma=self.sigma_z,
            )
        else:  # do not use x meta checks, just decode with noisy syndrome on standard check matrix
            z_decoded, self.z_bp_iters = self._decode_ss_no_meta(
                syndrome_w_err=z_syndrome_w_err,
                analog_tg_decoder=self.z_abpd,
                standard_decoder=self.z_bpd,
                bit_err_channel=z_err_channel,
                sigma=self.sigma_z,
            )

        if self.z_meta and self.Mz is not None:
            x_decoded, self.x_bp_iters = self._decode_ss_with_meta(
                syndrome_w_err=x_syndrome_w_err,
                meta_pcm=self.Mz,
                ss_bpd=self.ss_x_bpd,
                bit_err_channel=x_err_channel,
                sigma=self.sigma_x,
            )
        else:  # do not use x meta checks, just decode with noisy syndrome on check matrix or analog_tg method
            x_decoded, self.x_bp_iters = self._decode_ss_no_meta(
                syndrome_w_err=x_syndrome_w_err,
                analog_tg_decoder=self.x_abpd,
                standard_decoder=self.x_bpd,
                bit_err_channel=x_err_channel,
                sigma=self.sigma_x,
            )
        return x_decoded, z_decoded

    def _decode_ss_no_meta(
        self,
        syndrome_w_err: NDArray[np.float64],
        analog_tg_decoder: Any,  # noqa: ANN401
        standard_decoder: Any,  # noqa: ANN401
        bit_err_channel: NDArray[np.float64],
        sigma: float,
    ) -> tuple[NDArray[np.int32], int]:
        """Decoding of syndrome without meta checks.

        In case analog_tg is active, we use the analog tanner graph decoding method, which uses the analog syndrome.
        Otherwise, we decode the standard check matrix with BPOSD, or the AI decoder in case analog_info is active.
        """
        if self.analog_tg:
            decoded = self._analog_tg_decoding(
                decoder=analog_tg_decoder,
                hard_syndrome=get_binary_from_analog(syndrome_w_err),
                analog_syndrome=syndrome_w_err,
                bit_err_channel=bit_err_channel,
                sigma=sigma,
            )
            it = analog_tg_decoder.iter
        else:
            decoded = standard_decoder.decode(syndrome_w_err)
            it = standard_decoder.iter
        return decoded, it

    def _decode_ss_with_meta(
        self,
        syndrome_w_err: NDArray[np.float64],
        ss_bpd: Any,  # noqa: ANN401
        meta_pcm: NDArray[np.int32],
        bit_err_channel: NDArray[np.float64],
        sigma: float,
    ) -> tuple[NDArray[np.int32], int]:
        """Single-Stage decoding for given syndrome.

        If analog_tg is active, we use the analog tanner graph decoding method, which uses the analog syndrome to
        initialize the virtual nodes s.t. they contain the soft info.

        If analog_info is active, we use the analog_info decoder on the single-stage matrix.

        Otherwise, standard single-stage decoding is applied
        """
        if self.analog_tg:
            decoded = self._ss_analog_tg_decoding(
                decoder=ss_bpd,
                analog_syndrome=syndrome_w_err,
                meta_pcm=meta_pcm,
                bit_err_channel=bit_err_channel,
                sigma=sigma,
            )
        else:
            if self.analog_info:
                meta_bin = (meta_pcm @ get_binary_from_analog(syndrome_w_err)) % 2
                meta_syndr = get_signed_from_binary(meta_bin)  # for AI decoder we need {-1,+1} syndrome as input
            else:
                meta_syndr = (meta_pcm @ syndrome_w_err) % 2

            ss_syndr = np.hstack((syndrome_w_err, meta_syndr))
            # only first n bit are data, the other are virtual nodes and can be discarded for estimate
            decoded = ss_bpd.decode(ss_syndr)[: self.n]
        return decoded, ss_bpd.iter

    def _ss_analog_tg_decoding(
        self,
        decoder: Any,  # noqa: ANN401
        analog_syndrome: NDArray[np.float64],
        meta_pcm: NDArray[np.int32],
        bit_err_channel: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.int32]:
        """Decodes the noisy analog syndrome using the single stage analog tanner graph and BPOSD.

        That is, combines single-stage and analog tanner graph method.
        In the standard single-stage method, BP is initialized with the bit channel + the syndrome channel.
        In order for the virtual nodes to contain the analog info, we adapt the syndrome channel by computing
        the 'analog channel' given the analog syndrome.
        """
        analog_channel = get_virtual_check_init_vals(analog_syndrome, sigma)
        decoder.update_channel_probs(np.hstack((bit_err_channel, analog_channel)))

        bin_syndr = get_binary_from_analog(analog_syndrome)  # here we need to threshold since syndrome is analog
        meta_syndr = (meta_pcm @ bin_syndr) % 2
        ss_syndr = np.hstack((bin_syndr, meta_syndr))
        # only first n bit are data, return them
        return decoder.decode(ss_syndr)[: self.n]

    def _two_stage_decoding(
        self, x_syndrome_w_err: NDArray[np.float64], z_syndrome_w_err: NDArray[np.float64]
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Two stage decoding of single shot code.

        Meta checks on either side (or both) can be deactivated.

        If the meta checks for a side are activated the syndrome is thresholded and repaired using the meta code.

        If analog_tg is used for decoding, the analog syndrome is used to initialize the BP decoder and the
        thresholded (repaired) syndrome is used as input.

        In case analog_info is activated x_bpd/z_bpd have been set accordinlgy in the setup() method.
        """
        if self.x_meta and self.Mx is not None:
            # Mx corrects failures of Hx and Mz corrects failures of Mz
            # thus, meta syndrome of z_syndrome = Mx*z_syndrome where z_syndrome = Hx*z_error
            z_syndrome_repaired = self._meta_code_decoding(
                syndrome_w_err=z_syndrome_w_err,
                meta_pcm=self.Mx,
                m_bp=self.mz_bp,
            )
        else:
            z_syndrome_repaired = np.copy(z_syndrome_w_err).astype(np.int32)

        if self.z_meta and self.Mz is not None:
            x_syndrome_repaired = self._meta_code_decoding(
                syndrome_w_err=x_syndrome_w_err,
                meta_pcm=self.Mz,
                m_bp=self.mx_bp,
            )
        else:
            x_syndrome_repaired = np.copy(x_syndrome_w_err).astype(np.int32)

        if self.analog_tg:  # decode with analog tg method - update channel probs to init analog nodes
            x_decoded = self._analog_tg_decoding(
                decoder=self.x_abpd,
                hard_syndrome=x_syndrome_repaired,
                analog_syndrome=x_syndrome_w_err,
                bit_err_channel=self.z_bit_err_channel + self.y_bit_err_channel,
                sigma=self.sigma_x,
            )
            z_decoded = self._analog_tg_decoding(
                decoder=self.z_abpd,
                hard_syndrome=z_syndrome_repaired,
                analog_syndrome=z_syndrome_w_err,
                bit_err_channel=self.x_bit_err_channel + self.y_bit_err_channel,
                sigma=self.sigma_z,
            )
        else:
            x_decoded = self.x_bpd.decode(x_syndrome_repaired)
            z_decoded = self.z_bpd.decode(z_syndrome_repaired)

        return x_decoded, z_decoded

    def _meta_code_decoding(
        self,
        syndrome_w_err: NDArray[np.float64],
        meta_pcm: NDArray[np.int32],
        m_bp: Any,  # noqa: ANN401
    ) -> NDArray[np.int32]:
        """Decodes the noisy syndrome using the meta code.

        If analog_info or analog_tg is activated the analog syndrome is thresholded to binary to be able to use
        the meta code. The binary syndrome is repaired according to the meta code and a binary,
        repaired syndrome is returned.
        """
        if self.analog_info or self.analog_tg:
            syndrome_w_err_binary = get_binary_from_analog(syndrome_w_err)
        else:
            syndrome_w_err_binary = np.copy(syndrome_w_err)

        meta_syndrome = (meta_pcm @ syndrome_w_err_binary) % 2
        meta_decoded = m_bp.decode(meta_syndrome)
        syndrome_w_err_binary += meta_decoded  # repair syndrome
        syndrome_w_err_binary %= 2

        return syndrome_w_err_binary

    def _analog_tg_decoding(
        self,
        decoder: Any,  # noqa: ANN401
        analog_syndrome: NDArray[np.float64],
        hard_syndrome: NDArray[np.int32],
        bit_err_channel: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.int32]:
        """Decodes the noisy analog syndrome using the analog tanner graph and BPOSD.

        First, the channel probabilities need to be set according to the analog syndrome s.t. the decoder is
        initialized properly.
        Only the first n bits of the decoding estimate returned by the decoder are true estimates,
        the other bits are acting as 'virtual checks' and thus excluded from the result.
        The bit_err_channel is supposed to be the error channel for the n physical bits as usual.
        """
        analog_channel = get_virtual_check_init_vals(analog_syndrome, sigma)
        decoder.update_channel_probs(np.hstack((bit_err_channel, analog_channel)))
        # only first n bit are data so only return those
        return decoder.decode(hard_syndrome)[: self.n]

    def _single_stage_setup(
        self,
    ) -> None:
        """Sets up the single stage decoding.

        * BPOSD decoders for the single-stage check matrices (cf Higgot & Breuckmann) are setup in case
        there is a meta code for the respective side.
        * BPOSD decoders for the check matrices Hx/Hz are set up for the last, perfect round
        * In case analog_tg is activated, BPOSD for the analog tanner graph is setup
        * If analog_info is active, SI-decoder is used to decode single-stage matrices and standard check matrices
        instead of BPOSD.
        """
        # X-syndrome sx = Hz*ex
        # x_syndr_error == error on Hz => corrected with Mz

        # setup single stage matrices (Oscar&Niko method) H = [[H, Im],    H ~ mxn, I ~ mxm, M ~ lxm
        #                                                      [0, M]]
        if self.z_meta and self.Mz is not None:
            self.ss_z_pcm = build_single_stage_pcm(self.Hz, self.Mz)
        else:
            self.ss_z_pcm = None
        if self.x_meta and self.Mx is not None:
            self.ss_x_pcm = build_single_stage_pcm(self.Hx, self.Mx)
        else:
            self.ss_x_pcm = None

        # X-checks := (Hx|Mx) => Influenced by Z-syndrome error rate
        # ss_x_bpd used to decode X bit errors using Z-side check matrices
        # single shot bp operates on the single shot pcm (Hz|Mz)
        # hence we need x bit error rate and x syndrome error rate
        x_err_channel = self.x_bit_err_channel + self.y_bit_err_channel
        if self.ss_z_pcm is not None:
            self.ss_x_bpd = self.get_decoder(
                pcm=self.ss_z_pcm,
                channel_probs=np.hstack((x_err_channel, self.x_syndr_error_channel)),
                # sigma=self.sigma_x,
                # analog_info=self.analog_info,
            )

        if self.analog_tg:
            self.x_abpd = self.get_decoder(
                pcm=self.z_apcm,
                # second part dummy, needs to be updated for each syndrome
                channel_probs=np.hstack((x_err_channel, np.zeros(self.Hz.shape[0]))),
                # cutoff=self.cutoff,
                # analog_info=False,
                # sigma not needed, since we apply BPOSD
            )
        else:
            self.x_abpd = None
        self.x_bpd = self.get_decoder(
            pcm=self.Hz,
            channel_probs=x_err_channel,
            # cutoff=self.cutoff,
            # analog_info=self.analog_info,
        )

        z_err_channel = self.z_bit_err_channel + self.y_bit_err_channel
        if self.ss_x_pcm is not None:
            self.ss_z_bpd = self.get_decoder(
                pcm=self.ss_x_pcm,
                channel_probs=np.hstack((z_err_channel, self.z_syndr_error_channel)),
                # cutoff=self.cutoff,
                # analog_info=self.analog_info,
                # sigma=self.sigma_z,
            )

        if self.analog_tg:
            self.z_abpd = self.get_decoder(
                pcm=self.x_apcm,
                # second part dummy, needs to be updated for each syndrome
                channel_probs=np.hstack((z_err_channel, np.zeros(self.Hx.shape[0]))),
                # cutoff=self.cutoff,
                # analog_info=self.analog_info,
                # sigma not needed, since we apply BPOSD
            )
        else:
            self.z_abpd = None
        self.z_bpd = self.get_decoder(
            pcm=self.Hx,
            channel_probs=z_err_channel,
            # cutoff=self.cutoff,
            # analog_info=self.analog_info,
            # sigma=self.sigma_z,
        )

    def _two_stage_setup(
        self,
    ) -> None:
        """Sets up the two stage decoding.

            * In case meta codes are present, BPOSD decoders for the meta codes are setup
            * In case analo_tg is active, BPOSD decoders for the analog tanner graph are setup
            * Additionally, BPOSD for Hx, Hz are setup for the final round
        :return:
        """
        # used to decode errors in Hz == x-syndrome errors with x_synd_error rate
        if self.z_meta and self.Mz is not None:
            self.mx_bp = self.get_decoder(
                pcm=self.Mz,
                channel_probs=self.x_syndr_error_channel,
                # cutoff=self.cutoff,
                # analog_info=False,  # =False and sigma not needed, since we don't have soft info for meta code
            )

        # used to decode errors in Hx
        if self.x_meta and self.Mx is not None:
            self.mz_bp = self.get_decoder(
                pcm=self.Mx,
                channel_probs=self.z_syndr_error_channel,
                # cutoff=self.cutoff,
                # analog_info=False,  # =False and sigma not needed, since we don't have soft info for meta code
            )

        x_err_channel = self.x_bit_err_channel + self.y_bit_err_channel
        if self.analog_tg:
            self.x_abpd = self.get_decoder(
                pcm=self.z_apcm,
                channel_probs=np.hstack((x_err_channel, np.zeros(self.Hz.shape[0]))),
                # second part dummy, needs to be updated for each syndrome
                # cutoff=self.cutoff,
                # analog_info=self.analog_info,
                # sigma not needed, since we apply BPOSD
            )
        self.x_bpd = self.get_decoder(
            pcm=self.Hz,
            channel_probs=x_err_channel,
            # cutoff=self.cutoff,
            # analog_info=self.analog_info,
        )

        z_err_channel = self.z_bit_err_channel + self.y_bit_err_channel
        if self.analog_tg:
            self.z_abpd = self.get_decoder(
                pcm=self.x_apcm,
                # second part dummy, needs to be updated for each syndrome
                channel_probs=np.hstack((z_err_channel, np.zeros(self.Hx.shape[0]))),
                # cutoff=self.cutoff,
                # analog_info=self.analog_info,
                # sigma not needed, since we apply BPOSD
            )
        self.z_bpd = self.get_decoder(
            pcm=self.Hx,
            channel_probs=z_err_channel,
            # cutoff=self.cutoff,
            # analog_info=self.analog_info,
            # sigma not needed, since we don't have analog info for meta code
        )

    def get_decoder(
        self,
        pcm: NDArray[np.int32],
        channel_probs: NDArray[np.float64],
        # cutoff: int = 0,
        # sigma: float = 0.0,
        # analog_info: bool = False,
    ) -> Any:  # noqa: ANN401
        """Initialize decoder objects.

        If analog_info is activated, the SoftInfoBpDecoder is used instead of the BPOSD decoder.
        Note that analog_info and analog_tg cannot be used simultaneously.
        """
        return bposd_decoder(
            parity_check_matrix=pcm,
            channel_probs=channel_probs,
            max_iter=self.bp_params.max_bp_iter,
            bp_method=self.bp_params.bp_method,
            osd_order=self.bp_params.osd_order,
            osd_method=self.bp_params.osd_method,
            ms_scaling_factor=self.bp_params.ms_scaling_factor,
        )

    def construct_analog_pcms(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Constructs apcm = [H | I_m] where I_m is the m x m identity matrix."""
        return np.hstack([self.Hx, np.identity(self.Hx.shape[0], dtype=np.int32)]), np.hstack([
            self.Hz,
            np.identity(self.Hz.shape[0], dtype=np.int32),
        ])

    def save_results(
        self,
        x_success_cnt: int,
        z_success_cnt: int,
        runs: int,
        x_bp_iters: int,
        z_bp_iters: int,
    ) -> dict[str, Any]:
        """Saves the results of the simulation to a json file."""
        x_ler, x_ler_eb, x_wer, x_wer_eb = calculate_error_rates(x_success_cnt, runs, self.code_params)
        z_ler, z_ler_eb, z_wer, z_wer_eb = calculate_error_rates(z_success_cnt, runs, self.code_params)

        output = {
            "code_K": self.code_params["k"],
            "code_N": self.code_params["n"],
            "nr_runs": runs,
            "pers": self.data_err_rate,
            "sers": self.syndr_err_rate,
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
            "avg_x_bp_iter": x_bp_iters / runs,
            "avg_z_bp_iter": z_bp_iters / runs,
            "decoding_time": self._total_decoding_time,
        }

        output.update(self.input_values)
        output["bias"] = replace_inf(output["bias"])  # type: ignore[assignment, arg-type]

        with Path(self.outfile).open(encoding=locale.getpreferredencoding(False)) as f:
            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    indent=4,
                    default=lambda o: o.__dict__,
                )
            )
        return output

    def run(self, samples: int) -> dict[str, Any]:
        """Run the simulation."""
        x_success_cnt = 0
        z_success_cnt = 0
        x_bp_iters = 0
        z_bp_iters = 0
        for runs in range(1, samples + 1):
            out = self._single_sample()
            if not out[0]:
                x_success_cnt += 1
            if not out[1]:
                z_success_cnt += 1
            x_bp_iters += self.x_bp_iters
            z_bp_iters += self.z_bp_iters

            if runs % self.save_interval == 0:
                self.save_results(x_success_cnt, z_success_cnt, runs, x_bp_iters=x_bp_iters, z_bp_iters=z_bp_iters)
                if is_converged(
                    x_success_cnt,
                    z_success_cnt,
                    runs,
                    self.code_params,
                    self.eb_precision,
                ):
                    print("Result has converged.")  # noqa: T201
                    break

        x_ler, x_ler_eb, x_wer, x_wer_eb = calculate_error_rates(x_success_cnt, runs, self.code_params)
        z_ler, z_ler_eb, z_wer, z_wer_eb = calculate_error_rates(z_success_cnt, runs, self.code_params)
        avg_x_bp_iter = x_bp_iters / runs
        avg_z_bp_iter = z_bp_iters / runs
        output = {
            "code_K": self.code_params["k"],
            "code_N": self.code_params["n"],
            "nr_runs": runs,
            "pers": self.data_err_rate,
            "sers": self.syndr_err_rate,
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
            "avg_x_bp_iters": avg_x_bp_iter,
            "avg_z_bp_iters": avg_z_bp_iter,
            "decoding_time": self._total_decoding_time,
        }

        output.update(self.input_values)

        self.save_results(x_success_cnt, z_success_cnt, runs, x_bp_iters=x_bp_iters, z_bp_iters=z_bp_iters)
        return output
