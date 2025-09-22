# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the functions for the multiround decoding simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymatching import Matching
from scipy.sparse import block_diag, csr_matrix, eye, hstack

from ..utils.simulation_utils import (
    get_virtual_check_init_vals,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def build_multiround_pcm(pcm: NDArray[np.int32], repetitions: int, matrix_format: str = "csr") -> csr_matrix:
    """Builds the multiround parity-check matrix as described in the paper.

    Each row corresponds to a round of measurements, the matrix for r repetitions has the form
    H3D := (H_diag | id_r), where H_diag is the r-diagonal matrix containing the parity-check matrix H and
    id_diag is a staircase block matrix containing two m-identity matrices in each row and column except the first one.
    """
    if not isinstance(pcm, csr_matrix):
        pcm = csr_matrix(pcm)

    pcm_rows, _ = pcm.shape

    # Construct the block of PCMs
    h_3dpcm = block_diag([pcm] * (repetitions + 1), format=matrix_format)

    # Construct the block of identity matrices
    h_3did_diag = block_diag([eye(pcm_rows, format=matrix_format)] * (repetitions + 1), format=matrix_format)

    # Construct the block of identity matrices
    h_3did_offdiag = eye(pcm_rows * (repetitions + 1), k=-pcm_rows, format=matrix_format)

    # Construct the block of identity matrices
    h_3did = h_3did_diag + h_3did_offdiag

    # # hstack the two blocks
    return hstack([h_3dpcm, h_3did], format=matrix_format)


def move_syndrome(syndrome: NDArray[Any], data_type: Any = np.int32) -> NDArray[Any]:  # noqa: ANN401
    """Slides the window one region up, i.e., the syndrome of the first half is overwritten by the second half."""
    region_size = int(syndrome.shape[1] / 2)  # number of rounds in each region, T

    # zero likes syndrome
    new_syndrome = np.zeros(syndrome.shape, dtype=data_type)

    # move the syndromes from the Tth round to the 0th round
    new_syndrome[:, :region_size] = syndrome[:, region_size:]
    return new_syndrome


def get_updated_decoder(
    decoding_method: str,
    decoder: Any,  # noqa: ANN401
    new_channel: NDArray[np.float64],
    h3d: Any | None = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Updates the decoder with the new channel information and returns the updated decoder object."""
    if decoding_method == "bposd":
        decoder.update_channel_probs(new_channel)
        return decoder
    if decoding_method == "matching":
        weights = np.clip(
            np.log((1 - new_channel) / new_channel),
            a_min=-16777215,
            a_max=16777215,
        )
        return Matching(h3d, weights=weights)
    msg = "Unknown decoding method"
    raise ValueError(msg, decoding_method)


def decode_multiround(
    syndrome: NDArray[np.int32],
    pcm: NDArray[np.int32],
    decoder: Any,  # noqa: ANN401
    channel_probs: NDArray[np.float64],  # needed for matching decoder does not have an update weights method
    repetitions: int,
    analog_syndr: NDArray[np.float64] | None,
    last_round: bool = False,
    check_block_size: int = 0,
    sigma: float = 0.0,
    h3d: csr_matrix | None = None,  # needed for matching decoder
    decoding_method: str = "bposd",  # bposd or matching
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64], int]:
    """Overlapping window decoding.

    First, we compute the difference syndrome from the recorded syndrome of each measurement round for all measurement
    rounds of the current window (consisting of two regions with equal size).
    Then, we apply the correction returned from the decoder on the first region (commit region).
    We then propagate the syndrome through the whole window (i.e., to the end of the second region).
    """
    analog_tg = analog_syndr is not None
    # convert syndrome to difference syndrome
    diff_syndrome = syndrome.copy()
    diff_syndrome[:, 1:] = (syndrome[:, 1:] - syndrome[:, :-1]) % 2
    bp_iter = 0

    region_size = repetitions // 2  # assumes repetitions is even

    if analog_tg:
        # If we have analog information, we use it to initialize the time-like syndrome nodes, which are defined
        # in the block of the H3D matrix after the diagonal H block.
        analog_init_vals = get_virtual_check_init_vals(analog_syndr.flatten("F"), sigma)  # type: ignore[union-attr]

        new_channel = np.hstack((channel_probs[:check_block_size], analog_init_vals))

        # in the last round, we have a perfect syndrome round to make sure we're in the codespace
        if last_round:
            new_channel[-pcm.shape[0] :] = 1e-15

        decoder = get_updated_decoder(decoding_method, decoder, new_channel, h3d)

    elif last_round:
        new_channel = np.copy(channel_probs)
        new_channel[-pcm.shape[0] :] = 1e-15

        decoder = get_updated_decoder(decoding_method, decoder, new_channel, h3d)

    decoded = decoder.decode(diff_syndrome.flatten("F"))

    if decoding_method == "bposd":
        bp_iter = decoder.iter

    # extract space correction, first repetitions * n entire
    space_correction = decoded[: pcm.shape[1] * repetitions].reshape((repetitions, pcm.shape[1])).T
    # extract time correction

    if last_round is False:
        # this corresponds to the decoding on the second block of the H3D matrix
        time_correction = decoded[pcm.shape[1] * repetitions :].reshape((repetitions, pcm.shape[0])).T

        # append time correction with zeros
        time_correction = np.hstack((time_correction, np.zeros((pcm.shape[0], 1), dtype=np.int32)))

        # correct only in the commit region
        decoded = (np.cumsum(space_correction, 1) % 2)[:, region_size - 1]

        #  get the syndrome according to the correction
        corr_syndrome = (pcm @ decoded) % 2

        # propagate the syndrome correction through the tentative region
        syndrome[:, region_size:] = ((syndrome[:, region_size:] + corr_syndrome[:, None]) % 2).astype(np.int32)

        # apply the time correction of round region_size - 1 to the syndrome at the beginning of the tentative region
        syndrome[:, region_size] = ((syndrome[:, region_size] + time_correction[:, region_size - 1]) % 2).astype(
            np.int32
        )

    else:
        # correct in the commit and tentative region as the last round stabilizer is perfect
        decoded = (np.cumsum(space_correction, 1) % 2)[:, -1]

    return decoded.astype(np.int32), syndrome, analog_syndr, bp_iter
