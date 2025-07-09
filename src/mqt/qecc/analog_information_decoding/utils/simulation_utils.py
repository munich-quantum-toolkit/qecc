# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulation utilities for analog information decoding."""

from __future__ import annotations

import json
import locale
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ldpc.mod2 import rank
from scipy.special import erfc, erfcinv

from .data_utils import BpParams, calculate_error_rates, replace_inf

if TYPE_CHECKING:
    from numpy.typing import NDArray


def set_seed(value: float) -> None:
    """The appropriate way to set seeds when numba is used."""
    np.random.seed(value)  # noqa: NPY002


def alist2numpy(fname: str) -> NDArray[np.int32]:  # current original implementation is buggy
    """Converts an alist file to a numpy array."""
    alist_file: NDArray[np.str_] = np.loadtxt(fname, delimiter=",", dtype=str)
    matrix_dimensions = alist_file[0].split()
    m = int(matrix_dimensions[0])
    n = int(matrix_dimensions[1])

    mat: NDArray[np.int32] = np.zeros((m, n), dtype=np.int32)

    for i in range(m):
        columns = [item for item in alist_file[i + 4].split() if item.isdigit()]
        columns_two: NDArray[np.int32] = np.array(columns, dtype=np.int32)
        columns_two -= 1  # convert to zero indexing
        mat[i, columns_two] = 1

    return mat


# Rewrite such that call signatures of check_logical_err_h
# and check_logical_err_l are identical
def check_logical_err_h(
    check_matrix: NDArray[np.int_], original_err: NDArray[np.int_], decoded_estimate: NDArray[np.int_]
) -> bool:
    """Checks if the residual error is a logical error."""
    _, n = check_matrix.shape

    # compute residual err given original err
    residual_err: NDArray[np.int32] = np.zeros((n, 1), dtype=np.int32)
    for i in range(n):
        residual_err[i][0] = original_err[i] ^ decoded_estimate[i]

    ht = np.transpose(check_matrix)

    htr = np.append(ht, residual_err, axis=1)

    rank_ht = rank(check_matrix)  # rank A = rank A.T

    rank_htr = rank(htr)

    return (rank_ht < rank_htr) is True


# L is a numpy array, residual_err is vector s.t. dimensions match
# residual_err is a logical iff it commutes with logicals of other side
# i.e., an X residal is a logical iff it commutes with at least one Z logical and
# an Z residual is a logical iff it commutes with at least one Z logical
# Hence, L must be of same type as H and of different type than residual_err
def is_logical_err(logicals: NDArray[np.int_], residual_err: NDArray[np.int_]) -> bool:
    """Checks if the residual error is a logical error.

    :returns: True if its logical error, False otherwise (is a stabilizer).
    """
    l_check = (logicals @ residual_err) % 2
    return bool(l_check.any())  # check all zeros


# adapted from https://github.com/quantumgizmos/bp_osd/blob/a179e6e86237f4b9cc2c952103fce919da2777c8/src/bposd/css_decode_sim.py#L430
# and https://github.com/MikeVasmer/single_shot_3D_HGP/blob/master/sim_scripts/single_shot_hgp3d.cpp#L207
# channel_probs = [x,y,z], residual_err = [x,z]
def generate_err(
    nr_qubits: int,
    channel_probs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    residual_err: list[NDArray[np.int_]],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Computes error vector with X and Z part given channel probabilities and residual error.

    Assumes that residual error has two equally sized parts.
    """
    error_x = residual_err[0]
    error_z = residual_err[1]
    channel_probs_x = channel_probs[0]
    channel_probs_y = channel_probs[1]
    channel_probs_z = channel_probs[2]
    residual_err_x = residual_err[0]
    residual_err_z = residual_err[1]

    for i in range(nr_qubits):
        rand = np.random.default_rng().random()  # this returns a random float in [0,1)
        # e.g. if err channel is p = 0.3, then an error will be applied if rand < p
        if rand < channel_probs_z[i]:  # if probability for z error high enough, rand < p, apply
            # if there is a z error on the i-th bit, flip the bit but take residual error into account
            # nothing on x part - probably redundant anyways
            error_z[i] = (residual_err_z[i] + 1) % 2
        elif (  # if p = 0.3 then 0.3 <= rand < 0.6 is the same sized interval as rand < 0.3
            channel_probs_z[i] <= rand < (channel_probs_z[i] + channel_probs_x[i])
        ):
            # X error
            error_x[i] = (residual_err_x[i] + 1) % 2
        elif (  # 0.6 <= rand < 0.9
            (channel_probs_z[i] + channel_probs_x[i])
            <= rand
            < (channel_probs_x[i] + channel_probs_y[i] + channel_probs_z[i])
        ):
            # y error == both x and z error
            error_z[i] = (residual_err_z[i] + 1) % 2
            error_x[i] = (residual_err_x[i] + 1) % 2

    return error_x, error_z


def get_analog_llr(analog_syndrome: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Computes analog LLRs given analog syndrome and sigma."""
    if sigma <= 0.0:
        return np.zeros_like(analog_syndrome).astype(np.float64)
    return (2 * analog_syndrome) / (sigma**2)


def get_sigma_from_syndr_er(ser: float) -> float:
    """For analog Cat syndrome noise we need to convert the syndrome error model as described in the paper.

    :return: sigma.
    """
    if ser == 0.0:
        return 0.0
    return float(1 / np.sqrt(2) / (erfcinv(2 * ser)))  # see Eq. cref{eq:perr-to-sigma} in our paper


def get_error_rate_from_sigma(sigma: float) -> float:
    """For analog Cat syndrome noise we need to convert the syndrome error model as described in the paper.

    :return: sigma.
    """
    if sigma == 0.0:
        return 0.0
    return float(0.5 * erfc(1 / np.sqrt(2 * sigma**2)))  # see Eq. cref{eq:perr-to-sigma} in our paper


def get_virtual_check_init_vals(noisy_syndr: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Computes a vector of values v_i from the noisy syndrome bits y_i s.t.

    BP initializes the LLRs l_i of the analog nodes with the
    analog info values (see paper section). v_i := 1/(e^{y_i}+1).
    """
    if sigma <= 0.0:
        return np.zeros_like(noisy_syndr).astype(np.float64)
    llrs = get_analog_llr(noisy_syndr, sigma)
    return np.array(1 / (np.exp(np.abs(llrs)) + 1))


def generate_syndr_err(channel_probs: NDArray[np.float64]) -> NDArray[np.int32]:
    """Generates a random error vector given the error channel probabilities."""
    error: NDArray[np.int32] = np.zeros_like(channel_probs, dtype=np.int32)

    for i, p in np.ndenumerate(channel_probs):
        rand = np.random.default_rng().random()

        if rand < p:
            error[i] = 1

    return error


def get_noisy_analog_syndrome(perfect_syndr: NDArray[np.int_], sigma: float) -> NDArray[np.float64]:
    """Generate noisy analog syndrome vector given the perfect syndrome and standard deviation sigma (~ noise strength).

    Assumes perfect_syndr has entries in {0,1}.
    """
    # compute signed syndrome: 1 = check satisfied, -1 = check violated. float needed for Gaussian sampling call
    sgns: NDArray[np.float64] = np.where(
        perfect_syndr == 0.0,
        np.ones_like(perfect_syndr),
        np.full_like(perfect_syndr, -1.0),
    ).astype(float)

    # sample from Gaussian with zero mean and sigma std. dev: ~N(0, sigma_sq)
    return np.array(np.random.default_rng().normal(loc=sgns, scale=sigma, size=perfect_syndr.shape)).astype(np.float64)


def error_channel_setup(
    error_rate: float, xyz_error_bias: NDArray[np.float64], nr_qubits: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Set up an error_channel given the physical error rate, bias, and number of bits."""
    xyz_error_bias = np.array(xyz_error_bias)
    if xyz_error_bias[0] == np.inf:
        px = error_rate
        py = 0.0
        pz = 0.0
    elif xyz_error_bias[1] == np.inf:
        px = 0.0
        py = error_rate
        pz = 0.0
    elif xyz_error_bias[2] == np.inf:
        px = 0.0
        py = 0.0
        pz = error_rate
    else:
        px, py, pz = (
            error_rate * xyz_error_bias / np.sum(xyz_error_bias)
        )  # Oscar only considers X or Z errors. For reproducibility remove normalization

    channel_probs_x = np.ones(nr_qubits) * px
    channel_probs_z = np.ones(nr_qubits) * pz
    channel_probs_y = np.ones(nr_qubits) * py

    return channel_probs_x, channel_probs_y, channel_probs_z


def build_single_stage_pcm(pcm: NDArray[np.int_], meta: NDArray[np.int_]) -> NDArray[np.int_]:
    """Build the single statge parity check matrix."""
    id_r = np.identity(meta.shape[1])
    zeros = np.zeros((meta.shape[0], pcm.shape[1]))
    return np.block([[pcm, id_r], [zeros, meta]])


def get_signed_from_binary(binary_syndrome: NDArray[np.int_]) -> NDArray[np.int_]:
    """Maps the binary vector with {0,1} entries to a vector with {-1,1} entries."""
    return np.where(
        binary_syndrome == 0,
        np.full(shape=binary_syndrome.shape, fill_value=1),
        np.full(shape=binary_syndrome.shape, fill_value=-1),
    )


def get_binary_from_analog(analog_syndrome: NDArray[np.float64]) -> NDArray[np.int32]:
    """Returns the thresholded binary vector.

    Since in {-1,+1} notation -1 indicates a check violation, we map values <= 0 to 1 and values > 0 to 0.
    """
    return np.where(analog_syndrome <= 0.0, 1, 0).astype(np.int32)


def save_results(
    success_cnt: int,
    nr_runs: int,
    p: float,
    s: float,
    input_vals: dict[str, Any],
    outfile: str,
    code_params: dict[str, int],
    bp_params: BpParams | None,
    err_side: str = "X",
    bp_iterations: int | None = None,
) -> dict[str, Any]:
    """Save results of a simulation run to a json file."""
    ler, ler_eb, wer, wer_eb = calculate_error_rates(success_cnt, nr_runs, code_params)

    output: dict[str, Any] = {
        "code_K": code_params["k"],
        "code_N": code_params["n"],
        "nr_runs": nr_runs,
        "pers": p,
        "sers": s,
        f"{err_side}_ler": ler,
        f"{err_side}_ler_eb": ler_eb,
        f"{err_side}_wer": wer,
        f"{err_side}_wer_eb": wer_eb,
        f"{err_side}_success_cnt": success_cnt,
        "avg_bp_iterations": bp_iterations / nr_runs if bp_iterations is not None else 0,
        "bp_params": bp_params,
    }

    output.update(input_vals)
    output["bias"] = replace_inf(output["bias"])
    with Path(outfile).open(mode="w", encoding=locale.getpreferredencoding(False)) as out:
        json.dump(output, out, ensure_ascii=False, indent=4, default=lambda o: o.__dict__)
    return output
