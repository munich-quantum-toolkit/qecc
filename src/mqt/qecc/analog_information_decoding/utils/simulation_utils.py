# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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

from mqt.qecc.mod2 import rank
from mqt.qecc.noise import GaussianReadoutChannel, PauliChannel
from mqt.qecc.noise.sampling import sample_inhomogeneous_pauli

from .data_utils import calculate_error_rates, replace_inf

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .data_utils import BpParams


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
    check_matrix: NDArray[np.int32],
    original_err: NDArray[np.int32],
    decoded_estimate: NDArray[np.int32],
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
def is_logical_err(logicals: NDArray[np.int32], residual_err: NDArray[np.int32]) -> bool:
    """Checks if the residual error is a logical error.

    :returns: True if its logical error, False otherwise (is a stabilizer).
    """
    l_check = (logicals @ residual_err) % 2
    return bool(l_check.any())  # check all zeros


# adapted from https://github.com/quantumgizmos/bp_osd/blob/a179e6e86237f4b9cc2c952103fce919da2777c8/src/bposd/css_decode_sim.py#L430
# and https://github.com/MikeVasmer/single_shot_3D_HGP/blob/bdfb437b2abcfa514997f26be97a711b878448cb/sim_scripts/single_shot_hgp3d.cpp#L207
# channel_probs = [x,y,z], residual_err = [x,z]
def generate_err(
    nr_qubits: int,
    channel_probs: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    residual_err: list[NDArray[np.int32]],
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes error vector with X and Z part given channel probabilities and residual error.

    Assumes that residual error has two equally sized parts.
    """
    if nr_qubits != channel_probs[0].size:
        msg = f"nr_qubits={nr_qubits} does not match channel size {channel_probs[0].size}."
        raise ValueError(msg)
    generator = np.random.default_rng() if rng is None else rng
    return sample_inhomogeneous_pauli(channel_probs, (residual_err[0], residual_err[1]), generator)


def get_analog_llr(analog_syndrome: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Computes analog LLRs given analog syndrome and sigma."""
    if sigma <= 0.0:
        return np.zeros_like(analog_syndrome).astype(np.float64)
    return (2 * analog_syndrome) / (sigma**2)


def get_virtual_check_init_vals(noisy_syndr: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Computes a vector of values v_i from the noisy syndrome bits y_i s.t.

    BP initializes the LLRs l_i of the analog nodes with the analog info values (see paper section). v_i := 1/(e^{y_i}+1).
    """
    if sigma <= 0.0:
        return np.zeros_like(noisy_syndr).astype(np.float64)
    llrs = get_analog_llr(noisy_syndr, sigma)
    return np.array(1 / (np.exp(np.abs(llrs)) + 1))


def generate_syndr_err(channel_probs: NDArray[np.float64], rng: np.random.Generator | None = None) -> NDArray[np.int32]:
    """Generates a random error vector given the error channel probabilities."""
    probabilities = np.asarray(channel_probs, dtype=np.float64)
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        msg = "Syndrome-error probabilities must be finite and between 0 and 1."
        raise ValueError(msg)
    generator = np.random.default_rng() if rng is None else rng
    return np.asarray(generator.random(probabilities.shape) < probabilities, dtype=np.int32)


def get_noisy_analog_syndrome(
    perfect_syndr: NDArray[np.int32], sigma: float, rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Generate noisy analog syndrome vector given the perfect syndrome and standard deviation sigma (~ noise strength).

    Assumes perfect_syndr has entries in {0,1}.
    """
    GaussianReadoutChannel(sigma)  # used as validation
    if not np.all((perfect_syndr == 0) | (perfect_syndr == 1)):
        msg = "A perfect syndrome must contain only binary values."
        raise ValueError(msg)
    sgns: NDArray[np.float64] = np.where(
        np.isclose(perfect_syndr, 0.0, atol=0.0),
        np.ones_like(perfect_syndr),
        np.full_like(perfect_syndr, -1.0),
    ).astype(np.float64)

    generator = np.random.default_rng() if rng is None else rng
    return np.asarray(generator.normal(loc=sgns, scale=sigma, size=perfect_syndr.shape), dtype=np.float64)


def error_channel_setup(error_rate: float, xyz_error_bias: NDArray[np.float64]) -> PauliChannel:
    """Set up the Pauli error channel given the physical error rate and bias."""
    bias_values = tuple(float(value) for value in np.asarray(xyz_error_bias).tolist())
    if len(bias_values) != 3:
        msg = f"xyz_error_bias must contain exactly three values, got {len(bias_values)}."
        raise ValueError(msg)
    return PauliChannel.from_total_probability(error_rate, bias=bias_values)


def build_single_stage_pcm(pcm: NDArray[np.int32], meta: NDArray[np.int32]) -> NDArray[np.int32]:
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
