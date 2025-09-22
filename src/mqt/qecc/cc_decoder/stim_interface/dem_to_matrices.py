# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""contains functionality to convert a detector error model to check matrices."""

# Author: Oscar Higgott https://github.com/oscarhiggott/stimbposd/blob/main/src/stimbposd/dem_to_matrices.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix

if TYPE_CHECKING:
    import stim
    from numpy.typing import NDArray


def iter_set_xor(set_list: list[list[int]]) -> frozenset[int]:
    """Computes XOR between sets."""
    out: set[int] = set()
    for x in set_list:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)


def dict_to_csc_matrix(elements_dict: dict[int, frozenset[int]], shape: tuple[int, int]) -> csc_matrix:
    """Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict`.

    Parameters
    ----------
    elements_dict : Dict[int, FrozenSet[int]]
        A dictionary giving the indices of nonzero rows in each column. `elements_dict[i]` is a FrozenSet of ints
        giving the indices of nonzero rows in column `i`.
    shape : Tuple[int, int]
        The dimensions of the matrix to be returned

    Returns:
    -------
    scipy.sparse.csc_matrix
        The `scipy.sparse.csc_matrix` check matrix defined by `elements_dict` and `shape`
    """
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


@dataclass
class DemMatrices:
    """Class capturing the check matrices associated to a DEM."""

    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    edge_check_matrix: csc_matrix
    edge_observables_matrix: csc_matrix
    hyperedge_to_edge_matrix: csc_matrix
    priors: NDArray[np.float64]


def detector_error_model_to_check_matrices(
    dem: stim.DetectorErrorModel, allow_undecomposed_hyperedges: bool = True
) -> DemMatrices:
    """Convert a `stim.DetectorErrorModel` into a `DemMatrices` object.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        A stim DetectorErrorModel
    allow_undecomposed_hyperedges: bool
        If True, don't raise an exception if a hyperedge is not decomposable. Instead, the hyperedge `h` is still added
        to the `DemMatrices.check_matrix`, `DemMatrices.observables_matrix` and `DemMatrices.priors` but it will not
        have any edges in its decomposition in `DemMatrices.hyperedge_to_edge_matrix[:, h]`.

    Returns:
    -------
    DemMatrices
        A collection of matrices representing the stim DetectorErrorModel
    """
    hyperedge_ids: dict[frozenset[int], int] = {}
    edge_ids: dict[frozenset[int], int] = {}
    hyperedge_obs_map: dict[int, frozenset[int]] = {}
    edge_obs_map: dict[int, frozenset[int]] = {}
    priors_dict: dict[int, float] = {}
    hyperedge_to_edge: dict[int, frozenset[int]] = {}

    def handle_error(prob: float, detectors: list[list[int]], observables: list[list[int]]) -> None:
        hyperedge_dets = iter_set_xor(detectors)
        hyperedge_obs = iter_set_xor(observables)

        if hyperedge_dets not in hyperedge_ids:
            hyperedge_ids[hyperedge_dets] = len(hyperedge_ids)
            priors_dict[hyperedge_ids[hyperedge_dets]] = 0.0
        hid = hyperedge_ids[hyperedge_dets]
        hyperedge_obs_map[hid] = hyperedge_obs
        priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])

        eids = []
        for i in range(len(detectors)):
            e_dets = frozenset(detectors[i])
            e_obs = frozenset(observables[i])

            if len(e_dets) > 2:
                if not allow_undecomposed_hyperedges:
                    msg = (
                        "A hyperedge error mechanism was found that was not decomposed into edges. "
                        "This can happen if you do not Set `decompose_errors=True` as required when "
                        "calling `circuit.detector_error_model`."
                    )
                    raise ValueError(msg)
                continue

            if e_dets not in edge_ids:
                edge_ids[e_dets] = len(edge_ids)
            eid = edge_ids[e_dets]
            eids.append(eid)
            edge_obs_map[eid] = e_obs

        if hid not in hyperedge_to_edge:
            hyperedge_to_edge[hid] = frozenset(eids)

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: list[list[int]] = [[]]
            frames: list[list[int]] = [[]]
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets[-1].append(t.val)
                elif t.is_logical_observable_id():
                    frames[-1].append(t.val)
                elif t.is_separator():
                    dets.append([])
                    frames.append([])
            handle_error(p, dets, frames)
        elif instruction.type in {"detector", "logical_observable"}:
            pass
        else:
            raise NotImplementedError
    check_matrix = dict_to_csc_matrix(
        {v: k for k, v in hyperedge_ids.items()},
        shape=(dem.num_detectors, len(hyperedge_ids)),
    )
    observables_matrix = dict_to_csc_matrix(hyperedge_obs_map, shape=(dem.num_observables, len(hyperedge_ids)))
    priors = np.zeros(len(hyperedge_ids))
    for i, p in priors_dict.items():
        priors[i] = p
    hyperedge_to_edge_matrix = dict_to_csc_matrix(hyperedge_to_edge, shape=(len(edge_ids), len(hyperedge_ids)))
    edge_check_matrix = dict_to_csc_matrix(
        {v: k for k, v in edge_ids.items()}, shape=(dem.num_detectors, len(edge_ids))
    )
    edge_observables_matrix = dict_to_csc_matrix(edge_obs_map, shape=(dem.num_observables, len(edge_ids)))
    return DemMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        edge_check_matrix=edge_check_matrix,
        edge_observables_matrix=edge_observables_matrix,
        hyperedge_to_edge_matrix=hyperedge_to_edge_matrix,
        priors=priors,
    )
