# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Implementation of the Stim decoder for the MaxSat algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..decoder import LightsOut
from .dem_to_matrices import detector_error_model_to_check_matrices

if TYPE_CHECKING:
    import stim
    from numpy.typing import NDArray


class MaxSatStim:
    """MaxSat stim decoder implementation."""

    def __init__(self, model: stim.DetectorErrorModel) -> None:
        """Class for decoding stim circuits using the LightsOut MaxSAT decoder.

        Parameters.
        ----------
        model : stim.DetectorErrorModel
            The detector error model of the stim circuit to be decoded
        """
        self._matrices = detector_error_model_to_check_matrices(model, allow_undecomposed_hyperedges=True)
        self.num_detectors = model.num_detectors
        self.num_errors = model.num_errors
        qtf, ftq = self.check_matrix_to_adj_lists(self._matrices.check_matrix)
        self.problem = LightsOut(ftq, qtf)
        self.observables = self._matrices.observables_matrix
        self.problem.preconstruct_z3_instance(
            weights=np.array([self.weight_function(p) for p in self._matrices.priors])
        )

    @staticmethod
    def check_matrix_to_adj_lists(check_matrix: Any) -> tuple[dict[int, list[int]], dict[int, list[int]]]:  # noqa: ANN401
        """Converts a check matrix to two adjacency lists."""
        qtf: dict[int, list[int]] = {}
        ftq: dict[int, list[int]] = {}
        for row in range(check_matrix.shape[0]):
            for col in range(check_matrix.shape[1]):
                if check_matrix[row, col] == 1:
                    if row not in ftq:
                        ftq[row] = []
                    if col not in qtf:
                        qtf[col] = []
                    qtf[col].append(row)
                    ftq[row].append(col)
        return qtf, ftq

    @staticmethod
    def weight_function(x: np.float64) -> np.float64:
        """Return log likelihood weighting."""
        return cast("np.float64", np.log([(1 - x) / x])[0])

    def decode(self, syndrome: NDArray[int]) -> tuple[NDArray[int], int]:
        """Decode the syndrome and return a prediction of which observables were flipped.

        Parameters
        ----------
        syndrome : NDArray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns:
        -------
        NDArray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        lights = [bool(b) for b in syndrome]
        estimate, converge, _ = self.problem.solve(lights)
        convergence_bool = 0 if converge is False else 1
        return (self._matrices.observables_matrix @ estimate) % 2, convergence_bool

    def decode_batch(
        self,
        shots: NDArray[int],
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> tuple[NDArray[int], int, int]:
        """Decode a batch of shots of syndrome data by iterating over each shot.

        Parameters
        ----------
        shots : NDArray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns:
        -------
        NDArray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        not_converged_cnt = 0
        converged_cnt = 0
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[:, : self.num_detectors]
        predictions = np.zeros((shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool)
        for i in range(shots.shape[0]):
            predictions[i, :], convergence_bool = self.decode(shots[i, :])
            if convergence_bool == 1:
                converged_cnt += 1
            else:
                not_converged_cnt += 1
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions, converged_cnt, not_converged_cnt
