# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sinter integration of the maxsat decoder."""

from __future__ import annotations

import locale
import pathlib
from typing import TYPE_CHECKING

import stim
from sinter import CompiledDecoder, Decoder

from mqt.qecc.cc_decoder.stim_interface.max_sat_stim_decoder import MaxSatStim

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class SinterCompiledDecoderMaxSat(CompiledDecoder):  # type: ignore[misc]
    """MaxSAT decoder instantiation as CompiledDecoder."""

    def __init__(self, decoder: MaxSatStim, d: int | None = None, p: float | None = None) -> None:
        """Constructor initializing compiled decoder with stim decoder.

        If both ``d`` and ``p`` are given, each call of :meth:`decode_shots_bit_packed` appends a line to
        ``convergence_rate.txt`` in the working directory. The line holds ``d``, ``p`` and the running numbers of
        converged and not converged shots.

        Args:
            decoder: The MaxSAT decoder for the detector error model.
            d: The code distance for the convergence log.
            p: The error rate for the convergence log.

        Raises:
            ValueError: If only one of ``d`` and ``p`` is given.
        """
        if (d is None) != (p is None):
            msg = "Give both d and p to log the convergence, or neither."
            raise ValueError(msg)
        self.decoder = decoder
        self.d = d
        self.p = p
        self.measure_convergence = d is not None
        self.convergence_cnt = 0
        self.not_convergence_cnt = 0

    def decode_shots_bit_packed(
        self,
        bit_packed_detection_event_data: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Decode bitpacked shots from sinter simulation using batch decoder."""
        predictions, converged_cnt, not_converged_cnt = self.decoder.decode_batch(
            shots=bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        if self.measure_convergence:
            self.convergence_cnt += converged_cnt
            self.not_convergence_cnt += not_converged_cnt
            with pathlib.Path("convergence_rate.txt").open("a", encoding=locale.getpreferredencoding(False)) as f:
                f.write(
                    str(self.d)
                    + " "
                    + str(self.p)
                    + " "
                    + str(self.convergence_cnt)
                    + " "
                    + str(self.not_convergence_cnt)
                    + "\n"
                )

        return predictions


class SinterDecoderMaxSat(Decoder):  # type: ignore[misc]
    """Sinter implementation of MaxSAT decoder."""

    def __init__(self, d: int | None = None, p: float | None = None) -> None:
        """Init sinter decoder.

        Args:
            d: The code distance for the convergence log. See :class:`SinterCompiledDecoderMaxSat`.
            p: The error rate for the convergence log. See :class:`SinterCompiledDecoderMaxSat`.

        Raises:
            ValueError: If only one of ``d`` and ``p`` is given.
        """
        if (d is None) != (p is None):
            msg = "Give both d and p to log the convergence, or neither."
            raise ValueError(msg)
        self.d = d
        self.p = p

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> SinterCompiledDecoderMaxSat:
        """Return sinter compiled decoder initialized with the given DEM."""
        return SinterCompiledDecoderMaxSat(MaxSatStim(model=dem), d=self.d, p=self.p)

    def decode_via_files(  # ruff:ignore[no-self-use]
        self,
        num_shots: int,  # ruff:ignore[unused-method-argument]
        num_dets: int,  # ruff:ignore[unused-method-argument]
        num_obs: int,  # ruff:ignore[unused-method-argument]
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,  # ruff:ignore[unused-method-argument]
    ) -> None:
        """Performs decoding by reading problems from, and writing solutions to, file paths.

        Args:
            num_shots: The number of times the circuit was sampled. The number of problems
                to be solved.
            num_dets: The number of detectors in the circuit. The number of detection event
                bits in each shot.
            num_obs: The number of observables in the circuit. The number of predicted bits
                in each shot.
            dem_path: The file path where the detector error model should be read from,
                e.g. using `stim.DetectorErrorModel.from_file`. The error mechanisms
                specified by the detector error model should be used to configure the
                decoder.
            dets_b8_in_path: The file path that detection event data should be read from.
                Note that the file may be a named pipe instead of a fixed size object.
                The detection events will be in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of detection events per shot is available via the `num_dets`
                argument or via the detector error model at `dem_path`.
            obs_predictions_b8_out_path: The file path that decoder predictions must be
                written to. The predictions must be written in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of observables per shot is available via the `num_obs` argument or
                via the detector error model at `dem_path`.
            tmp_dir: Any temporary files generated by the decoder during its operation MUST
                be put into this directory. The reason for this requirement is because
                sinter is allowed to kill the decoding process without warning, without
                giving it time to clean up any temporary objects. All cleanup should be done
                via sinter deleting this directory after killing the decoder.
        """
        dem = stim.DetectorErrorModel.from_file(dem_path)
        max_sat = MaxSatStim(
            model=dem,
        )
        shots = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        )
        predictions, _, _ = max_sat.decode_batch(shots)
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )


def sinter_decoders(d: int | None = None, p: float | None = None) -> dict[str, Decoder]:
    """Return a dictionary of the available sinter decoders.

    Args:
        d: The code distance for the convergence log. See :class:`SinterCompiledDecoderMaxSat`.
        p: The error rate for the convergence log. See :class:`SinterCompiledDecoderMaxSat`.

    Returns:
        The decoders by name.
    """
    return {"maxsat": SinterDecoderMaxSat(d=d, p=p)}
