# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""tests for the color code stim decoder main functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

from mqt.qecc.cc_decoder.stim_interface.max_sat_sinter_decoder import (
    SinterCompiledDecoderMaxSat,
    SinterDecoderMaxSat,
    sinter_decoders,
)
from mqt.qecc.cc_decoder.stim_interface.max_sat_stim_decoder import MaxSatStim

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def detector_error_model() -> stim.DetectorErrorModel:
    """Return d=3 color code dem."""
    return stim.DetectorErrorModel("""
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11
        error(0.5) D0 D1 D2
        error(0.5) D0 D1 L0
        error(0.5) D0 D2
        error(0.5) D0 D3
        error(0.5) D0 L0
        error(0.5) D1 D2
        error(0.5) D1 D4
        error(0.5) D1 L0
        error(0.5) D2
        error(0.5) D2 D5
        error(0.5) D3 D4 D5
        error(0.5) D3 D4 L0
        error(0.5) D3 D5
        error(0.5) D3 D6
        error(0.5) D3 L0
        error(0.5) D4 D5
        error(0.5) D4 D7
        error(0.5) D4 L0
        error(0.5) D5
        error(0.5) D5 D8
        error(0.5) D6 D7 D8
        error(0.5) D6 D7 L0
        error(0.5) D6 D8
        error(0.5) D6 D9
        error(0.5) D6 L0
        error(0.5) D7 D8
        error(0.5) D7 D10
        error(0.5) D7 L0
        error(0.5) D8
        error(0.5) D8 D11""")


def test_decode_shots_bit_packed(detector_error_model: stim.DetectorErrorModel) -> None:
    """Test bit packed shot decoding."""
    bit_packed_shots = np.array([[0, 0]]).astype(np.uint8)
    max_sat_stim = MaxSatStim(detector_error_model)
    sinter_decoder = SinterCompiledDecoderMaxSat(max_sat_stim)
    result = sinter_decoder.decode_shots_bit_packed(bit_packed_shots)

    assert len(result) == 1
    assert result[0][0] == 1 or result[0][0]


def test_decode_via_files(detector_error_model: stim.DetectorErrorModel) -> None:
    """Test via file decoding."""
    decoder = SinterDecoderMaxSat()
    result = decoder.compile_decoder_for_dem(detector_error_model)
    assert result.decoder is not None
    assert result.decoder.num_detectors == detector_error_model.num_detectors
    assert result.decoder.observables.shape == (1, 30)
    assert len(result.decoder.problem.helper_vars) == detector_error_model.num_detectors


def test_sinter_decoders_rejects_unknown_keyword() -> None:
    """A keyword argument other than d and p is an error."""
    with pytest.raises(TypeError):
        sinter_decoders(timeout=60)  # ty: ignore[unknown-argument]


def test_sinter_decoder_requires_both_d_and_p() -> None:
    """The convergence log needs both the distance and the error rate."""
    with pytest.raises(ValueError, match="both d and p"):
        SinterDecoderMaxSat(d=3)


def test_convergence_log(
    detector_error_model: stim.DetectorErrorModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With d and p, decoding appends the convergence counts to convergence_rate.txt in the working directory."""
    monkeypatch.chdir(tmp_path)
    # sinter passes these arguments by keyword.
    compiled = sinter_decoders(d=3, p=0.1)["maxsat"].compile_decoder_for_dem(dem=detector_error_model)
    compiled.decode_shots_bit_packed(bit_packed_detection_event_data=np.array([[0, 0]], dtype=np.uint8))
    assert (tmp_path / "convergence_rate.txt").read_text(encoding="utf-8").split() == ["3", "0.1", "1", "0"]
