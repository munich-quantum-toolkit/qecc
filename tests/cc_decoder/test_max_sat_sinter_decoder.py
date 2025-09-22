# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""tests for the color code stim decoder main functionality."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from mqt.qecc.cc_decoder.stim_interface.max_sat_sinter_decoder import SinterCompiledDecoderMaxSat, SinterDecoderMaxSat
from mqt.qecc.cc_decoder.stim_interface.max_sat_stim_decoder import MaxSatStim


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
