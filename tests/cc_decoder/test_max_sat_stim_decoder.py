# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""test the max sat stim decoder integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim

if TYPE_CHECKING:
    from numpy.typing import NDArray
from mqt.qecc.cc_decoder.stim_interface.max_sat_stim_decoder import MaxSatStim


@pytest.fixture
def hamming_code() -> NDArray[bool]:
    """Return the hamming code check matrix."""
    return np.array([
        [True, True, False, True, True, False, False],
        [False, True, True, False, True, True, False],
        [False, False, False, True, True, True, True],
    ])


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


def test_check_matrix_to_adj_lists(hamming_code: NDArray[bool]) -> None:
    """Test the matrix to adjacency lists function."""
    expected_qft = {0: [0], 1: [0, 1], 3: [0, 2], 4: [0, 1, 2], 2: [1], 5: [1, 2], 6: [2]}
    expected_ftq = {0: [0, 1, 3, 4], 1: [1, 2, 4, 5], 2: [3, 4, 5, 6]}
    qft, ftq = MaxSatStim.check_matrix_to_adj_lists(hamming_code)
    assert expected_qft == qft
    assert expected_ftq == ftq


def test_decode_batch(detector_error_model: stim.DetectorErrorModel) -> None:
    """Test the batch decoding function integration."""
    shots = np.array([[0, 0]]).astype(np.uint8)
    maxsatstim = MaxSatStim(detector_error_model)
    res_pred, res_conv, res_not_cong_cnt = maxsatstim.decode_batch(
        shots=shots, bit_packed_shots=True, bit_packed_predictions=True
    )
    assert len(res_pred) == 1
    assert res_conv == 1
    assert res_not_cong_cnt == 0
