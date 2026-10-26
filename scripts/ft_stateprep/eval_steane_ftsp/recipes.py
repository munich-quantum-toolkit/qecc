# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthesis recipes for the state-preparation circuits of Weilandt, Peham, Wille, Phys. Rev. A (2026).

Each ``reproduce_<code>()`` rebuilds the four circuits C1-C4 (``<code>_heuristic_{0..3}.qasm`` in ``circuits/``)
following the synthesis notebooks that produced them. The notebooks and ``check_reproduction.py`` both call these
functions, so they cannot drift apart.

Sources:
    cc_6_6_6_d5, cc_4_8_8_d5:  backtracking_circ_construction.ipynb, find_4 on the code's heuristic circuit
    rotated_surface_d5:        25_1_5_rot_surface_code.ipynb (fault sets are empty, so C1 = C2 = C3 = C4)
    eve_20_2_6:                20_2_6_code.ipynb, distance set to 5 (a distance of 6 would select the d = 7 branch
                               of ``get_fs_based_on_d``); C4 with penalty CNOTs from the first two layers of reversed C1
    cc_6_6_6_d7, cc_4_8_8_d7:  backtracking_circ_construction.ipynb, hand-picked penalty CNOTs; C2-C4 use the frozen
                               April 2025 elimination in legacy_synthesis.py, which seeds C3 and C4 with the row
                               echelon form of Hx (the current library finds different circuits for the same inputs)
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from legacy_synthesis import heuristic_gaussian_elimination

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    FaultyStatePrepCircuit,
    heuristic_prep_circuit,
    heuristic_reference_prep_circuit,
)
from mqt.qecc.circuit_synthesis.state_prep import _build_state_prep_circuit_from_back  # noqa: PLC2701
from mqt.qecc.circuit_synthesis.synthesis_utils import get_fs_based_on_d
from mqt.qecc.codes import HexagonalColorCode, RotatedSurfaceCode, SquareOctagonColorCode

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

logger = logging.getLogger(__name__)

CHECK_MATRIX_DIR = Path(__file__).resolve().parent / "check_matrix"

SynthCode = Literal["original", "c1"]


def _eve_20_2_6() -> CSSCode:
    # The code has distance 6, but the circuits were synthesized with distance 5 (one fault level, as for d = 5):
    # get_fs_based_on_d would select its distance-7 branch for a distance of 6.
    return CSSCode.from_file(CHECK_MATRIX_DIR / "eve_20_2_6.txt", distance=5)


# The code each circuit set was synthesized for, with the distance used for synthesis and fault-set selection.
CODES: dict[str, Callable[[], CSSCode]] = {
    "cc_4_8_8_d5": lambda: SquareOctagonColorCode(5),
    "cc_6_6_6_d5": lambda: HexagonalColorCode(5),
    "rotated_surface_d5": lambda: RotatedSurfaceCode(distance=5),
    "eve_20_2_6": _eve_20_2_6,
    "cc_4_8_8_d7": lambda: SquareOctagonColorCode(7),
    "cc_6_6_6_d7": lambda: HexagonalColorCode(7),
}


def _filter_fault_set(fault_set: npt.NDArray[np.int8], threshold: int) -> npt.NDArray[np.int8]:
    """Keep the faults of weight above ``threshold``."""
    return fault_set[np.where(fault_set.sum(axis=1) > threshold)]


def _first_occurrences(cnots: list[tuple[int, int]], check_control: bool = True) -> list[tuple[int, int]]:
    seen = set()
    result = []
    for pair in cnots:
        key = pair[0] if check_control else pair[1]
        if key not in seen:
            result.append(pair)
            seen.add(key)
    return result


def _penalty_layers(cnots: list[tuple[int, int]], layers: int, check_control: bool = True) -> list[tuple[int, int]]:
    """Return the CNOTs of the first ``layers`` first-occurrence layers of ``cnots`` (``get_penalty_cnots``)."""
    penalty_cnots: list[tuple[int, int]] = []
    remaining = cnots.copy()
    for _ in range(layers):
        if not remaining:
            break
        current_layer = _first_occurrences(remaining, check_control=check_control)
        penalty_cnots.extend(current_layer)
        remaining = [c for c in remaining if c not in current_layer]
    return penalty_cnots


def _find_4(
    c1: FaultyStatePrepCircuit,
    code: CSSCode,
    synth_code: SynthCode = "original",
    c4_penalty_layers: int = 0,
) -> list[FaultyStatePrepCircuit]:
    """Build C2-C4 from C1 (``find_4`` of the notebooks).

    Args:
        c1: The reference circuit C1.
        code: The code C1 was synthesized for.
        synth_code: Code passed to the synthesis of C2-C4: ``"original"`` is ``code``, ``"c1"`` is the code
            reconstructed from C1 (``c1.code``, as in backtracking_circ_construction.ipynb on the current branch).
        c4_penalty_layers: If > 0, C4 is built with the first that many layers of the reversed C1 CNOTs as penalty
            CNOTs (20_2_6_code.ipynb).
    """
    target = code if synth_code == "original" else c1.code

    xfs1, zfs1 = get_fs_based_on_d(c1)
    if xfs1.size:
        spc2 = heuristic_reference_prep_circuit(target, ref_x_fs=xfs1)
        _, zfs2 = get_fs_based_on_d(spc2)
    else:
        spc2 = c1
        zfs2 = zfs1

    if zfs1.size and zfs2.size:
        c1c2zfs = np.unique(np.vstack((zfs1, zfs2), dtype=np.int8), axis=0)
        spc3 = heuristic_reference_prep_circuit(target, ref_z_fs=c1c2zfs, guide_by_x=False)
    elif zfs1.size or zfs2.size:
        # The notebooks have no C4 for this case (c1c2zfs would be undefined).
        msg = "Only one of the C1/C2 Z fault sets is non-empty; no recipe for C4."
        raise RuntimeError(msg)
    else:
        return [c1, spc2, c1, spc2]
    xfs3, _ = get_fs_based_on_d(spc3)

    penalty_cnots = None
    if c4_penalty_layers:
        penalty_cnots = _penalty_layers(list(reversed(c1.circ.cnots)), c4_penalty_layers)
    spc4 = heuristic_reference_prep_circuit(
        target, ref_x_fs=xfs3, ref_z_fs=c1c2zfs, guide_by_x=False, penalty_cnots=penalty_cnots
    )
    return [c1, spc2, spc3, spc4]


def reproduce_cc_6_6_6_d5(synth_code: SynthCode = "original") -> list[FaultyStatePrepCircuit]:
    """[[19,1,5]] hexagonal color code."""
    code = CODES["cc_6_6_6_d5"]()
    return _find_4(heuristic_prep_circuit(code), code, synth_code)


def reproduce_cc_4_8_8_d5(synth_code: SynthCode = "original") -> list[FaultyStatePrepCircuit]:
    """[[17,1,5]] square-octagon color code."""
    code = CODES["cc_4_8_8_d5"]()
    return _find_4(heuristic_prep_circuit(code), code, synth_code)


def reproduce_rotated_surface_d5(synth_code: SynthCode = "original") -> list[FaultyStatePrepCircuit]:
    """[[25,1,5]] rotated surface code."""
    code = CODES["rotated_surface_d5"]()
    return _find_4(heuristic_prep_circuit(code), code, synth_code)


def reproduce_eve_20_2_6(synth_code: SynthCode = "original") -> list[FaultyStatePrepCircuit]:
    """[[20,2,6]] code from check_matrix/eve_20_2_6.txt, synthesized with distance 5."""
    code = CODES["eve_20_2_6"]()
    return _find_4(heuristic_prep_circuit(code), code, synth_code, c4_penalty_layers=2)


# backtracking_circ_construction.ipynb, cells 91 and 92
_CC_6_6_6_D7_C4_PENALTY = [
    (31, 24), (23, 26), (32, 33), (36, 35), (14, 20), (7, 34), (13, 27), (6, 29), (12, 19), (4, 16), (15, 25),
    (5, 28), (10, 21), (2, 30), (28, 32), (30, 16), (19, 23), (25, 35), (33, 24), (27, 34), (18, 21), (20, 31),
    (9, 28), (17, 21), (8, 34), (3, 29), (22, 35), (26, 36), (24, 17), (1, 27), (0, 29), (35, 15), (11, 27),
    (34, 2),
    (14, 31), (12, 29), (10, 20), (2, 26), (5, 36), (27, 14), (33, 26), (30, 23), (15, 32), (6, 16), (31, 32),
    (8, 16), (22, 25), (9, 23), (28, 12), (3, 25), (13, 26), (18, 31), (19, 35), (7, 23), (4, 14), (25, 9),
    (20, 24), (34, 36), (1, 26), (11, 30), (32, 16),
]  # fmt: skip


# backtracking_circ_construction.ipynb, cell 107
_CC_4_8_8_D7_C2_PENALTY = [
    (0, 7), (1, 24), (2, 25), (3, 10), (4, 26), (5, 28), (9, 8), (12, 15), (23, 30), (27, 21), (29, 6), (11, 13),
    (14, 16), (17, 19), (20, 18), (6, 7), (8, 18), (10, 13), (15, 1), (19, 22), (21, 0), (24, 14), (25, 28),
    (26, 16), (16, 12), (22, 29),
]  # fmt: skip

# backtracking_circ_construction.ipynb, cell 121 (penalty_cols_z + penalty_cols_z2)
_CC_4_8_8_D7_C3_PENALTY = [
    (0, 7), (1, 24), (2, 25), (3, 10), (4, 26), (5, 28), (9, 8), (12, 15), (23, 30), (27, 21), (29, 6), (11, 13),
    (14, 16), (17, 19), (20, 18), (11, 3), (15, 1), (17, 2), (19, 22), (20, 9), (21, 0), (24, 14), (0, 29), (1, 4),
    (2, 5), (8, 23), (10, 27), (16, 12),
    (22, 28), (13, 10), (19, 25), (18, 8), (26, 4), (16, 14), (21, 27), (17, 2), (6, 29), (20, 9), (15, 30),
    (7, 0), (24, 1), (11, 3), (2, 5), (17, 19), (27, 7), (9, 15), (16, 6), (26, 24), (10, 21), (20, 18), (11, 13),
    (15, 12), (19, 22), (9, 16), (18, 23), (27, 26),
]  # fmt: skip

# backtracking_circ_construction.ipynb, cells 134-136
_CC_4_8_8_D7_C4_PENALTY = [
    (17, 25), (15, 18), (20, 21), (30, 23), (11, 27), (10, 13), (7, 29), (6, 16), (2, 28), (3, 26), (5, 22),
    (23, 20), (28, 17), (16, 27), (25, 22), (29, 30), (13, 15), (26, 21), (8, 18), (14, 30), (1, 24), (0, 29),
    (18, 12), (9, 20), (4, 27), (19, 25),
    (11, 24), (25, 17), (13, 23), (16, 15), (7, 28), (10, 21), (17, 5), (4, 26), (0, 15), (9, 30), (3, 24), (1, 21),
    (6, 29), (23, 27), (19, 28), (8, 30), (14, 16), (15, 30),
    (11, 26), (13, 28), (16, 23), (10, 27), (9, 21), (25, 2), (7, 16), (6, 28), (1, 27), (23, 0),
]  # fmt: skip


def _legacy_prep(code: CSSCode, **kwargs: object) -> FaultyStatePrepCircuit:
    """Synthesize a circuit with the frozen April 2025 elimination (see legacy_synthesis.py).

    For codes with distance > 5 it seeds the elimination with the row echelon form of Hx whenever a reference Z fault
    set is given (C3 and C4); this is what makes the distance-7 search tractable.
    """
    with contextlib.redirect_stdout(io.StringIO()):  # the legacy function prints every elimination step
        checks, cnots = heuristic_gaussian_elimination(code.Hx, parallel_elimination=True, code=code, **kwargs)
    circ = _build_state_prep_circuit_from_back(checks, cnots, zero_state=True)
    return FaultyStatePrepCircuit(circ, code.x_distance // 2, code.z_distance // 2, code=code)


def _reproduce_d7(
    name: str,
    c2_penalty: list[tuple[int, int]],
    c3_penalty: list[tuple[int, int]],
    c4_penalty: list[tuple[int, int]],
) -> list[FaultyStatePrepCircuit]:
    """Distance-7 recipe: C1 greedy, C2-C4 with the frozen April 2025 elimination (backtracking notebook)."""
    code = CODES[name]()
    c1 = heuristic_prep_circuit(code)
    xfs1, zfs1 = get_fs_based_on_d(c1)
    logger.info("%s: C1 done (%d CX), |xfs1| = %d, |zfs1| = %d", name, c1.circ.num_cnots(), len(xfs1), len(zfs1))

    c2 = _legacy_prep(code, penalty_cols=list(c2_penalty), ref_x_fs=xfs1)
    _, zfs2 = get_fs_based_on_d(c2)
    c1c2zfs = np.unique(np.vstack((zfs1, zfs2), dtype=np.int8), axis=0)
    logger.info("%s: C2 done (%d CX)", name, c2.circ.num_cnots())

    # The [[37,1,7]] notebook passed num_rands=0, an argument that was never committed; it is dropped here.
    c3 = _legacy_prep(code, penalty_cols=list(c3_penalty), ref_z_fs=c1c2zfs, guide_by_x=False)
    xfs3, _ = get_fs_based_on_d(c3)
    ref_x_1fs = _filter_fault_set(c3.x_fault_sets[0].faults, 2)
    logger.info("%s: C3 done (%d CX)", name, c3.circ.num_cnots())

    c4 = _legacy_prep(
        code,
        penalty_cols=list(c4_penalty),
        ref_x_fs=xfs3,
        ref_z_fs=c1c2zfs,
        guide_by_x=True,
        ref_x_1fs=ref_x_1fs,
    )
    logger.info("%s: C4 done (%d CX)", name, c4.circ.num_cnots())
    return [c1, c2, c3, c4]


def reproduce_cc_6_6_6_d7() -> list[FaultyStatePrepCircuit]:
    """[[37,1,7]] hexagonal color code (backtracking_circ_construction.ipynb, cells 63-93)."""
    return _reproduce_d7("cc_6_6_6_d7", [], [], _CC_6_6_6_D7_C4_PENALTY)


def reproduce_cc_4_8_8_d7() -> list[FaultyStatePrepCircuit]:
    """[[31,1,7]] square-octagon color code (backtracking_circ_construction.ipynb, cells 104-137)."""
    return _reproduce_d7(
        "cc_4_8_8_d7",
        _CC_4_8_8_D7_C2_PENALTY,
        _CC_4_8_8_D7_C3_PENALTY,
        _CC_4_8_8_D7_C4_PENALTY,
    )


# Codes whose C2-C4 come from ``_find_4`` and therefore accept ``synth_code``.
FIND_4_CODES = ("cc_4_8_8_d5", "cc_6_6_6_d5", "rotated_surface_d5", "eve_20_2_6")

RECIPES: dict[str, Callable[..., list[FaultyStatePrepCircuit]]] = {
    "cc_4_8_8_d5": reproduce_cc_4_8_8_d5,
    "cc_6_6_6_d5": reproduce_cc_6_6_6_d5,
    "rotated_surface_d5": reproduce_rotated_surface_d5,
    "eve_20_2_6": reproduce_eve_20_2_6,
    "cc_4_8_8_d7": reproduce_cc_4_8_8_d7,
    "cc_6_6_6_d7": reproduce_cc_6_6_6_d7,
}
