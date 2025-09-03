# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Class to synthesize a fault tolerant state preparation circuit for the rotated surface code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stim import Circuit

from mqt.qecc.circuit_synthesis.circuit_utils import compact_stim_circuit
from mqt.qecc.codes import RotatedSurfaceCode

if TYPE_CHECKING:
    import numpy.typing as npt

    from mqt.qecc.circuit_synthesis.noise import NoiseModel


class FTSurfaceCodeStatePrep:
    """Class to synthesize a fault tolerant state preparation circuit for the rotated surface code."""

    def __init__(
        self, distance: int | tuple[int, int]) -> None:
        """Initialize the FT_SurfaceCodeStatePrep class.

        Args:
            distance (int): The distance of the surface code.
            zero_state (bool): If True, prepare the |0> state, otherwise prepare the |+> state.
            kwargs (dict, optional): Additional keyword arguments for customization.
                Defaults to None.
        """
        if isinstance(distance, tuple):
            self.distance_x = distance[0]
            self.distance_z = distance[1]
        else:
            self.distance_x = distance
            self.distance_z = distance
        self.n = self.distance_x * self.distance_z
        self.code = RotatedSurfaceCode(x_distance=self.distance_x, z_distance=self.distance_z)

    def _get_qubit_pos(self, index_offset: int) -> Circuit:
        """Get the qubit positions for the circuit.

        Args:
            index_offset (int): The offset to apply to the qubit indices.

        Returns:
            Circuit: The circuit with qubit positions.
        """
        qubit_pos = ""
        col_offset = 0 if index_offset == 0 else index_offset // self.distance_z + 1
        for col in range(self.distance_x):
            for row in range(self.distance_z):
                col_adjusted = col + col_offset
                qubit_pos += f"QUBIT_COORDS({col_adjusted},{row}) {row * self.distance_x + col + index_offset}\n"
        return Circuit(qubit_pos)

    def _generate_patch_circuit(self, index_offset: int = 0) -> Circuit:
        """Generate the state preparation circuit."""
        circ = Circuit()

        for up, down in zip(list(range(self.distance_z // 2 , self.distance_z - 1)), list(range(self.distance_z // 2 - 1, -1, -1))):
            down_row = _generate_row(
                start_qubit_idx=down * self.distance_x,
                small_stabilizer_left=(down % 2 == 1),
                direction_down=True,
                vertical_cx_direction="left",
                horizontal_cx_direction="left",
                width=self.distance_x,
            )
            up_row = _generate_row(
                start_qubit_idx=up * self.distance_x,
                small_stabilizer_left=(up % 2 == 1),
                direction_down=False,
                vertical_cx_direction="right",
                horizontal_cx_direction="right",
                width=self.distance_x,
            )
            both = down_row + up_row
            both.offset(index_offset)
            circ += Circuit(both.to_string())
        return circ

    def get_circuit_logical_x(self, noise_model: NoiseModel) -> Circuit:
        """Get the circuit with detectors added.

        Returns:
            Circuit: The circuit with detectors.
        """
        circ : Circuit = compact_stim_circuit(self._generate_patch_circuit())

        noisy_circ = noise_model.apply(circ)
        noisy_circ += Circuit("M " + " ".join(map(str, range(self.n))) + "\n") # TODO check why apply removes measurements

        det = self._detectors(self.code.Hz, self.code.Lz)
        pos = self._get_qubit_pos(0)
        self.circ_Lx = pos + noisy_circ + det
        return self.circ_Lx

    def get_circuit_logical_z(self, noise_model: NoiseModel) -> Circuit:
        """Get the circuit with detectors added for logical Z.

        Returns:
            Circuit: The circuit with detectors for logical Z.
        """
        # workaround to measure logical Z for 0 state in middle row
        middle_row = self.distance_z // 2
        middle_row_qubits = list(range(middle_row * self.distance_x, (middle_row + 1) * self.distance_x))
        patch1 = Circuit("R " + " ".join(map(str, middle_row_qubits)) + "\n")
        patch1 += Circuit("H " + " ".join(map(str, middle_row_qubits)) + "\n")

        patch1 += self._generate_patch_circuit()

        patch1_noisy = self._get_qubit_pos(0) + noise_model.apply(patch1)
        measure = Circuit("MX " + " ".join(map(str, range(0, self.n))) + "\n")

        det = self._detectors(self.code.Hx, self.code.Lx)
        self.circ_Lz = patch1_noisy + measure + det
        return self.circ_Lz

    def _detectors(self, detectors: npt.NDArray[np.int8], logicals: npt.NDArray[np.int8], offset: int = 0) -> Circuit:
        """Prepare the circuit to measure the logical error rate of the logical X operator."""
        # add detectors
        det_circ = ""
        for m, matrix_type in [(detectors, "detectors"), (logicals, "observables")]:
            for stab in m:
                indices = [i + offset for i, val in enumerate(stab) if val == 1]
                x_coord = indices[0] % self.distance_x
                y_coord = indices[0] // self.distance_x
                # convert indices into stim rec's
                recs = [i - self.n - 2 * offset for i in indices]
                if matrix_type == "detectors":
                    det_circ += f"DETECTOR({x_coord}, {y_coord}, 0) " + " ".join(f"rec[{i}]" for i in recs) + "\n"
                else:
                    det_circ += "OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{i}]" for i in recs) + "\n"
        return Circuit(det_circ)


class SurfaceCodeRow:
    """Class containing the h and cx gates of a single row of the surface code state preparation circuit."""

    def __init__(self, h_qubits: list[int], cx_qubits: list[int], reset_qubits: list[int]) -> None:
        """Initialize the SurfaceCodeRow class."""
        self.h_qubits = h_qubits
        self.cx_qubits = cx_qubits
        self.reset_qubits = reset_qubits

    # override + operator
    def __add__(self, other: SurfaceCodeRow) -> SurfaceCodeRow:
        """Combine two SurfaceCodeRow objects."""
        return SurfaceCodeRow(
            h_qubits=self.h_qubits + other.h_qubits,
            cx_qubits=self.cx_qubits + other.cx_qubits,
            reset_qubits=self.reset_qubits + other.reset_qubits,
        )

    def offset(self, offset: int) -> None:
        """Offset the qubit indices in the row by a given value."""
        self.h_qubits = [i + offset for i in self.h_qubits]
        self.cx_qubits = [i + offset for i in self.cx_qubits]
        self.reset_qubits = [i + offset for i in self.reset_qubits]

    def to_string(self, skip_hs: bool = False) -> str:
        """Return a string representation of the row."""
        circ_str = "R " + " ".join(map(str, self.reset_qubits)) + "\n"
        if not skip_hs:
            circ_str += f"{'H ' + ' '.join(map(str, self.h_qubits))}\n"
        return circ_str + f"{'CX ' + ' '.join(map(str, self.cx_qubits))}\n"


def _generate_row(
    start_qubit_idx: int,
    small_stabilizer_left: bool,
    direction_down: bool,
    horizontal_cx_direction: str,
    vertical_cx_direction: str,
    width: int,
) -> SurfaceCodeRow:
    """Generate one row of the circuit.

    Args:
        start_qubit_idx (int): The starting index of the qubits in this row.
        small_stabilizer_left (bool): If True, the small (weight 2) stabilizer is on the left.
        direction_down (bool): If True, the stabilizer is build downwards, otherwise upwards.
        horizontal_cx_direction (str): The direction of the horizontal CX gates, either 'left' or 'right'.
        vertical_cx_direction (str): One of 'left', 'right', 'straight'.
        width (int): The width of the code patch row.
    """
    start_reset = start_qubit_idx if direction_down else start_qubit_idx + width
    end_reset = start_qubit_idx + width if direction_down else start_qubit_idx + 2 * width
    reset_qubits = list(range(start_reset, end_reset))

    # Invert error propagation direction if stabilizer is not build downwards
    if not direction_down:
        if vertical_cx_direction == "left":
            vertical_cx_direction = "right"
        elif vertical_cx_direction == "right":
            vertical_cx_direction = "left"
        if horizontal_cx_direction == "left":
            horizontal_cx_direction = "right"
        elif horizontal_cx_direction == "right":
            horizontal_cx_direction = "left"

    small_stabilizer_index = start_qubit_idx + width - 1 if not small_stabilizer_left else start_qubit_idx

    # place Hadamar gates
    horizontal_directions_align = (small_stabilizer_left and horizontal_cx_direction == "left") or (
        not small_stabilizer_left and horizontal_cx_direction == "right"
    )
    qubits_h: list[int] = []
    if horizontal_directions_align:
        qubits_h += list(range(start_qubit_idx, start_qubit_idx + width, 2))
    else:
        qubits_h += list(range(start_qubit_idx + 1, start_qubit_idx + width, 2))
        qubits_h.append(small_stabilizer_index)

    if not direction_down:
        qubits_h = [i + width for i in qubits_h]
        small_stabilizer_index += width

    qubits_h.sort()

    qubits_cx_h: list[int] = []
    qubits_cx_v_first: list[int] = []
    qubits_cx_v_second: list[int] = []
    vertical_step = width if direction_down else -width
    # horizontal CX
    for i in qubits_h:
        if i == small_stabilizer_index:
            continue
        qubits_cx_h += [i, i + 1] if horizontal_cx_direction == "right" else [i, i - 1]
    # directional CX
    for i in qubits_h:
        # skip small stabs one
        if i == small_stabilizer_index:
            continue
        if vertical_cx_direction == "left":
            if horizontal_cx_direction == "left":
                qubits_cx_v_first += [i, i + vertical_step - 1]
                qubits_cx_v_second += [i, i + vertical_step]
            else:
                qubits_cx_v_first += [i + 1, i + vertical_step]
                qubits_cx_v_second += [i + 1, i + vertical_step + 1]
        elif vertical_cx_direction == "right":
            if horizontal_cx_direction == "right":
                qubits_cx_v_first += [i, i + vertical_step + 1]
                qubits_cx_v_second += [i, i + vertical_step]
            else:
                qubits_cx_v_first += [i - 1, i + vertical_step]
                qubits_cx_v_second += [i - 1, i + vertical_step - 1]
        elif vertical_cx_direction == "straight":
            qubits_cx_v_first += [i, i + vertical_step]
            if horizontal_cx_direction == "left":
                qubits_cx_v_second += [i - 1, i + vertical_step - 1]
            else:
                qubits_cx_v_second += [i + 1, i + vertical_step + 1]
    # small stabilizer cx
    qubits_cx_v_second += [small_stabilizer_index, small_stabilizer_index + vertical_step]

    return SurfaceCodeRow(h_qubits=qubits_h, cx_qubits=qubits_cx_h + qubits_cx_v_first + qubits_cx_v_second, reset_qubits=reset_qubits)
