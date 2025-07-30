# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Class to synthesize a fault tolerant state preparation circuit for the rotated surface code."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stim import Circuit

from mqt.qecc.circuit_synthesis.circuit_utils import compact_stim_circuit
from mqt.qecc.codes import RotatedSurfaceCode

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from mqt.qecc.circuit_synthesis.noise import NoiseModel


class FTSurfaceCodeStatePrep:
    """Class to synthesize a fault tolerant state preparation circuit for the rotated surface code."""

    def __init__(
        self, distance: int | tuple[int, int], zero_state: bool = True, kwargs: dict[str, str] | None = None
    ) -> None:
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
        self.zero_state = zero_state
        self.n = self.distance_x * self.distance_z
        self.kwargs = kwargs if kwargs is not None else {}
        self._generate_circuit()
        self.code = RotatedSurfaceCode(x_distance=self.distance_x, z_distance=self.distance_z)

    def _generate_circuit(self) -> None:
        """Generate the state preparation circuit."""
        # This method should contain the logic to generate the circuit.
        # For now, it is a placeholder.
        qubit_pos = ""

        # Generate qubit placement on a square grid (just for crumble)
        for i in range(self.distance_x):
            for j in range(self.distance_z):
                qubit_pos += f"QUBIT_COORDS({i},{j}) {j * self.distance_x + i}\n"
        qubit_pos += "\n"

        # Reset all qubits
        qubit_pos += "R " + " ".join(map(str, range(self.n))) + "\n"
        self.qubit_pos = Circuit(qubit_pos)

        # Check kwargs for additional parameters
        if "horizontal_cx_direction" in self.kwargs:
            horizontal_cx_direction = self.kwargs["horizontal_cx_direction"]
        else:
            horizontal_cx_direction = "right"
        if "vertical_cx_direction" in self.kwargs:
            vertical_cx_direction = self.kwargs["vertical_cx_direction"]
        else:
            vertical_cx_direction = "left"

        circ_all_rows = SurfaceCodeRow([], [])

        for i in list(range(self.distance_z // 2 - 1, -1, -1)) + list(range(self.distance_z // 2, self.distance_z - 1)):
            circ_all_rows += _generate_row(
                start_qubit_idx=i * self.distance_x,
                small_stabilizer_left=(i % 2 == 1),
                direction_down=(i < self.distance_z // 2),
                vertical_cx_direction=vertical_cx_direction,
                horizontal_cx_direction=horizontal_cx_direction,
                width=self.distance_x,
            )
        measurements = ""
        measurements += "M " + " ".join(map(str, range(self.n))) + "\n"
        self.measurements = Circuit(measurements)

        self.circ = compact_stim_circuit(Circuit(circ_all_rows.to_string()))

    def get_circuit(self) -> Circuit:
        """Get the generated circuit.

        Returns:
            Circuit: The generated circuit.
        """
        return self.circ

    def get_circuit_logical_x(self, noise_model: NoiseModel) -> Circuit:
        """Get the circuit with detectors added.

        Returns:
            Circuit: The circuit with detectors.
        """
        noisy_circ = noise_model.apply(self.circ)
        det = self._detectors(self.code.Hz, self.code.Lz)
        self.circ_Lx = self.qubit_pos + noisy_circ + self.measurements + det
        return self.circ_Lx

    def _detectors(self, detectors: npt.NDArray[np.int8], logicals: npt.NDArray[np.int8]) -> Circuit:
        """Prepare the circuit to measure the logical error rate of the logical X operator."""
        # add detectors
        det_circ = ""
        for m, matrix_type in [(detectors, "detectors"), (logicals, "observables")]:
            for stab in m:
                indices = [i for i, val in enumerate(stab) if val == 1]
                x_coord = indices[0] % self.distance_x
                y_coord = indices[0] // self.distance_x
                # convert indices into stim rec's
                recs = [i - self.n for i in indices]
                if matrix_type == "detectors":
                    det_circ += f"DETECTOR({x_coord}, {y_coord}, 0) " + " ".join(f"rec[{i}]" for i in recs) + "\n"
                else:
                    det_circ += "OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{i}]" for i in recs) + "\n"
        return Circuit(det_circ)


class SurfaceCodeRow:
    """Class containing the h and cx gates of a single row of the surface code state preparation circuit."""

    def __init__(self, h_qubits: list[int], cx_qubits: list[int]) -> None:
        """Initialize the SurfaceCodeRow class."""
        self.h_qubits = h_qubits
        self.cx_qubits = cx_qubits

    # override + operator
    def __add__(self, other: SurfaceCodeRow) -> SurfaceCodeRow:
        """Combine two SurfaceCodeRow objects."""
        return SurfaceCodeRow(
            h_qubits=self.h_qubits + other.h_qubits,
            cx_qubits=self.cx_qubits + other.cx_qubits,
        )

    def to_string(self) -> str:
        """Return a string representation of the row."""
        return f"{'h ' + ' '.join(map(str, self.h_qubits))}\n" + f"{'cx ' + ' '.join(map(str, self.cx_qubits))}\n"


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

    return SurfaceCodeRow(h_qubits=qubits_h, cx_qubits=qubits_cx_h + qubits_cx_v_first + qubits_cx_v_second)
