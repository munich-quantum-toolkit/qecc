"""Class to synthesize a fault tolerant state preparation circuit for the rotated surface code."""

from __future__ import annotations

from stim import Circuit

from mqt.qecc.circuit_synthesis.circuit_utils import compact_stim_circuit


class FTSurfaceCodeStatePrep:
    """Class to synthesize a fault tolerant state preparation circuit for the rotated surface code."""

    def __init__(self, distance: int | tuple[int, int], zero_state: bool = True, kwargs: dict | None = None) -> None:
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
        self.kwargs = kwargs if kwargs is not None else {}
        self._generate_circuit()

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

        circ_str = ""
        # Reset all qubits
        circ_str += "R " + " ".join(map(str, range(self.distance_x * self.distance_z))) + "\n"

        # Check kwargs for additional parameters
        if "horizontal_cx_direction" in self.kwargs:
            horizontal_cx_direction = self.kwargs["horizontal_cx_direction"]
        else:
            horizontal_cx_direction = "right"
        if "vertical_cx_direction" in self.kwargs:
            vertical_cx_direction = self.kwargs["vertical_cx_direction"]
        else:
            vertical_cx_direction = "left"

        for i in range(self.distance_z - 1):
            circ_str += _generate_row(
                start_qubit_idx=i * self.distance_x,
                small_stabilizer_left=(i % 2 == 1),
                direction_down=(i < self.distance_z // 2),
                vertical_cx_direction=vertical_cx_direction,
                horizontal_cx_direction=horizontal_cx_direction,
                width=self.distance_x,
            )

        # self.circ = Circuit(qubit_pos) + compact_stim_circuit(Circuit(circ_str))
        self.circ = Circuit(qubit_pos) + Circuit(circ_str)

    def get_circuit(self) -> Circuit:
        """Get the generated circuit.

        Returns:
            Circuit: The generated circuit.
        """
        return self.circ


def _generate_row(
    start_qubit_idx: int,
    small_stabilizer_left: bool,
    direction_down: bool,
    horizontal_cx_direction: str,
    vertical_cx_direction: str,
    width: int,
) -> str:
    """Generate one row of the circuit.

    Args:
        start_qubit_idx (int): The starting index of the qubits in this row.
        small_stabilizer_left (bool): If True, the small (weight 2) stabilizer is on the left.
        direction_down (bool): If True, the stabilizer is build downwards, otherwise upwards.
        horizontal_cx_direction (str): The direction of the horizontal CX gates, either 'left' or 'right'.
        vertical_cx_direction (str): One of 'left', 'right', 'straight'.
        width (int): The width of the code patch row.
    """
    circ_str = ""

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
    circ_str += "H " + " ".join(map(str, qubits_h)) + "\n"

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
        match vertical_cx_direction:
            case "left":
                if horizontal_cx_direction == "left":
                    qubits_cx_v_first += [i, i + vertical_step - 1]
                    qubits_cx_v_second += [i, i + vertical_step]
                else:
                    qubits_cx_v_first += [i + 1, i + vertical_step]
                    qubits_cx_v_second += [i + 1, i + vertical_step + 1]
            case "right":
                if horizontal_cx_direction == "right":
                    qubits_cx_v_first += [i, i + vertical_step + 1]
                    qubits_cx_v_second += [i, i + vertical_step]
                else:
                    qubits_cx_v_first += [i - 1, i + vertical_step]
                    qubits_cx_v_second += [i - 1, i + vertical_step - 1]
            case "straight":
                qubits_cx_v_first += [i, i + vertical_step]
                if horizontal_cx_direction == "left":
                    qubits_cx_v_second += [i - 1, i + vertical_step - 1]
                else:
                    qubits_cx_v_second += [i + 1, i + vertical_step + 1]

    circ_str += "CX " + " ".join(map(str, qubits_cx_h + qubits_cx_v_first + qubits_cx_v_second)) + "\n"

    return circ_str
