"""Test the snake builder."""

from __future__ import annotations

import csv
import pathlib
from typing import cast
from unittest.mock import patch

import networkx as nx
import numpy as np
import stim

from mqt.qecc import CSSCode
from mqt.qecc.cococo import snake_builder

pos = tuple[int, int]

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_snake_builder_stdw():
    """Tests the stabilizers for a 4-snake with d=5 with the STDW scheme."""
    m = 10
    n = 18

    # generate hexagonal networkx graph
    g = nx.hexagonal_lattice_graph(m=m, n=n, periodic=False, with_positions=True, create_using=None)

    # positions of logical qubits per triangle
    positions = [
        [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (5, 2),
            (4, 2),
            (3, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (4, 4),
            (3, 4),
            (2, 4),
            (3, 5),
            (4, 5),
            (3, 6),
            (3, 7),
        ],
        [
            (6, 2),
            (6, 3),
            (6, 4),
            (7, 4),
            (7, 5),
            (6, 5),
            (5, 5),
            (5, 6),
            (6, 6),
            (7, 6),
            (8, 7),
            (7, 7),
            (6, 7),
            (5, 7),
            (4, 8),
            (5, 8),
            (6, 8),
            (7, 8),
            (8, 8),
        ],
        [
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
            (8, 10),
            (8, 11),
            (7, 11),
            (6, 11),
            (5, 11),
            (5, 12),
            (6, 12),
            (7, 12),
            (7, 13),
            (6, 13),
            (5, 13),
            (6, 14),
            (7, 14),
            (6, 15),
            (6, 16),
        ],
        [
            (9, 11),
            (9, 12),
            (9, 13),
            (10, 13),
            (10, 14),
            (9, 14),
            (8, 14),
            (8, 15),
            (9, 15),
            (10, 15),
            (11, 16),
            (10, 16),
            (9, 16),
            (8, 16),
            (7, 17),
            (8, 17),
            (9, 17),
            (10, 17),
            (11, 17),
        ],
    ]

    d = 5
    snake = snake_builder.SnakeBuilderSTDW(g, positions, d)

    z_plaquettes, x_plaquettes = snake.find_stabilizers()

    # generate check matrix
    hz = snake.gen_check_matrix(z_plaquettes)
    hx = snake.gen_check_matrix(x_plaquettes)
    hz = hz.tolist()
    hx = hx.tolist()

    hx_desired: list[list[int]] = []
    path = PROJECT_ROOT / "cococo/hx_desired_stdw.csv"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        hx_desired.extend([int(x) for x in row] for row in reader)

    hz_desired: list[list[int]] = []
    path = PROJECT_ROOT / "cococo/hz_desired_stdw.csv"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        hz_desired.extend([int(x) for x in row] for row in reader)

    assert hx == hx_desired, "The X stabilizers of a STDW snake do not look as expected."
    assert hz == hz_desired, "The Z stabilizers of a STDW snake do not look as expected."

    with patch("matplotlib.pyplot.show"):
        snake.plot_stabilizers(x_plaquettes)
        snake.plot_stabilizers(z_plaquettes)

    # also just run those functions which were not covered so far
    res = snake.find_stabilizers_zz()
    assert snake.test_zz_stabs(res), "ZZ construction not right."


def test_snake_builder():
    """Tests the stabilizers for a 4-snake."""
    # lattice
    m = 8
    n = 8
    g = nx.hexagonal_lattice_graph(m=m, n=n, periodic=False, with_positions=True, create_using=None)

    positions = [
        {(2, 6): 0, (2, 7): 2, (1, 9): 1, (2, 9): 5, (3, 9): 3, (3, 8): 4, (2, 8): 6},
        {(3, 6): 0, (3, 7): 2, (4, 9): 1, (4, 8): 5, (5, 6): 3, (4, 6): 4, (4, 7): 6},
        {
            (5, 7): 0,
            (5, 8): 2,
            (4, 10): 1,
            (5, 10): 5,
            (6, 10): 3,
            (6, 9): 4,
            (5, 9): 6,
        },
        {(6, 7): 0, (6, 8): 2, (7, 10): 1, (7, 8): 6, (7, 9): 5, (8, 7): 3, (7, 7): 4},
    ]

    sb = snake_builder.SnakeBuilderSteane(g, positions)
    _ = sb.generate_x_stabilizers()
    _ = sb.generate_z_stabilizers()
    checks_z, checks_x = sb.translate_checks()

    error_message = "The check matrix differs from the aimed one for given 4-snake."

    aim_x_check: list[list[int]] = []
    path = PROJECT_ROOT / "cococo/hx_desired_steane.csv"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        aim_x_check.extend([int(x) for x in row] for row in reader)

    aim_z_check: list[list[int]] = []
    path = PROJECT_ROOT / "cococo/hz_desired_steane.csv"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        aim_z_check.extend([int(x) for x in row] for row in reader)

    assert checks_x == aim_x_check, error_message
    assert checks_z == aim_z_check, error_message


def check_matchable(h: np.ndarray) -> None:
    """Checks whether max 2 nonzero entries per col."""
    _num_rows, num_cols = np.shape(h)
    for i in range(num_cols):
        col = h[:, i]
        num_nonzero = np.sum(col)
        if num_nonzero > 2:
            msg = f"Column {i} has {num_nonzero} non-zero entries (expected ≤ 2)."
            raise AssertionError(msg)


def translate_intstabs_to_str(plaquettes: list[list[int]], q: int, stab_type: str) -> list[str]:
    """Translates plaquettes into list of strings to use with stim.

    Args:
        plaquettes (list[list[int]]): plaquettes.
        q (int): number of physical qubits.
        stab_type (str): stabilizer type (Z or X)

    Returns:
        list[str]: _description_
    """
    stabs_str = []
    for plaquette in plaquettes:
        temp = "_" * q
        for el in plaquette:
            temp = temp[:el] + stab_type + temp[el + 1 :]
        stabs_str.append(temp)
    return stabs_str


def encoding_circuit(
    snake: snake_builder.SnakeBuilderSC, opx: list[pos], opz: list[pos]
) -> tuple[stim.Circuit, stim.Circuit]:
    """Checks whether an encoding circuit of a state defined by tableau can be built. just as sanity check."""
    stars_int = [[snake.trans_dict[el] for el in op] for op in snake.stars]

    plaquettes_int = [[snake.trans_dict[el] for el in op] for op in snake.plaquettes]

    q = len(snake.qubit_edges)
    stabs_str_z = translate_intstabs_to_str(plaquettes_int, q, "Z")
    stabs_str_z = [stim.PauliString(el) for el in stabs_str_z]
    stabs_str_x = translate_intstabs_to_str(stars_int, q, "X")
    stabs_str_x = [stim.PauliString(el) for el in stabs_str_x]
    # initialize + state, i.e. add logical X

    # ADD Z OPERATOR, i.e. initialize |0>
    op = [snake.trans_dict[el] for el in opz]

    temp = "_" * q
    for el in op:
        temp = temp[:el] + "Z" + temp[el + 1 :]

    stabilizers = stabs_str_z + stabs_str_x + [stim.PauliString(temp)]
    tableau = stim.Tableau.from_stabilizers(stabilizers)
    circuit_0 = tableau.to_circuit("elimination")

    # ADD X OPERATOR i.e. initialize |+>
    op = [snake.trans_dict[el] for el in opx]

    temp = "_" * q
    for el in op:
        temp = temp[:el] + "X" + temp[el + 1 :]

    stabilizers = stabs_str_z + stabs_str_x + [stim.PauliString(temp)]
    tableau = stim.Tableau.from_stabilizers(stabilizers)
    circuit_p = tableau.to_circuit("elimination")

    return circuit_p, circuit_0


def logicals(
    snake: snake_builder.SnakeBuilderSC, d: int, hx: np.ndarray, hz: np.ndarray
) -> tuple[list[pos], list[pos]]:
    """Creates logical ops.

    Args:
        snake (snake_builder.SnakeBuilderSC): _description_
        d (int): _description_
        hx (np.ndarray): _description_
        hz (np.ndarray): _description_

    Returns:
        tuple[list, list]: _description_
    """
    code = CSSCode(distance=d, Hx=hx, Hz=hz)

    assert len(code.Lx) == 1, "More than one qubit encoded!"
    assert len(code.Lz) == 1, "More than one qubit encoded!"

    # translate Lz into list of edges on the graph
    trans_dict_rev = {value: key for key, value in snake.trans_dict.items()}
    opz_final = []
    for i, el in enumerate(code.Lz[0]):
        if el == 1:
            opz_final.append(trans_dict_rev[i])
    opx_final = []
    for i, el in enumerate(code.Lx[0]):
        if el == 1:
            opx_final.append(trans_dict_rev[i])
    return opx_final, opz_final


def test_snake_builder_sc():
    """Tests a surface code snake."""
    m, n = 20, 20
    g = nx.grid_2d_graph(m, n)

    # Define the position with the origin at the lower left
    {(x, y): (x, y) for x, y in g.nodes()}  # Keep y as positive

    d = 5

    positions_smooth = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
        [(5, 13), (6, 13), (7, 13), (8, 13), (9, 13), (10, 13)],
    ]
    positions_rough = [
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (4, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
        ],
        [
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
        ],
    ]

    snake = snake_builder.SnakeBuilderSC(g, positions_rough, positions_smooth, d)
    _, _ = snake.create_stabs()
    hx, hz, _trans_dict = snake.gen_checks()

    check_matchable(hx)
    check_matchable(hz)

    opx, opz = logicals(snake, d, hx, hz)
    _circuit_p, _circuit_0 = encoding_circuit(
        snake, opx, opz
    )  # only checks whether construction of encoding circuit works

    with patch("matplotlib.pyplot.show"):
        snake.plot_stabs(cast("list[tuple[pos,pos]]|None", opz), cast("list[tuple[pos,pos]]|None", opx), size=(8, 8))

    hx_desired: list[list[int]] = []
    path = PROJECT_ROOT / "cococo/hx_desired_sc.csv"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        hx_desired.extend([int(x) for x in row] for row in reader)

    hz_desired: list[list[int]] = []
    path = PROJECT_ROOT / "cococo/hz_desired_sc.csv"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        reader = csv.reader(f)
        hz_desired.extend([int(x) for x in row] for row in reader)

    assert hx.tolist() == hx_desired, "The X stabilizers of a SC snake do not look as expected."
    assert hz.tolist() == hz_desired, "The Z stabilizers of a SC snake do not look as expected."
