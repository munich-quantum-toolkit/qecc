# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test cat state preparation and simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ldpc import mod2

from mqt.qecc.circuit_synthesis import CatStatePreparationExperiment, cat_state_balanced_tree, cat_state_line

if TYPE_CHECKING:
    import numpy.typing as npt
    from stim import Circuit


def _is_cat_state(circ: Circuit) -> bool:
    """Check if a circuit prepares a cat state."""
    w = circ.num_qubits
    _, _, z2x, z2z, _x_signs, z_signs = circ.to_tableau().to_numpy()
    circ_tab = np.hstack((z2x, z2z, z_signs.reshape((w, 1)))).astype(np.int8)

    cat_state_tab = np.zeros((w, 2 * w + 1), dtype=np.int8)
    cat_state_tab[0, :w] = 1
    for i in range(1, w):
        cat_state_tab[i, w + i - 1] = 1
        cat_state_tab[i, w + i] = 1

    return bool(mod2.rank(np.vstack((circ_tab, cat_state_tab))) == w)


@pytest.mark.parametrize("w", [1, 2, 4, 8, 16])
def test_balanced_tree(w: int) -> None:
    """Test cat state preparation using balanced tree structure."""
    circ = cat_state_balanced_tree(w)
    assert _is_cat_state(circ)


@pytest.mark.parametrize("w", [3, 5, 6, 7])
def test_balanced_tree_no_power_two(w: int) -> None:
    """Test cat state preparation using balanced tree structure."""
    # Check that ValueError is raised for non-power of two
    with pytest.raises(ValueError, match=r"w must be a power of two."):
        cat_state_balanced_tree(w)


@pytest.mark.parametrize("w", [1, 2, 4, 8, 16])
def test_line(w: int) -> None:
    """Test cat state preparation using a linear-depth circuit."""
    # Test the tree
    circ = cat_state_line(w)
    assert _is_cat_state(circ)


def _fit_improvement(x: npt.NDArray[float], y: npt.NDArray[float]) -> float:
    fit_quad = np.polyfit(x, y, 2)
    fit_cube = np.polyfit(x, y, 3)

    # compute RSS (residual sum of squares)
    rss2: float = np.sum((np.polyval(fit_quad, x) - y) ** 2)
    rss3: float = np.sum((np.polyval(fit_cube, x) - y) ** 2)

    # fraction of RSS reduced by the cubic
    return (rss2 - rss3) / rss2


def _pure_coeff_and_rss(x: npt.NDArray[float], y: npt.NDArray[float], degree: int) -> tuple[float, float]:
    # closed-form least squares for y ≃ c * x**degree
    x_d = x**degree
    c = np.dot(x_d, y) / np.dot(x_d, x_d)
    rss: float = np.sum((c * x_d - y) ** 2)
    return c, rss


def test_cat_state_experiment_nonft() -> None:
    """Test non-ft cat state preparation."""
    c1 = cat_state_line(6)
    c2 = cat_state_line(6)
    perm = [0, 1, 2, 3, 4, 5]  # identity permutation
    sim = CatStatePreparationExperiment(c1, c2, perm)

    ps = np.arange(0.01, 0.1, 0.01)
    _, _, errs, _ = sim.cat_prep_experiment(ps, shots_per_p=1000000)

    # In the given circuit structure, single errors can lead to weight 3 errors.
    # Since c1 and c2 prepare the circuit in the same way, such errors cancel out and won't be detected.
    # Therefore we expect the probability of a weight 3 error to scale as O(p^2).
    errs_w3 = errs[:, 3]
    n = len(ps)

    # get coefficients and RSS
    _, rss2 = _pure_coeff_and_rss(ps, errs_w3, 2)
    _, rss3 = _pure_coeff_and_rss(ps, errs_w3, 3)

    # AIC = n*ln(RSS/n) + 2*k, here k=1
    aic2 = n * np.log(rss2 / n) + 2 * 1
    aic3 = n * np.log(rss3 / n) + 2 * 1

    # Quadratic should fit better than cubic
    assert aic2 < aic3


def test_cat_state_experiment_ft() -> None:
    """Test ft cat state preparation."""
    c1 = cat_state_line(6)
    c2 = cat_state_line(6)
    perm = [0, 4, 2, 3, 1, 5]  # ft permutation
    sim = CatStatePreparationExperiment(c1, c2, perm)

    ps = np.arange(0.01, 0.1, 0.01)
    _, _, errs, _ = sim.cat_prep_experiment(ps, shots_per_p=1000000)

    # In the given circuit structure, single errors can lead to weight 3 errors.
    # Since c1 and c2 are connected via a permuted CNOT such errors won't cancel out and won't be detected.
    # Therefore we expect the probability of a weight 3 error to scale as O(p^3).
    errs_w3 = errs[:, 3]
    n = len(ps)

    # get coefficients and RSS
    _, rss2 = _pure_coeff_and_rss(ps, errs_w3, 2)
    _, rss3 = _pure_coeff_and_rss(ps, errs_w3, 3)

    # AIC = n*ln(RSS/n) + 2*k, here k=1
    aic2 = n * np.log(rss2 / n) + 2 * 1
    aic3 = n * np.log(rss3 / n) + 2 * 1

    # Cubic should fit better than quadratic
    assert aic2 > aic3
