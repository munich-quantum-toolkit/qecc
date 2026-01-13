# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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
from ldpc.mod2.mod2_numpy import rank

from mqt.qecc.circuit_synthesis import CatStatePreparationExperiment, cat_state_balanced_tree, cat_state_line
from mqt.qecc.circuit_synthesis.cat_states import (
    cat_state_pruned_balanced_circuit,
    check_ft_partial_cnot,
    fault_gens_from_circuit,
    recursive_fuse_cat_state,
    search_ft_cnot_cegar,
    search_ft_cnot_local_search,
    search_ft_cnot_smt,
    simulate_recursive_cat_construction,
)

if TYPE_CHECKING:
    import stim


def _is_cat_state(circ: stim.Circuit) -> bool:
    """Check if a circuit prepares a cat state."""
    w = circ.num_qubits
    _, _, z2x, z2z, _x_signs, z_signs = circ.to_tableau().to_numpy()
    circ_tab = np.hstack((z2x, z2z, z_signs.reshape((w, 1)))).astype(np.int8)

    cat_state_tab = np.zeros((w, 2 * w + 1), dtype=np.int8)
    cat_state_tab[0, :w] = 1
    for i in range(1, w):
        cat_state_tab[i, w + i - 1] = 1
        cat_state_tab[i, w + i] = 1

    return bool(rank(np.vstack((circ_tab, cat_state_tab))) == w)


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


def test_cat_state_experiment_nonft() -> None:
    """Test non-ft cat state preparation."""
    c1 = cat_state_line(6)
    c2 = cat_state_line(6)
    perm = [0, 1, 2, 3, 4, 5]  # Identity permutation
    sim = CatStatePreparationExperiment(c1, c2, perm)

    ps = [0.06, 0.05]
    _, _, errs, _ = sim.cat_prep_experiment(ps, shots_per_p=2000000)

    errs_w2 = errs[:, 2]
    errs_w3 = errs[:, 3]

    for i, p in enumerate(ps):
        assert errs_w2[i] > (2 / 3 * p) ** 2
        assert errs_w2[i] < 10 * (2 / 3 * p) ** 2

        assert errs_w3[i] > (2 / 3 * p) ** 3
        assert errs_w3[i] > 10 * (2 / 3 * p) ** 3


def test_cat_state_experiment_ft() -> None:
    """Test ft cat state preparation."""
    c1 = cat_state_line(6)
    c2 = cat_state_line(6)
    perm = [0, 4, 2, 3, 1, 5]  # FT permutation
    sim = CatStatePreparationExperiment(c1, c2, perm)

    ps = [0.06, 0.05]
    _, _, errs, _ = sim.cat_prep_experiment(ps, shots_per_p=3000000)

    errs_w2 = errs[:, 2]
    errs_w3 = errs[:, 3]

    for i, p in enumerate(ps):
        assert errs_w2[i] > (2 / 3 * p) ** 2
        assert errs_w2[i] < 10 * (2 / 3 * p) ** 2

        assert errs_w3[i] > (2 / 3 * p) ** 3
        assert errs_w3[i] < 10 * (2 / 3 * p) ** 3


def _cat_fault_gens(w: int) -> stim.Circuit:
    return fault_gens_from_circuit(cat_state_pruned_balanced_circuit(w))


def test_check_ft() -> None:
    """Test correctness of ft partial CNOT checking."""
    w1 = 6
    w2 = 4
    gens1 = _cat_fault_gens(w1)
    gens2 = _cat_fault_gens(w2)
    t = w1 // 2
    ctrls_non_ft = [0, 1, 2, 3]
    perm = [0, 1, 2, 3]

    is_ft, _ = check_ft_partial_cnot(gens1, w1, gens2, w2, ctrls_non_ft, perm, t)
    assert not is_ft

    ctrls_ft = [0, 1, 2, 4]
    is_ft, _ = check_ft_partial_cnot(gens1, w1, gens2, w2, ctrls_ft, perm, t)
    assert is_ft


@pytest.mark.parametrize(("w1", "w2"), [(2, 2), (3, 2), (4, 2), (5, 2), (6, 3), (7, 4), (8, 6), (9, 6)])
def test_cegar_synthesis_sat(w1: int, w2: int) -> None:
    """Test correctness of ft partial CNOTs constructed by CEGAR search."""
    t = w1 // 2
    c1 = cat_state_pruned_balanced_circuit(w1)
    c2 = cat_state_pruned_balanced_circuit(w2)
    ctrls, perm, _info = search_ft_cnot_cegar(c1, c2, t)

    assert ctrls is not None
    assert perm is not None
    assert len(ctrls) == len(perm)
    assert len(perm) == w2

    gens1 = fault_gens_from_circuit(c1)
    gens2 = fault_gens_from_circuit(c2)
    # validate CNOT by counting faults
    is_ft, _ = check_ft_partial_cnot(gens1, w1, gens2, w2, ctrls, perm, t)

    assert is_ft


@pytest.mark.parametrize(("w1", "w2"), [(2, 2), (3, 2), (4, 2), (5, 2), (6, 3), (7, 4), (8, 6), (9, 6)])
def test_local_search_synthesis_sat(w1: int, w2: int) -> None:
    """Test correctness of ft partial CNOTs constructed by local search."""
    c1 = cat_state_pruned_balanced_circuit(w1)
    c2 = cat_state_pruned_balanced_circuit(w2)
    t = w1 // 2
    seed = 1234
    ctrls, perm, _info = search_ft_cnot_local_search(c1, c2, t, ctrls=10, ctrl_moves=5, perm_iters=10, seed=seed)

    assert ctrls is not None
    assert perm is not None
    assert len(ctrls) == len(perm)
    assert len(perm) == w2

    gens1 = _cat_fault_gens(w1)
    gens2 = _cat_fault_gens(w2)
    # validate CNOT by counting faults
    is_ft, _ = check_ft_partial_cnot(gens1, w1, gens2, w2, ctrls, perm, t)

    assert is_ft


# @pytest.mark.parametrize(("w1", "w2"), [(4, 2)])
@pytest.mark.parametrize(("w1", "w2"), [(2, 2), (3, 2), (4, 2), (5, 2), (6, 3), (7, 4), (8, 6), (9, 6)])
def test_smt_synthesis_sat(w1: int, w2: int) -> None:
    """Test correctness of ft partial CNOTs constructed by direct SMT encoding."""
    c1 = cat_state_pruned_balanced_circuit(w1)
    c2 = cat_state_pruned_balanced_circuit(w2)
    t = w1 // 2
    seed = 1234
    ctrls, perm, _info = search_ft_cnot_smt(c1, c2, t, ctrls=20, seed=seed)

    assert ctrls is not None
    assert perm is not None
    assert len(ctrls) == len(perm)
    assert len(perm) == w2

    gens1 = _cat_fault_gens(w1)
    gens2 = _cat_fault_gens(w2)
    # validate CNOT by counting faults
    is_ft, _ = check_ft_partial_cnot(gens1, w1, gens2, w2, ctrls, perm, t)

    assert is_ft


@pytest.mark.parametrize(("w1", "w2"), [(6, 2), (7, 3), (8, 5), (9, 5)])
def test_cegar_synthesis_unsat(w1: int, w2: int) -> None:
    """Test correctness of ft partial CNOTs constructed by CEGAR search."""
    c1 = cat_state_pruned_balanced_circuit(w1)
    c2 = cat_state_pruned_balanced_circuit(w2)
    t = w1 // 2
    ctrls, perm, _info = search_ft_cnot_cegar(c1, c2, t)
    assert ctrls is None
    assert perm is None


@pytest.mark.parametrize(("w1", "w2"), [(6, 2), (7, 3), (8, 5), (9, 5)])
def test_local_search_synthesis_unsat(w1: int, w2: int) -> None:
    """Test correctness of ft partial CNOTs constructed by LOCAL_SEARCH search."""
    c1 = cat_state_pruned_balanced_circuit(w1)
    c2 = cat_state_pruned_balanced_circuit(w2)
    t = w1 // 2
    seed = 1234
    ctrls, perm, _info = search_ft_cnot_local_search(c1, c2, t, ctrls=10, ctrl_moves=5, perm_iters=10, seed=seed)
    assert ctrls is None
    assert perm is None


@pytest.mark.parametrize(("w1", "w2"), [(6, 2), (7, 3), (8, 5), (9, 5)])
def test_smt_synthesis_unsat(w1: int, w2: int) -> None:
    """Test correctness of ft partial CNOTs constructed by SMT search."""
    c1 = cat_state_pruned_balanced_circuit(w1)
    c2 = cat_state_pruned_balanced_circuit(w2)
    t = w1 // 2
    seed = 1234
    ctrls, perm, _info = search_ft_cnot_smt(c1, c2, t, ctrls=10, seed=seed)
    assert ctrls is None
    assert perm is None


def test_recursive_fuse_cat_state():
    """Test the recursive construction of fault-tolerant cat states."""
    w = 8
    t = 2
    circ, measurements = recursive_fuse_cat_state(w, t)

    # Check the circuit structure
    assert circ.num_qubits >= w, "The circuit should have at least `w` qubits."
    assert len(measurements) > 0, "There should be at least one measurement step."

    # Check the measurements
    for ancillas, data_qubits in measurements:
        assert len(ancillas) <= w, "Ancilla qubits exceed the total qubits."
        assert len(data_qubits) <= w, "Data qubits exceed the total qubits."


@pytest.mark.parametrize("w", [5, 6, 7, 8, 9])
def test_simulate_recursive_cat_construction(w: int) -> None:
    """Test the simulation of recursive cat state construction."""
    t = 2
    p = 0.01
    n_samples = 10000

    acceptance_rate, acceptance_rate_error, error_rates, error_rates_error = simulate_recursive_cat_construction(
        w, t, p, n_samples
    )

    # Check the results
    assert 0 < acceptance_rate < 1, "Acceptance rate should be between 0 and 1."
    assert acceptance_rate_error >= 0, "Acceptance rate error should be non-negative."
    assert len(error_rates) == w // 2 + 1, "Error rates length mismatch."
    assert len(error_rates_error) == w // 2 + 1, "Error rates error length mismatch."
