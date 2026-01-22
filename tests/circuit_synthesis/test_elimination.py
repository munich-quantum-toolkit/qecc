import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.elimination import EliminationConfig, eliminate, is_identity, get_candidate_transvections, ParallelFilter, is_terminal_transvection
from mqt.qecc.codes.pauli import StabilizerTableau, SymplecticMatrix


@pytest.fixture
def identity_tableau() -> StabilizerTableau:
    """Fixture to create an identity stabilizer tableau."""
    tableau_matrix = np.array([[1,0],[0,1]], dtype=np.int8)
    return StabilizerTableau(SymplecticMatrix(tableau_matrix))

@pytest.fixture
def cnot_tableau() -> StabilizerTableau:
    """Fixture to create a CNOT stabilizer tableau."""
    tableau_matrix = np.array([[1,0,0,0],
                               [0,1,1,0],
                               [0,0,1,0],
                               [0,0,0,1]], dtype=np.int8)
    return StabilizerTableau(SymplecticMatrix(tableau_matrix))


@pytest.fixture
def non_css_config() -> EliminationConfig:
    """Fixture to create a non-CSS elimination configuration."""
    return EliminationConfig(
        termination_criterion=is_terminal_transvection,
        sorted_candidate_ops=get_candidate_transvections,
        filters=[ParallelFilter()],
    )
    

@pytest.mark.parametrize(
    "tableau_matrix",
    [
        # "identity_tableau",
        "cnot_tableau",
    ]
)
def test_eliminate_non_css(tableau_matrix: StabilizerTableau, identity_tableau: StabilizerTableau, non_css_config: EliminationConfig, request) -> None:
    """Test the eliminate function."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    operations, result_tableau = eliminate(target_tableau, non_css_config)
    assert result_tableau == identity_tableau
    
    
