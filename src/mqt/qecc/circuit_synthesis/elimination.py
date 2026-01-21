# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for performing Gaussian elimination on GF2 and symplectic matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..codes.pauli import StabilizerTableau


@dataclass
class EliminationConfig:
    """Configuration for elimination methods."""



class TableauOperation(ABC):
    """Represents an operation performed during tableau elimination."""

    @abstractmethod
    def apply(self, tableau: StabilizerTableau) -> None:
        """Apply the operation to the given stabilizer tableau.

        Args:
            tableau (StabilizerTableau): The stabilizer tableau to apply the operation to.
        """
        pass


class Transvection(TableauOperation):


def eliminate(target_tableau: StabilizerTableau, config: EliminationConfig) -> list[TableauOperation]:
    """Perform Gaussian elimination on the given stabilizer tableau.

    Args:
        target_tableau (StabilizerTableau): The stabilizer tableau to be eliminated.
        config (EliminationConfig): Configuration parameters for the elimination process.

    Returns:
        None: The function modifies the target_tableau in place.
    """
    # if target_tableau.is_css():
    #     _eliminate_css(target_tableau, config)

    # else:
    #     _eliminate_non_css(target_tableau, config)
