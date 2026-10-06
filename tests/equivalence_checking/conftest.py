# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for equivalence-checking tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from mqt.qecc.mod2 import is_in_row_space, rank

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    import numpy as np
    import numpy.typing as npt


def assert_same_row_space(
    transformed: npt.NDArray[np.integer],
    target: npt.NDArray[np.integer],
) -> None:
    """Assert that two matrices span the same row space."""
    assert rank(transformed) == rank(target)
    assert all(is_in_row_space(row, target) for row in transformed)


@pytest.fixture
def trace_stages(monkeypatch: pytest.MonkeyPatch) -> Callable[[ModuleType, Mapping[str, str]], list[str]]:
    """Return a function that records, in call order, the labels of the called pipeline stages of a module.

    The stages are given as a mapping from their labels to the names of the functions implementing them.
    """

    def _trace(module: ModuleType, stages: Mapping[str, str]) -> list[str]:
        called: list[str] = []

        def _recording(label: str, stage: Callable[..., Any]) -> Callable[..., Any]:
            def _stage(*args: Any, **kwargs: Any) -> Any:
                called.append(label)
                return stage(*args, **kwargs)

            return _stage

        for label, name in stages.items():
            monkeypatch.setattr(module, name, _recording(label, getattr(module, name)))
        return called

    return _trace
