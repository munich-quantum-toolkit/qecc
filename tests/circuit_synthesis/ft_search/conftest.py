# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pytest configuration of the ft_search tests."""

from __future__ import annotations

import os
import sys

import pytest

from mqt.qecc.circuit_synthesis.ft_search._core import memo, parts


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the slow tests where they would take hours: without numba's compiler and on non-Linux CI runners.

    The Linux CI runs, with the compiler, run every test.
    """
    no_jit = os.environ.get("NUMBA_DISABLE_JIT") == "1"
    other_ci = os.environ.get("CI") is not None and sys.platform != "linux"
    if not (no_jit or other_ci):
        return
    skip = pytest.mark.skip(reason="slow search test: runs on Linux with numba's compiler")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_process_state() -> None:
    """Every test starts without memoized results or known parts."""
    memo.clear()
    parts.clear()
