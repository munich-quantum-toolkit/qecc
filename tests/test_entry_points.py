# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the console scripts of MQT QECC."""

from __future__ import annotations

from importlib.metadata import EntryPoint, distribution

import pytest

CONSOLE_SCRIPTS = [ep for ep in distribution("mqt.qecc").entry_points if ep.group == "console_scripts"]


def test_console_scripts_are_declared() -> None:
    """MQT QECC declares at least one console script."""
    assert CONSOLE_SCRIPTS


@pytest.mark.parametrize("entry_point", CONSOLE_SCRIPTS, ids=lambda ep: ep.name)
def test_console_script_loads(entry_point: EntryPoint) -> None:
    """Every declared console script points to a function that exists."""
    assert callable(entry_point.load())
