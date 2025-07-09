# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions needed across python tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def check_and_load_json(file_name: str, results_dir: str) -> dict[str, Any]:
    """Check that the results directory contains exactly one file with the given name and load it as JSON."""
    results_path = Path(results_dir)
    assert results_path.exists()
    assert results_path.is_dir()
    assert len(list(results_path.iterdir())) == 1
    result_file = results_path / file_name
    assert result_file.exists()
    assert result_file.is_file()
    with result_file.open("r") as f:
        result: dict[str, Any] = json.load(f)

    for file in results_path.iterdir():
        file.unlink()
    results_path.rmdir()

    return result
