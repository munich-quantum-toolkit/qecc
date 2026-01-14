# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods and utilities for code switching compilation."""

from __future__ import annotations

from .code_switching_compiler import CompilerConfig, MinimalCodeSwitchingCompiler
from .compilation_utils import insert_switch_placeholders, naive_switching

__all__ = [
    "CompilerConfig",
    "MinimalCodeSwitchingCompiler",
    "insert_switch_placeholders",
    "naive_switching",
]
