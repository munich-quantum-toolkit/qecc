# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods and utilities for code switching compilation."""

from __future__ import annotations

from .code_switching_compiler import CompilerConfig, MinimalCodeSwitchingCompiler
from .compilation_utils import count_code_switches, insert_switch_placeholders

__all__ = [
    "CompilerConfig",
    "MinimalCodeSwitchingCompiler",
    "count_code_switches",
    "insert_switch_placeholders",
]
