# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods and utilities for code switching compilation."""

from __future__ import annotations

from .code_switching_compiler import CodeSwitchGraph, insert_switch_placeholders

__all__ = ["CodeSwitchGraph", "insert_switch_placeholders"]
