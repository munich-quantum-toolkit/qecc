# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Type definitions."""

from typing import TypedDict

pos = tuple[int, int]
Layout = dict[str | int, pos | list[pos]]
VdpDict = dict[str | pos | tuple[pos, pos], list[pos]]


class HistoryTemp(TypedDict, total=False):  # pragma: no cover
    """Type for history dictionaries."""

    scores: list[int]
    layout_init: dict[str | int, pos | list[pos]]
    layout_final: dict[str | int, pos | list[pos]]
