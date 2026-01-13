# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Testing of layouts. Just runs an instance each."""

from mqt.qecc.cococo import layouts

pos = tuple[int, int]


def test_scalable_layout():
    """Just runs an instance of every layout type. If no error occurs all is fine. TODO more sophisticated tests."""
    m = 5
    n = 5
    factories: list[pos] = []
    remove_edges = False
    for layout_type in ("single", "pair", "triple", "hex"):
        try:
            layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
        except Exception as e:  # noqa: PERF203
            msg = f"Problem with {layout_type} scalable layouts."
            raise ValueError(msg) from e
