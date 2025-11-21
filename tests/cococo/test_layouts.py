# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Testing of layouts. Just runs an instance each."""

from mqt.qecc.cococo import layouts

pos = tuple[int,int]


def test_scalable_layout():
    """Just runs an instance of every layout type. If no error occurs all is fine. TODO more sophisticated tests."""
    m = 5
    n = 5
    factories: list[pos] = []
    remove_edges = False
    try:
        _, _, _ = layouts.gen_layout_scalable("single", m, n, factories, remove_edges)
    except:  # noqa: E722
        msg = "Problem with single layout"
        raise ValueError(msg)
    try:
        _, _, _ = layouts.gen_layout_scalable("pair", m, n, factories, remove_edges)
    except:  # noqa: E722
        msg = "Problem with pair layout"
        raise ValueError(msg)
    try:
        _, _, _ = layouts.gen_layout_scalable("triple", m, n, factories, remove_edges)
    except:  # noqa: E722
        msg = "Problem with triple layout"
        raise ValueError(msg)
    try:
        _, _, _ = layouts.gen_layout_scalable("hex", m, n, factories, remove_edges)
    except:  # noqa: E722
        msg = "Problem with hex layout"
        raise ValueError(msg)
