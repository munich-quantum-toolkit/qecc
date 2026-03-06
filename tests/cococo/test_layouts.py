# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Testing of layouts. Just runs an instance each."""

import json
import pathlib

from mqt.qecc.cococo import layouts

pos = tuple[int, int]

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_scalable_layout():
    """Just runs an instance of every layout type. If no error occurs all is fine."""
    m = 5
    n = 5
    factories: list[pos] = []
    remove_edges = False
    # first check whether some error is thrown.
    for layout_type in ("single", "pair", "triple", "hex"):
        try:
            layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
        except Exception as e:  # noqa: PERF203
            msg = f"Problem with {layout_type} scalable layouts."
            raise ValueError(msg) from e

    # explicit tests.
    path = PROJECT_ROOT / "cococo/layouts_small_test.json"
    with pathlib.Path(path).open(encoding="utf-8") as f:
        dct_reference = json.load(f)

    factories = []
    remove_edges = False
    m = 2
    n = 2

    # single
    layout_type = "single"
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    assert set(data_qubit_locs) == {tuple(x) for x in dct_reference["single"]["data_qubit_locs"]}, (
        "data_qubit_locs for small single layout incorrect."
    )
    assert set(factory_ring) == {tuple(x) for x in dct_reference["single"]["factory_ring"]}, (
        "factory_ring for small single layout incorrect."
    )
    assert set(g.nodes()) == {tuple(x) for x in dct_reference["single"]["nodes"]}, (
        "nodes for small single layout incorrect."
    )
    assert set(g.edges()) == {(tuple(x[0]), tuple(x[1])) for x in dct_reference["single"]["edges"]}, (
        "edges for small single layout incorrect."
    )

    # pair
    layout_type = "pair"
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    assert set(data_qubit_locs) == {tuple(x) for x in dct_reference["pair"]["data_qubit_locs"]}, (
        "data_qubit_locs for small pair layout incorrect."
    )
    assert set(factory_ring) == {tuple(x) for x in dct_reference["pair"]["factory_ring"]}, (
        "factory_ring for small pair layout incorrect."
    )
    assert set(g.nodes()) == {tuple(x) for x in dct_reference["pair"]["nodes"]}, (
        "nodes for small pair layout incorrect."
    )
    assert set(g.edges()) == {(tuple(x[0]), tuple(x[1])) for x in dct_reference["pair"]["edges"]}, (
        "edges for small pair layout incorrect."
    )

    # triple
    layout_type = "triple"
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    assert set(data_qubit_locs) == {tuple(x) for x in dct_reference["triple"]["data_qubit_locs"]}, (
        "data_qubit_locs for small triple layout incorrect."
    )
    assert set(factory_ring) == {tuple(x) for x in dct_reference["triple"]["factory_ring"]}, (
        "factory_ring for small triple layout incorrect."
    )
    assert set(g.nodes()) == {tuple(x) for x in dct_reference["triple"]["nodes"]}, (
        "nodes for small triple layout incorrect."
    )
    assert set(g.edges()) == {(tuple(x[0]), tuple(x[1])) for x in dct_reference["triple"]["edges"]}, (
        "edges for small triple layout incorrect."
    )

    # hex
    layout_type = "hex"
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    assert set(data_qubit_locs) == {tuple(x) for x in dct_reference["hex"]["data_qubit_locs"]}, (
        "data_qubit_locs for small hex layout incorrect."
    )
    assert set(factory_ring) == {tuple(x) for x in dct_reference["hex"]["factory_ring"]}, (
        "factory_ring for small hex layout incorrect."
    )
    assert set(g.nodes()) == {tuple(x) for x in dct_reference["hex"]["nodes"]}, "nodes for small hex layout incorrect."
    assert set(g.edges()) == {(tuple(x[0]), tuple(x[1])) for x in dct_reference["hex"]["edges"]}, (
        "edges for small hex layout incorrect."
    )

    # hex with remove_edges=True
    layout_type = "hex"
    remove_edges = True
    g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
    assert set(data_qubit_locs) == {tuple(x) for x in dct_reference["hex_no_edges_data"]["data_qubit_locs"]}, (
        "data_qubit_locs for small hex (no edges data) layout incorrect."
    )
    assert set(factory_ring) == {tuple(x) for x in dct_reference["hex_no_edges_data"]["factory_ring"]}, (
        "factory_ring for small hex (no edges data) layout incorrect."
    )
    assert set(g.nodes()) == {tuple(x) for x in dct_reference["hex_no_edges_data"]["nodes"]}, (
        "nodes for small hex (no edges data) layout incorrect."
    )
    assert set(g.edges()) == {(tuple(x[0]), tuple(x[1])) for x in dct_reference["hex_no_edges_data"]["edges"]}, (
        "edges for small hex (no edges data) layout incorrect."
    )
