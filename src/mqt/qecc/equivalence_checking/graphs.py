# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Colored graph isomorphism used by the equivalence-checking decision procedures."""

from __future__ import annotations

from collections import Counter

import networkx as nx


def _colored_graph_isomorphism(
    graph1: nx.Graph, graph2: nx.Graph, *, edge_colors: bool = False
) -> dict[int, int] | None:
    """Find an isomorphism that preserves the ``color`` attribute of nodes and, optionally, edges.

    The graphs have to be either both simple graphs or both multigraphs. In multigraphs,
    the colors of the parallel edges between two nodes have to be preserved as a set.

    The node colors are first refined with the Weisfeiler-Lehman procedure. Differing refinements
    refute an isomorphism, while matching ones prune the search of the VF2 matcher, which otherwise
    backtracks heavily on the highly symmetric incidence graphs of codes.
    Both graphs are annotated with the refinement as ``refined_color`` node attribute.

    Returns:
        The node mapping from ``graph1`` to ``graph2``, or ``None`` if the graphs are not isomorphic.
    """
    for graph in (graph1, graph2):
        # the Weisfeiler-Lehman procedure of networkx does not support multigraphs
        hashes = nx.weisfeiler_lehman_subgraph_hashes(
            _merge_parallel_edges(graph) if isinstance(graph, nx.MultiGraph) else graph,
            node_attr="color",
            edge_attr="color" if edge_colors else None,
        )
        nx.set_node_attributes(graph, {node: node_hashes[-1] for node, node_hashes in hashes.items()}, "refined_color")

    if Counter(nx.get_node_attributes(graph1, "refined_color").values()) != Counter(
        nx.get_node_attributes(graph2, "refined_color").values()
    ):
        return None

    isomorphism = nx.algorithms.isomorphism
    categorical_edge_match = (
        isomorphism.categorical_multiedge_match if graph1.is_multigraph() else isomorphism.categorical_edge_match
    )
    matcher = isomorphism.GraphMatcher(
        graph1,
        graph2,
        node_match=isomorphism.categorical_node_match(["color", "refined_color"], [None, None]),
        edge_match=categorical_edge_match("color", None) if edge_colors else None,
    )
    return matcher.mapping if matcher.is_isomorphic() else None


def _merge_parallel_edges(graph: nx.MultiGraph) -> nx.Graph:
    """Merge parallel edges into single edges that are colored by the sorted colors of the merged edges."""
    merged = nx.Graph()
    merged.add_nodes_from(graph.nodes(data=True))
    for node, neighbors in graph.adjacency():
        for neighbor, parallel_edges in neighbors.items():
            colors = sorted(str(data.get("color")) for data in parallel_edges.values())
            merged.add_edge(node, neighbor, color=tuple(colors))
    return merged
