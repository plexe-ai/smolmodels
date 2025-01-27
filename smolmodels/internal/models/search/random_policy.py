"""
Module: random_policy

This module defines the `RandomSearchPolicy` class, which implements a search policy
that selects nodes randomly from a graph.

Classes:
    - RandomSearchPolicy: A search policy that selects nodes randomly from a graph.

Dependencies:
    - Graph: Represents a directed graph structure consisting of nodes and edges.
    - Node: Represents a node in the graph, with attributes for managing connections
      and execution results.
    - SearchPolicy: Abstract base class for defining search policies on a graph.

Example Usage:
    >>> from smolmodels.internal.models.entities.graph import Graph
    >>> from smolmodels.internal.models.search.random_policy import RandomSearchPolicy
    >>> graph = Graph()
    >>> policy = RandomSearchPolicy(graph)
    >>> selected_node = policy.select_node_enter()
"""

import random
from typing import List

from smolmodels.internal.models.entities.graph import Graph
from smolmodels.internal.models.entities.node import Node
from smolmodels.internal.models.search.policy import SearchPolicy


class RandomSearchPolicy(SearchPolicy):
    """
    A search policy that selects nodes randomly from a graph.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the RandomSearchPolicy with a graph.

        :param graph: The graph to search.
        """
        super().__init__(graph)

    def select_node_enter(self, n: int = 1) -> List[Node]:
        """
        Select a node to enter randomly from the graph.

        :param n: The number of nodes to select. Currently, only 1 is supported.
        :return: A list containing one randomly selected node.
        :raises NotImplementedError: If n is not 1.
        """
        if not self.graph.good_nodes:
            return [self.graph.nodes[0]] if self.graph.nodes else []
        return random.sample(
            self.graph.good_nodes, min(n, len(self.graph.good_nodes))
        )  # todo: this should actually be entering unexplored nodes

    def select_node_expand(self, n: int = 1) -> List[Node]:
        """
        Select a node to expand randomly from the graph.

        :param n: The number of nodes to select. Currently, only 1 is supported.
        :return: A list containing one randomly selected node.
        :raises NotImplementedError: If n is not 1.
        """
        return self.select_node_enter()
