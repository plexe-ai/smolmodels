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
        if n != 1:
            raise NotImplementedError("Returning multiple nodes is not supported yet.")
        return [self.graph.good_nodes[random.randint(0, len(self.graph.nodes) - 1)]]

    def select_node_expand(self, n: int = 1) -> List[Node]:
        """
        Select a node to expand randomly from the graph.

        :param n: The number of nodes to select. Currently, only 1 is supported.
        :return: A list containing one randomly selected node.
        :raises NotImplementedError: If n is not 1.
        """
        if n != 1:
            raise NotImplementedError("Returning multiple nodes is not supported yet.")
        return self.select_node_enter()
