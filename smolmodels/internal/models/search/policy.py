import abc
from typing import List

from smolmodels.internal.models.entities.graph import Graph
from smolmodels.internal.models.entities.node import Node


class SearchPolicy(abc.ABC):
    """
    Abstract base class for defining search policies on a graph.
    """

    @abc.abstractmethod
    def __init__(self, graph: Graph):
        """
        Initialize the search policy with a graph.

        :param graph: The graph on which the search policy will operate.
        """
        self.graph = graph

    @abc.abstractmethod
    def select_node_enter(self, n: int = 1) -> List[Node]:
        """
        Select nodes to enter the search.

        :param n: The number of nodes to select.
        :return: A list of selected nodes.
        """
        pass

    @abc.abstractmethod
    def select_node_expand(self, n: int = 1) -> List[Node]:
        """
        Select nodes to expand in the search.

        :param n: The number of nodes to select.
        :return: A list of selected nodes.
        """
        pass
