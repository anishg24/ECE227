from abc import ABC, abstractmethod
from typing import Iterable
import networkx as nx
from util import RNG

class PropagationAlgorithm(ABC):
    @abstractmethod
    def propagate(self, active_node: str, neighbors: list[str] | Iterable[str]) -> None:
        pass


class IndependentCascadeModel(PropagationAlgorithm, ABC):
    graph: nx.Graph
    probability: float

    def __init__(self, graph: nx.Graph, probability: float):
        assert 0. <= probability <= 1.

        self.graph = graph
        self.probability = probability

    def propagate(self, active_node, neighbors) -> None:
        for neighbor in neighbors:
            if not self.graph.nodes[neighbor]["active"]:
                self.graph.nodes[neighbor]["active"] = RNG.random() < self.probability