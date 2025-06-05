from abc import ABC, abstractmethod
from typing import Iterable
import networkx as nx
from random import Random

class PropagationAlgorithm(ABC):
    @abstractmethod
    def propagate(self, active_node: str, neighbors: list[str] | Iterable[str], rng: Random) -> list[str]:
        pass


class IndependentCascadeModel(PropagationAlgorithm, ABC):
    graph: nx.Graph
    probability: float

    def __init__(self, graph: nx.Graph, probability: float):
        assert 0. <= probability <= 1.

        self.graph = graph
        self.probability = probability
        self.rng = Random(227)

    def propagate(self, active_node, neighbors) -> list[str]:
        activated = []
        for neighbor in neighbors:
            if not self.graph.nodes[neighbor]["active"]:
                if self.rng.random() < self.probability:
                    self.graph.nodes[neighbor]["active"] = True
                    activated.append(neighbor)
        return activated
