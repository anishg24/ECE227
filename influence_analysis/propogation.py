from abc import ABC, abstractmethod
from typing import Iterable
from random import Random
from graph_tool import Graph, VertexPropertyMap

class PropagationAlgorithm(ABC):
    @abstractmethod
    def propagate(self, neighbors: Iterable[int], active_property: VertexPropertyMap, rng: Random) -> list[int]:
        pass


class IndependentCascadeModel(PropagationAlgorithm, ABC):
    graph: Graph
    probability: float

    def __init__(self, graph: Graph, probability: float):
        assert 0. <= probability <= 1.

        self.graph = graph
        self.probability = probability

    def propagate(self, neighbors, active_property, rng) -> list[int]:
        activated = []
        for neighbor in neighbors:
            if not active_property[neighbor]:
                activation = rng.random() < self.probability
                active_property[neighbor] = activation
                if activation:
                    activated.append(neighbor)
        return activated