import networkx as nx
from abc import ABC, abstractmethod


class InfluenceAlgorithm(ABC):
    @abstractmethod
    def get_seed_nodes(self, **kwargs) -> list[str]:
        pass


class DegreeCentralityAlgorithm(InfluenceAlgorithm):
    graph: nx.Graph
    seed_nodes: list[str]

    def __init__(self, graph: nx.Graph, num_seed: int = None):
        self.graph = graph
        self.seed_nodes = list(nx.degree_centrality(self.graph).keys())[:num_seed]

    def get_seed_nodes(self) -> list[str]:
        return self.seed_nodes