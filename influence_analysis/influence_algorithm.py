import networkx as nx
from abc import ABC, abstractmethod
from tqdm import tqdm
from influence_analysis.simulator_greedy import Simulator

class InfluenceAlgorithm(ABC):
    @abstractmethod
    def get_seed_nodes(self, **kwargs) -> list[str]:
        pass


class DegreeCentralityAlgorithm(InfluenceAlgorithm, ABC):
    graph: nx.Graph
    seed_nodes: list[str]

    def __init__(self, graph: nx.Graph, num_seed: int = None):
        self.graph = graph
        self.seed_nodes = list(nx.degree_centrality(self.graph).keys())[:num_seed]

    def get_seed_nodes(self) -> list[str]:
        return self.seed_nodes
    
class EigenVectorCentralityAlgorithm(InfluenceAlgorithm, ABC):
    graph: nx.Graph
    seed_nodes: list[str]

    def __init__(self, graph: nx.Graph, num_seed: int = None):
        self.graph = graph
        centrality = nx.eigenvector_centrality(self.graph, max_iter=10000)  # ensure convergence
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        self.seed_nodes = [node for node, _ in sorted_nodes[:num_seed]]
    def get_seed_nodes(self) -> list[str]:
        return self.seed_nodes
    
class PageRankCentralityAlgorithm(InfluenceAlgorithm, ABC):
    graph: nx.Graph
    seed_nodes: list[str]

    def __init__(self, graph: nx.Graph, num_seed: int = None, alpha: float = 0.85):
        self.graph = graph
        centrality = nx.pagerank(self.graph, alpha=alpha)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        self.seed_nodes = [node for node, _ in sorted_nodes[:num_seed]]

    def get_seed_nodes(self) -> list[str]:
        return self.seed_nodes
    
class GreedyAlgorithm(InfluenceAlgorithm, ABC):
    graph: nx.Graph
    seed_nodes: list[str]

    def __init__(self, graph: nx.Graph, simulator: Simulator, num_seed: int):
        self.graph = graph
        self.seed_nodes = []
        self.simulator = simulator
        self.num_seed = num_seed

    def get_seed_nodes(self) -> list[str]:
        return self.seed_nodes
    
    def run(self):
        candidates = set(self.graph.nodes)
        for _ in tqdm(range(self.num_seed), desc="Selecting seed nodes"):
            best_node = None
            best_gain = -1
            for node in tqdm(candidates - set(self.seed_nodes), leave=False, desc="Evaluating candidates"):
                trial_seeds = self.seed_nodes + [node]
                gain = self.simulator.estimate_spread(trial_seeds)
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
            
            self.seed_nodes.append(best_node)
            print(f"Current Seed Nodes: {self.seed_nodes}")
            print(f"{best_gain} nodes activated")
            print(f"Percent Active: {(best_gain/self.graph.number_of_nodes())*100:.2f}%")

    

        
    



    