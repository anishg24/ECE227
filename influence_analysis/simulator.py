import networkx as nx
from random import sample
import random
from pathlib import Path

from influence_analysis.graphs import GraphGenerator
from influence_analysis.propogation import PropagationAlgorithm, IndependentCascadeModel
import numpy as np
from numba import vectorize, uint8, uint32, float32, jit

@vectorize([uint8(uint32, float32)], nopython=True)
def propagate(n, threshold_prob):
    threshold = 1.0 - (1.0 - threshold_prob) ** n
    random_val = np.random.rand()

    return np.uint8(random_val < threshold)


class Simulator:
    graph: nx.Graph
    prop_alg: PropagationAlgorithm

    def __init__(self, graph: nx.Graph, prop_alg: PropagationAlgorithm, num_timestep: int):
        self.graph = graph
        self.prop_alg = prop_alg
        self.num_timestep = num_timestep

        def find_active(node: str) -> bool:
            return self.graph.nodes[node]["active"]

        def find_inactive(node: str) -> bool:
            return not self.graph.nodes[node]["active"]

        self.active_nodes_view = nx.subgraph_view(self.graph, filter_node=find_active)
        self.inactive_nodes_view = nx.subgraph_view(self.graph, filter_node=find_inactive)

        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()

    def timestep(self, disable_multiple_activation: bool = True) -> None:
        for node, attr in self.graph.nodes(data=True):
            if attr["active"] and (disable_multiple_activation and not attr["already_spread"]):
                self.prop_alg.propagate(node, self.graph.neighbors(node))
                self.graph.nodes[node]["already_spread"] = True
    
    def reset(self) -> None:
        for node, attr in self.graph.nodes(data=True):
            self.graph.nodes[node]["already_spread"] = False
            self.graph.nodes[node]["active"] = False

    def matrix_sim(self, seeds: list[int], n_iter: int = 100, prop_prob: np.float32 = np.float32(0.3)) -> int:
        activated_nodes: np.array = np.zeros(self.adj_matrix.shape[0], dtype=np.uint8)
        for i in seeds:
            activated_nodes[i] = 1

        currently_activated: np.array = activated_nodes.copy()
        for _ in range(n_iter):
            propagate_nodes = np.matmul(currently_activated, self.adj_matrix)
            propagate_nodes = propagate_nodes * (1 - currently_activated)
            propagate_nodes = propagate_nodes.astype(np.uint32)
            newly_active_nodes = propagate(propagate_nodes, prop_prob)

            currently_activated = newly_active_nodes.copy()
            activated_nodes = activated_nodes | newly_active_nodes

            if all(activated_nodes):
                break

        return np.bitwise_count(activated_nodes)
    
    def estimate_spread(self, seeds: list[int]) -> int:
        self.seed_nodes(seeds)
        rng = random.Random(227)
        activated_nodes = seeds.copy()
        time = 1
        while time <= self.num_timestep and len(activated_nodes) != self.graph.number_of_nodes:
            newly_activated_nodes = []
            for node in activated_nodes:
                if not self.graph.nodes[node]["already_spread"]:
                    res_act = self.prop_alg.propagate(node, self.graph.neighbors(node), rng)
                    self.graph.nodes[node]["already_spread"] = True
                    newly_activated_nodes.extend(res_act)
            if len(newly_activated_nodes) == 0:
                break
            activated_nodes.extend(newly_activated_nodes)
            time += 1
                
        num_active_nodes = len(activated_nodes)
        return num_active_nodes

    def seed_node(self, node: int) -> None:
        self.graph.nodes[node]["active"] = True

    def seed_nodes(self, nodes: list[int]) -> None:
        for node in nodes:
            self.seed_node(node)

    def seed_random_nodes(self, num_seed: int = 10) -> list[int]:
        seeds: list[int] = sample(list(self.graph.nodes()), num_seed)
        self.seed_nodes(seeds)
        return seeds

    def seed_random_percentage(self, percentage: float = 0.1) -> list[int]:
        assert 0 < percentage < 1.
        return self.seed_random_nodes(int(percentage * len(self.graph.nodes)))

    def get_active_nodes(self) -> list[int]:
        return list(self.active_nodes_view.nodes())

    def get_num_active_nodes(self) -> int:
        return self.active_nodes_view.number_of_nodes()

    def get_inactive_nodes(self) -> list[int]:
        return list(self.inactive_nodes_view.nodes())

    def get_num_inactive_nodes(self) -> int:
        return self.inactive_nodes_view.number_of_nodes()

if __name__ == '__main__':
    # graph = GraphGenerator.get_collab_graph(Path("../data/collab.txt"))
    # simulator = Simulator(graph, IndependentCascadeModel(graph, 0.3), 100)

    # simulator.matrix_sim([23], 100)

    prob = np.float32(0.3)

    adj_matrix = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]], dtype=np.int8)
    activated_nodes = np.array([1, 1, 0])

    propagate_nodes = np.matmul(activated_nodes, adj_matrix)
    propagate_nodes = propagate_nodes * (1 - activated_nodes)
    propagate_nodes = propagate_nodes.astype(np.uint32)

    newly_active_nodes = propagate(propagate_nodes, prob)

    activated_nodes = activated_nodes | newly_active_nodes

    print(propagate_nodes)
    print(newly_active_nodes)
    print(activated_nodes)