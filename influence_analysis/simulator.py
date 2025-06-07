import networkx as nx
from random import sample
import random
from influence_analysis.propogation import PropagationAlgorithm


class Simulator:
    graph: nx.Graph
    prop_alg: PropagationAlgorithm

    def __init__(self, graph: nx.Graph, prop_alg: PropagationAlgorithm):
        self.graph = graph
        self.prop_alg = prop_alg
        # self.num_timestep = num_timestep

        def find_active(node: str) -> bool:
            return self.graph.nodes[node]["active"]

        def find_inactive(node: str) -> bool:
            return not self.graph.nodes[node]["active"]

        self.active_nodes_view = nx.subgraph_view(self.graph, filter_node=find_active)
        self.inactive_nodes_view = nx.subgraph_view(self.graph, filter_node=find_inactive)

    def timestep(self, disable_multiple_activation: bool = True) -> None:
        for node, attr in self.graph.nodes(data=True):
            if attr["active"] and (disable_multiple_activation and not attr["already_spread"]):
                self.prop_alg.propagate(node, self.graph.neighbors(node))
                self.graph.nodes[node]["already_spread"] = True
    
    def reset(self) -> None:
        for node in self.graph.nodes():
            self.graph.nodes[node]["already_spread"] = False
            self.graph.nodes[node]["active"] = False

    def average_spread(self, seeds: list[str], num_trials: int = 40) -> float:
        total = 0.0
        for trial in range(num_trials):
            self.reset()
            spread = self.estimate_spread(seeds)
            total += spread
        return total / num_trials

    
    def estimate_spread(self, seeds: list[int]) -> int:
        self.seed_nodes(seeds)
        # rng = random.Random(227)
        activated_nodes = set(seeds.copy())

        time = 1
        while len(activated_nodes) != self.graph.number_of_nodes():
            newly_activated_nodes = set()
            for node in activated_nodes:
                if not self.graph.nodes[node]["already_spread"]:
                    res_act = self.prop_alg.propagate(node, self.graph.neighbors(node))
                    self.graph.nodes[node]["already_spread"] = True
                    newly_activated_nodes.update(res_act)
            if not newly_activated_nodes:
                break
            activated_nodes.update(newly_activated_nodes)
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