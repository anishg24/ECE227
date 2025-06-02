from graph_tool import Graph, VertexPropertyMap
from random import sample
import random
from influence_analysis.propogation import PropagationAlgorithm

class Simulator:
    graph: Graph
    prop_alg: PropagationAlgorithm

    def __init__(self, graph: Graph, prop_alg: PropagationAlgorithm, num_timestep: int):
        self.graph = graph
        self.prop_alg = prop_alg
        self.num_timestep = num_timestep

        self.active = self.graph.new_vertex_property("bool")
        self.active.set_value(False)

        self.already_spread = self.graph.new_vertex_property("bool")
        self.already_spread.set_value(False)

    def timestep(self, disable_multiple_activation: bool = True) -> None:
        for node, attr in self.graph.nodes(data=True):
            if attr["active"] and (disable_multiple_activation and not attr["already_spread"]):
                self.prop_alg.propagate(node, self.graph.neighbors(node))
                self.graph.nodes[node]["already_spread"] = True
    
    def reset(self) -> None:
        self.active.set_value(False)
        self.already_spread.set_value(False)
    
    def estimate_spread(self, seeds: list[str]) -> int:
        self.seed_nodes(seeds)
        rng = random.Random(227)
        activated_nodes = set(sorted(seeds))
        time = 1
        num_vertices = len(self.graph.get_vertices())
        while time <= self.num_timestep and len(activated_nodes) != num_vertices:
            newly_activated_nodes = set()
            for node in sorted(activated_nodes):
                vertex = self.graph.vertex(node)
                if not self.already_spread[vertex]:
                    res_act = self.prop_alg.propagate(self.graph.iter_out_neighbours(vertex), self.active, rng)
                    self.already_spread[vertex] = True
                    newly_activated_nodes = newly_activated_nodes.union(set(res_act))
            activated_nodes = activated_nodes.union(newly_activated_nodes)
            time += 1
                
        num_active_nodes = len(activated_nodes)
        return num_active_nodes

    def seed_node(self, node: str) -> None:
        self.active[self.graph.vertex(node)] = True

    def seed_nodes(self, nodes: list[str]) -> None:
        for node in nodes:
            self.seed_node(node)

    def seed_random_nodes(self, num_seed: int = 10) -> list[str]:
        seeds: list[str] = sample(list(self.graph.nodes()), num_seed)
        self.seed_nodes(seeds)
        return seeds

    def seed_random_percentage(self, percentage: float = 0.1) -> list[str]:
        assert 0 < percentage < 1.
        return self.seed_random_nodes(int(percentage * len(self.graph.nodes)))