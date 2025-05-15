import networkx as nx
from random import sample

from propogation import PropagationAlgorithm
from influence_algorithm import InfluenceAlgorithm

class Simulator:
    graph: nx.Graph
    influence_alg: InfluenceAlgorithm
    prop_alg: PropagationAlgorithm

    def __init__(self, graph: nx.Graph, influence_alg: InfluenceAlgorithm, prop_alg: PropagationAlgorithm):
        self.graph = graph
        self.influence_alg = influence_alg
        self.prop_alg = prop_alg

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

    def seed(self) -> None:
        self.seed_nodes(self.influence_alg.get_seed_nodes())

    def seed_node(self, node: str) -> None:
        self.graph.nodes[node]["active"] = True

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

    def get_active_nodes(self) -> list[str]:
        return list(self.active_nodes_view.nodes())

    def get_num_active_nodes(self) -> int:
        return self.active_nodes_view.number_of_nodes()

    def get_inactive_nodes(self) -> list[str]:
        return list(self.inactive_nodes_view.nodes())

    def get_num_inactive_nodes(self) -> int:
        return self.inactive_nodes_view.number_of_nodes()

if __name__ == '__main__':
    from graphs import GraphGenerator
    from pathlib import Path
    from propogation import IndependentCascadeModel
    from influence_algorithm import DegreeCentralityAlgorithm
    from tqdm import tqdm

    graph = GraphGenerator.get_collab_graph(Path("../data/collab.txt"))
    prob = 0.3
    num_timestep = 1000
    num_seeds = 10
    influence_algorithm = DegreeCentralityAlgorithm(graph, num_seeds)
    prop_alg = IndependentCascadeModel(graph, prob)
    simulator = Simulator(graph, influence_algorithm, prop_alg)

    #simulator.seed_node('3466') # Seed one node manually
    #simulator.seed_random_nodes(num_seeds) # Seed 5 random nodes
    simulator.seed() # Seed according to the influence algorithm
    print(f"Seeded Nodes: {simulator.get_active_nodes()}")

    with tqdm(range(num_timestep), desc="Simulation Timestep") as t:
        print(f"Starting Simulation")
        for _ in t:
            simulator.timestep()
        print(f"Simulation Complete!")
        elapsed = t.format_dict['elapsed']
        print(f"Time: {elapsed:.4f} seconds ({elapsed/num_timestep:.4f} second per timestep)")
        print()

    print(f"Post-Simulation Results after {num_timestep} time steps")
    print(f"{simulator.get_num_active_nodes()} nodes activated")
    print(f"Percent Active: {(simulator.get_num_active_nodes()/graph.number_of_nodes())*100:.2f}%")
    print(simulator.get_active_nodes())
