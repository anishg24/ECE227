import networkx as nx
from random import sample
from propogation import PropagationAlgorithm

class Simulator:
    def __init__(self, graph: nx.Graph, prop_alg: PropagationAlgorithm):
        self.graph: nx.Graph = graph
        self.prop_alg: PropagationAlgorithm = prop_alg

        def find_active(node: str) -> bool:
            return self.graph.nodes[node]["active"]

        def find_inactive(node: str) -> bool:
            return not self.graph.nodes[node]["active"]

        self.active_nodes_view = nx.subgraph_view(self.graph, filter_node=find_active)
        self.inactive_nodes_view = nx.subgraph_view(self.graph, filter_node=find_inactive)

    def timestep(self) -> None:
        for node in self.graph.nodes:
            if self.graph.nodes[node]["active"]:
                for neighbor in self.graph.neighbors(node):
                    if not self.graph.nodes[neighbor]["active"]:
                        self.graph.nodes[neighbor]["active"] = self.prop_alg.propagate()

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
        return len(self.get_active_nodes())

    def get_inactive_nodes(self) -> list[str]:
        return list(self.inactive_nodes_view.nodes())

    def get_num_inactive_nodes(self) -> int:
        return len(self.get_inactive_nodes())

if __name__ == '__main__':
    from graphs import GraphGenerator
    from pathlib import Path
    from propogation import IndependentCascadeModel
    from tqdm import tqdm

    graph = GraphGenerator.get_collab_graph(Path("../data/collab.txt"))
    prob = 0.3
    num_timestep = 1000
    prop_alg = IndependentCascadeModel(prob)
    simulator = Simulator(graph, prop_alg)

    simulator.seed_node('3466') # Seed one node manually
    print(f"Seeded Nodes: {simulator.seed_random_nodes(5)}")

    print(f"Starting Simulation")
    for _ in tqdm(range(num_timestep), desc="Simulation Timestep"):
        simulator.timestep()
    print(f"Simulation Complete!")

    print("Post-Simulation Results")
    print(f"{simulator.get_num_active_nodes()} nodes activated: {simulator.get_active_nodes()}")
    print(f"Percent Active: {(simulator.get_num_active_nodes()/graph.number_of_nodes())*100:.2f}%")
