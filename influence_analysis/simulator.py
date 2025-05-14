import networkx as nx
from propogation import PropagationAlgorithm

class Simulator:
    def __init__(self, graph: nx.Graph, prop_alg: PropagationAlgorithm):
        self.graph: nx.Graph = graph
        self.prop_alg: PropagationAlgorithm = prop_alg

        self.current_node: str = ''

        def find_inactive_neighbor(node: str) -> bool:
            return self.graph.has_edge(self.current_node, node) and not self.graph.nodes[node]["active"]

        def find_active(node: str) -> bool:
            return self.graph.nodes[node]["active"]

        self.inactive_neighbors_view = nx.subgraph_view(self.graph, filter_node=find_inactive_neighbor)
        self.active_nodes_view = nx.subgraph_view(self.graph, filter_node=find_active)

    def timestep(self) -> None:
        for node in self.graph.nodes:
            self.current_node = node
            for neighbor in self.inactive_neighbors_view.nodes():
                self.graph.nodes[neighbor]["active"] = self.prop_alg.propagate()

    def get_active_nodes(self) -> list[str]:
        return list(self.active_nodes_view.nodes())

    def get_num_active_nodes(self) -> int:
        return len(self.get_active_nodes())

if __name__ == '__main__':
    from graphs import GraphGenerator
    from pathlib import Path
    from propogation import IndependentCascadeModel

    graph = GraphGenerator.get_collab_graph(Path("../data/collab.txt"))
    graph.nodes['3466']["active"] = True
    prop_alg = IndependentCascadeModel(0.3)
    simulator = Simulator(graph, prop_alg)

    print("started simulation")
    simulator.timestep()
    print("ran simulation")
    print(simulator.get_active_nodes())
    print(simulator.get_num_active_nodes())
