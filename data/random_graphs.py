import networkx as nx
from pathlib import Path

def dump_graph(graph: nx.Graph, txt_file: Path) -> None:
    with txt_file.open("w") as txt:
        for (u, v) in graph.edges():
            txt.write(f"{u}\t{v}\n")


def get_random_graph(num_nodes: int, p: float = 0.05, directed: bool = False) -> nx.Graph:
        """
        Generate a deterministic random graph based on number of nodes.
        Uses Erdos-Renyi model with a fixed seed for reproducibility.
        """
        seed = 227
        graph = nx.gnp_random_graph(num_nodes, p, seed=seed, directed=directed)
        return graph 


def get_scale_free_graph(num_nodes: int, p: float = 0.05) -> nx.Graph:
        """
        Generate a deterministic scale-free graph based on number of nodes.
        """
        seed = 227
        graph = nx.scale_free_graph(num_nodes, seed=seed)
        return graph 

if __name__ == "__main__":
    prob = 0.3

    er_graph = get_random_graph(1500, prob)
    sf_graph = get_scale_free_graph(2000)

    dump_graph(er_graph, Path("./random.txt"))
    dump_graph(sf_graph, Path("./scale_free.txt"))


