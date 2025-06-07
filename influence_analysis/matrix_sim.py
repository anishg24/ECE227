import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

# collab_graph = GraphGenerator.get_collab_graph(Path("./data/collab.txt"))
DTYPE = torch.float32
def sim_repeat(graph: nx.Graph, frontier, acted, adj, prop_prob: np.float32, device: torch.device) -> int:
    acted_count = 0
    while True:
        candidates_count = torch.mv(adj, frontier) # find the candidates and their counts
        candidates = candidates_count * (1 - acted)  # mask off the candidates that have already been acted
        acted = torch.logical_or(acted, candidates).to(DTYPE)  # update the acted nodes
        threasholds = 1 - torch.pow(1 -prop_prob, candidates_count)
        generated = torch.rand(graph.number_of_nodes(), dtype=torch.float32, device=device)
        # generated = torch.rand(collab_graph.number_of_nodes(), dtype=torch.float32).to(device)
        frontier = (generated < threasholds).to(DTYPE)
        if acted_count == acted.sum():
            break
        acted_count = acted.sum()
    return acted.sum()

def matrix_sim_torch(graph: nx.Graph, seeds: list[int], n_iter: int = 100, prop_prob: np.float32 = 0.3, device: torch.device = 'cuda') -> int:
    adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=DTYPE).T.to_sparse().to(device)

    total_active = 0
    for _ in tqdm(range(n_iter), desc="Matrix Sim"):
        frontier = torch.zeros(graph.number_of_nodes(), dtype=DTYPE).to(device)
        acted = torch.zeros(graph.number_of_nodes(), dtype=DTYPE).to(device)
        frontier[seeds] = 1

        # perform simulation
        total_active += sim_repeat(graph, frontier, acted, adj, prop_prob, device)

    return total_active / n_iter

# seeds = [1, 200, 234, 237, 997]
# matrix_sim_cuda(seeds, 100, 0.3, 'cuda')