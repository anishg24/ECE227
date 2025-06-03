import networkx as nx
from abc import ABC, abstractmethod
from tqdm import tqdm
from influence_analysis.simulator import Simulator
import heapq
from joblib import Parallel, delayed
import os

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
    
    # def run(self):
    #     activated_nodes = []
    #     candidates = set(self.graph.nodes)
    #     with tqdm(range(self.num_seed), desc="Selecting seed nodes") as t:
    #         for _ in t:
    #             best_node = None
    #             best_gain = -1
    #             for node in tqdm(candidates - set(self.seed_nodes), leave=False, desc="Evaluating candidates"):
    #                 trial_seeds = self.seed_nodes + [node]
    #                 gain = self.simulator.estimate_spread(trial_seeds)
    #                 if gain > best_gain:
    #                     best_gain = gain
    #                     best_node = node
                
    #             self.seed_nodes.append(best_node)
    #             activated_nodes.append(best_gain)
    #             print(f"\nCurrent Seed Nodes: {self.seed_nodes}")
    #             print(f"{best_gain} nodes activated")
    #             print(f"Percent Active: {(best_gain/self.graph.number_of_nodes())*100:.2f}%")
    #             elapsed = t.format_dict['elapsed']
    #             elapsed_str = t.format_interval(elapsed)
    #             print(f"Total elapsed time: {elapsed_str}")
    #     return self.seed_nodes, activated_nodes

    def run(self):
        from joblib import Parallel, delayed
        activated_nodes = []
        candidates = set(self.graph.nodes)

        def compute_gain(node):
            trial_seeds = self.seed_nodes + [node]
            gain = self.simulator.estimate_spread(trial_seeds)
            self.simulator.reset()
            return node, gain

        with tqdm(range(self.num_seed), desc="Selecting seed nodes") as t:
            for _ in t:
                remaining = list(candidates - set(self.seed_nodes))

                results = Parallel(n_jobs=40)(delayed(compute_gain)(node) for node in tqdm(remaining, leave=False, desc="Evaluating candidates"))
                best_node, best_gain = max(results, key=lambda x: x[1])

                self.seed_nodes.append(best_node)
                activated_nodes.append(best_gain)

                print(f"\nCurrent Seed Nodes: {self.seed_nodes}")
                print(f"{best_gain} nodes activated")
                print(f"Percent Active: {(best_gain / self.graph.number_of_nodes()) * 100:.2f}%")

                elapsed = t.format_dict['elapsed']
                elapsed_str = t.format_interval(elapsed)
                print(f"Total elapsed time: {elapsed_str}")

        return self.seed_nodes, activated_nodes



class CostEffectiveLazyForwardAlgorithm(InfluenceAlgorithm, ABC):
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
        activated_nodes = []
        self.seed_nodes = []
        candidates = set(self.graph.nodes)

        # Step 1: Initialize priority queue with initial marginal gains
        priority_queue = []
        base_spread = 0
        print("Computing initial marginal gains...")
        def compute_gain(v):
            gain = self.simulator.estimate_spread([v])
            self.simulator.reset()
            return (-gain, v, 0)
        
        priority_queue = Parallel(n_jobs=os.cpu_count()//2)(
            delayed(compute_gain)(v) for v in tqdm(candidates, desc="Initializing CELF queue")
        )
        heapq.heapify(priority_queue)

        with tqdm(range(self.num_seed), desc="Selecting seed nodes") as t:
            for i in t:
                while True:
                    neg_gain, v, last_updated = heapq.heappop(priority_queue)
                    if last_updated == len(self.seed_nodes):
                        # Gain is valid, select the node
                        self.seed_nodes.append(v)
                        new_spread = self.simulator.estimate_spread(self.seed_nodes.copy())
                        self.simulator.reset()
                        marginal_gain = new_spread - base_spread
                        base_spread = new_spread
                        activated_nodes.append(marginal_gain)

                        print(f"\nCurrent Seed Nodes: {self.seed_nodes}")
                        print(f"{marginal_gain:.2f} new nodes activated")
                        print(f"Percent Active: {(base_spread / self.graph.number_of_nodes()) * 100:.2f}%")
                        break
                    else:
                        # Recompute gain with current seed set
                        updated_gain = self.simulator.estimate_spread(self.seed_nodes.copy() + [v]) - base_spread
                        self.simulator.reset()
                        heapq.heappush(priority_queue, (-updated_gain, v, len(self.seed_nodes)))
                    

        return self.seed_nodes, activated_nodes
    
    # def run(self):
    #     activated_nodes = []
    #     self.seed_nodes = []
    #     candidates = set(self.graph.nodes)

    #     # Step 1: Initialize priority queue with initial marginal gains
    #     priority_queue = []
    #     base_spread = 0
    #     print("Computing initial marginal gains...")
    #     for v in tqdm(candidates, desc="Initializing CELF queue"):
    #         gain = self.simulator.estimate_spread([v])
    #         self.simulator.reset()
    #         heapq.heappush(priority_queue, (-gain, v, 0)) 

    #     with tqdm(range(self.num_seed), desc="Selecting seed nodes") as t:
    #         for i in t:
    #             while True:
    #                 neg_gain, v, last_updated = heapq.heappop(priority_queue)

    #                 if last_updated == len(self.seed_nodes):
    #                     # Gain is valid, select the node
    #                     self.seed_nodes.append(v)
    #                     new_spread = self.simulator.estimate_spread(self.seed_nodes)
    #                     self.simulator.reset()
    #                     marginal_gain = new_spread - base_spread
    #                     base_spread = new_spread
    #                     activated_nodes.append(marginal_gain)

    #                     print(f"\nCurrent Seed Nodes: {self.seed_nodes}")
    #                     print(f"{marginal_gain:.2f} new nodes activated")
    #                     print(f"Percent Active: {(base_spread / self.graph.number_of_nodes()) * 100:.2f}%")

    #                     break
    #                 else:
    #                     # Recompute gain with current seed set
    #                     updated_gain = self.simulator.estimate_spread(self.seed_nodes + [v]) - base_spread
    #                     self.simulator.reset()
    #                     heapq.heappush(priority_queue, (-updated_gain, v, len(self.seed_nodes)))

    #     return self.seed_nodes, activated_nodes



    

        
    



    