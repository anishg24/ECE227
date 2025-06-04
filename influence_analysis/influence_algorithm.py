import networkx as nx
from abc import ABC, abstractmethod
import random
from typing import List, Tuple
import networkx as nx
import tqdm
import psutil
import os

import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
from tqdm import tqdm
from influence_analysis.simulator import Simulator
import heapq
from joblib import Parallel, delayed
import os

class InfluenceAlgorithm(ABC):
    @abstractmethod
    def get_seed_nodes(self, **kwargs) -> list[str]:
        pass


class DegreeCentralityAlgorithm(InfluenceAlgorithm):
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












class GeneticAlgorithm(InfluenceAlgorithm, ABC):
    graph: nx.Graph
    seed_nodes: list[str]
    GA_params: dict

    def __init__(self, graph: nx.Graph, num_seed: int, GA_params: dict = None):
        self.graph = graph
        self.num_seed = num_seed
        if GA_params is None:
            self.GA_params = {
                "POP_SIZE": 50,  # population size
                "GENERATIONS": 100,  # number of generations
                "ELITE_COUNT": 2,  # number of elites
                "TOUR_SIZE": 4,  # tournament size
                "MUT_RATE": 0.1,  # mutation rate
                "N_SIM": 100,  # number of simulations
                "IC_PROB": 0.05,  # default edge activation probability
            }
        else:
            self.GA_params = GA_params

        edges = [
            (int(u), int(v))   # None → use default_p
            for u, v in self.graph.edges
        ]

        import fast_ic as fic

        self.fG = fic.Graph(edges)

        # set random seed for reproducibility
        random.seed(0)

    # FIXME: independent_cascade could use the one previously defined in the project.
    def independent_cascade(
        self,
        seeds: List[int],
        p: float = 0.1,
    ) -> int:
        """
        One stochastic run of the Independent Cascade model.

        Args
        ----
        G     : NetworkX graph.  Edges may carry a 'weight' attribute
                overriding p for that edge.
        seeds : Initial active nodes.
        p     : Default activation probability (used if no weight on edge).

        Returns
        -------
        Total number of activated nodes at the end of the cascade.
        """
        active = set(seeds)
        frontier = set(seeds)          # nodes that became active in the last round

        while frontier:
            new_frontier = set()
            for u in frontier:
                for v in self.graph.neighbors(u):
                    if v in active:          # already active → skip
                        continue
                    # Edge-specific probability if present, otherwise default p
                    prob = self.graph[u][v].get("weight", p)
                    if random.random() < prob:
                        active.add(v)
                        new_frontier.add(v)
            frontier = new_frontier
        return len(active)

    # ---------------------------------------------------------------------
    # ---------- 2.  Genetic operators ------------------------------------
    # ---------------------------------------------------------------------

    def one_point_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Cut both parents at the same random point and swap the tails."""
        k = len(parent1)
        cut = random.randint(1, k-1)          # forbid 0 and k (would copy intact)
        child1 = np.concatenate((parent1[:cut], parent2[cut:]))
        child2 = np.concatenate((parent2[:cut], parent1[cut:]))
        return child1, child2


    def mutate(self, chrom: List[int], nodes: List[int], mut_rate: float) -> None:
        """
        In-place per-gene mutation: with probability `mut_rate`
        replace that position by a **random** node ID.
        """
        for i in range(len(chrom)):
            if random.random() < mut_rate:
                chrom[i] = random.choice(nodes)

    def tournament_select(
            self,
            population: List[List[int]],
            fitnesses: List[float],
    ) -> List[int]:
        """Return the **chromosome** that wins a tour of `tour_size` random competitors."""
        contenders = random.sample(range(len(population)), self.GA_params["TOUR_SIZE"])
        best_idx   = max(contenders, key=lambda idx: fitnesses[idx])
        return population[best_idx][:]        # copy so we can modify later


    # ---------------------------------------------------------------------
    # ---------- 4.  GA main loop -----------------------------------------
    # ---------------------------------------------------------------------

    def evaluate_chromosome(self, chrom):
        """Evaluate a single chromosome by running multiple independent cascade simulations."""
        total = 0
        chrom = [node for node in chrom]
        for _ in range(self.GA_params["N_SIM"]):
            total += self.independent_cascade(chrom, self.GA_params["IC_PROB"])
        return total / self.GA_params["N_SIM"]

    def evaluate_population(
            self,
            population: List[List[int]],
            num_processes: int = None,
            fast_independent_cascade: bool = False
    ) -> List[float]:
        """
        Compute fitness for every chromosome – expensive step!
        Uses N_SIM repeated cascades per chromosome.
        Evaluates chromosomes in parallel using multiprocessing.

        Args:
            population: List of chromosomes to evaluate
            num_processes: Number of processes to use for parallel evaluation.
                          If None, uses the number of CPU cores.
        """
        print(f"Memory before evaluation: {get_memory_usage():.2f} MB")
        if fast_independent_cascade:
            fitness = [fic.evaluate_chromosome(self.fG, [int(ch) for ch in chrom], self.GA_params["N_SIM"], self.GA_params["IC_PROB"]) for chrom in tqdm.tqdm(population, desc="Evaluating population")]
        else:
            fitness = [self.evaluate_chromosome(chrom) for chrom in tqdm.tqdm(population, desc="Evaluating population")]
        print(f"Memory after evaluation: {get_memory_usage():.2f} MB")
        return fitness

    # Main genetic algorithm
    def get_seed_nodes(
            self,
            num_processes: int = 4
    ) -> Tuple[List[int], float]:
        """
        Run the GA and return the best seed set + its fitness.

        Args:
            num_processes: Number of processes to use for parallel evaluation.
        """
        print(f"Initial memory: {get_memory_usage():.2f} MB")
        nodes = list(self.graph.nodes())
        print(f"Memory after node list: {get_memory_usage():.2f} MB")

        # FIXME: initialise population randomly, this could be changed to a more effective one such as centrality based
        population = [np.random.choice(nodes, self.num_seed, replace=False) for _ in range(self.GA_params["POP_SIZE"])]
        print(f"Memory after population creation: {get_memory_usage():.2f} MB")

        best_chrom, best_fit = None, float("-inf")

        for gen in range(1, self.GA_params["GENERATIONS"] + 1):
            print(f"\nGeneration {gen}")
            print(f"Memory before fitness evaluation: {get_memory_usage():.2f} MB")
            fitness = self.evaluate_population(population, num_processes=num_processes)
            print(f"Memory after fitness evaluation: {get_memory_usage():.2f} MB")

            # record global best
            gen_best_idx = max(range(self.GA_params["POP_SIZE"]), key=lambda i: fitness[i])
            if fitness[gen_best_idx] > best_fit:
                best_chrom, best_fit = population[gen_best_idx][:], fitness[gen_best_idx]

            # ---------- elitism: copy ELITE_COUNT fittest unchanged
            elites_idx = sorted(range(self.GA_params["POP_SIZE"]), key=lambda i: fitness[i], reverse=True)[:self.GA_params["ELITE_COUNT"]]
            new_population = [population[i][:] for i in elites_idx]
            print(f"Memory after elitism: {get_memory_usage():.2f} MB")

            # ---------- create offspring until population is full
            while len(new_population) < self.GA_params["POP_SIZE"]:
                # parent selection
                parent1 = self.tournament_select(population, fitness)
                parent2 = self.tournament_select(population, fitness)

                # crossover (always)
                child1, child2 = self.one_point_crossover(parent1, parent2)

                # mutation (always scanned)
                self.mutate(child1, nodes, self.GA_params["MUT_RATE"])
                self.mutate(child2, nodes, self.GA_params["MUT_RATE"])

                new_population.extend([child1, child2])

            # trim in case we over-filled (can happen when POP_SIZE is odd)
            population = new_population[:self.GA_params["POP_SIZE"]]
            print(f"Memory after population update: {get_memory_usage():.2f} MB")

            # optional: progress log
            print(f"Gen {gen:3d}  |  best fitness so far = {best_fit:.2f}")

        return best_chrom, best_fit
