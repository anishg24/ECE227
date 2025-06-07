from abc import ABC, abstractmethod
import random
from typing import List, Tuple
import networkx as nx
import tqdm
import psutil
from joblib import Parallel, delayed
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
import time
from influence_analysis.matrix_sim import matrix_sim_torch


class InfluenceAlgorithm(ABC):
    @abstractmethod
    def get_seed_nodes(self, **kwargs) -> list[int]:
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

class GreedyAlgorithm(InfluenceAlgorithm):
    graph: nx.Graph
    seed_nodes: list[int]

    def __init__(self, graph: nx.Graph, simulator: Simulator, num_seed: int):
        self.graph = graph
        self.seed_nodes = []
        self.simulator = simulator
        self.num_seed = num_seed

    def get_seed_nodes(self) -> list[int]:
        return self.seed_nodes

    def run(self):
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



class CostEffectiveLazyForwardAlgorithm(InfluenceAlgorithm):
    graph: nx.Graph
    seed_nodes: list[int]

    def __init__(self, graph: nx.Graph, simulator: Simulator, num_seed: int):
        self.graph = graph
        self.seed_nodes = []
        self.simulator = simulator
        self.num_seed = num_seed

    def get_seed_nodes(self) -> list[int]:
        return self.seed_nodes
    
    def run(self, num_trials=20):
        start_time = time.time()
        activated_nodes = []
        self.seed_nodes = []
        candidates = set(self.graph.nodes)
        priority_queue = []
        base_spread = 0

        def compute_gain(v):
            gain = self.simulator.average_spread([v], num_trials)
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
                        self.seed_nodes.append(v)
                        new_spread = self.simulator.average_spread(self.seed_nodes.copy(), num_trials)
                        marginal_gain = new_spread - base_spread
                        base_spread = new_spread
                        activated_nodes.append(marginal_gain)



                        elapsed = time.time() - start_time
                        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                        break
                    else:
                        updated_gain = self.simulator.average_spread(self.seed_nodes.copy() + [v], num_trials) - base_spread
                        heapq.heappush(priority_queue, (-updated_gain, v, len(self.seed_nodes)))


        return self.seed_nodes, activated_nodes


class GeneticAlgorithm(InfluenceAlgorithm, ABC):
    graph: nx.Graph
    seed_nodes: list[str]
    GA_params: dict

    def __init__(self, graph: nx.Graph, simulator: Simulator, num_seed: int, GA_params: dict = None, memory_trace: bool = False):
        """
        Initialize a Genetic Algorithm for influence maximization.

        Parameters
        ----------
        graph : nx.Graph
            The input graph on which to run influence maximization
        num_seed : int
            Number of seed nodes to select
        GA_params : dict, optional
            Dictionary of genetic algorithm parameters:
            - POP_SIZE: Population size (default: 50)
            - GENERATIONS: Number of generations to run (default: 100)
            - ELITE_COUNT: Number of elite solutions to preserve (default: 2)
            - TOUR_SIZE: Tournament selection size (default: 4)
            - MUT_RATE: Mutation rate (default: 0.1)
            - N_SIM: Number of Monte Carlo simulations (default: 100)
            - IC_PROB: Edge activation probability for IC model (default: 0.05)
        """
        self.graph = graph
        self.num_seed = num_seed
        self.simulator = simulator
        self.memory_trace = memory_trace
        if GA_params is None:
            self.GA_params = {
                "POP_SIZE": 50,  # population size
                "GENERATIONS": 10,  # number of generations
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

        try:
            import fast_ic as fic
            self.fG = fic.Graph(edges)
        except ImportError:
            print("Fast Independent Cascade not found, using Python IC model")
            self.fG = None

        # set random seed for reproducibility
        random.seed(0)

    # ---------------------------------------------------------------------
    # -------------  Genetic operators ------------------------------------
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
    # -------------  GA main loop -----------------------------------------
    # ---------------------------------------------------------------------

    def evaluate_chromosome(self, chrom):
        """Evaluate a single chromosome by running multiple independent cascade simulations."""
        total = 0
        chrom = [node for node in chrom]
        for _ in range(self.GA_params["N_SIM"]):
            # total += self.independent_cascade(chrom, self.GA_params["IC_PROB"])
            total += self.simulator.estimate_spread(chrom)
            self.simulator.reset()
        return total / self.GA_params["N_SIM"]

    def evaluate_population(
            self,
            population: List[List[int]],
            # fast_independent_cascade: bool = False
    ) -> List[float]:
        """
        Compute fitness for every chromosome – expensive step!
        Uses N_SIM repeated cascades per chromosome.
        Evaluates chromosomes in parallel using multiprocessing.

        Args:
            population: List of chromosomes to evaluate
        """
        if self.memory_trace: print(f"Memory before evaluation: {get_memory_usage():.2f} MB")
        # if fast_independent_cascade:
        #     fitness = [fic.evaluate_chromosome(self.fG, [int(ch) for ch in chrom], self.GA_params["N_SIM"], self.GA_params["IC_PROB"]) for chrom in tqdm.tqdm(population, desc="Evaluating population")]
        # else:
        fitness = Parallel(n_jobs=os.cpu_count()//2)(delayed(matrix_sim_torch)(self.graph, chrom, self.GA_params["N_SIM"], self.GA_params["IC_PROB"]) for chrom in tqdm(population, desc="Evaluating population", leave=False))
        if self.memory_trace: print(f"Memory after evaluation: {get_memory_usage():.2f} MB")
        return fitness

    # Main genetic algorithm
    def run(
            self,
    ) -> Tuple[List[int], float]:
        """
        Run the GA and return the best seed set + its fitness.

        Args:
            num_processes: Number of processes to use for parallel evaluation.
        """
        if self.memory_trace: print(f"Initial memory: {get_memory_usage():.2f} MB")
        nodes = list(self.graph.nodes())
        if self.memory_trace: print(f"Memory after node list: {get_memory_usage():.2f} MB")

        # FIXME: initialise population randomly, this could be changed to a more effective one such as centrality based
        population = [np.random.choice(nodes, self.num_seed, replace=False) for _ in range(self.GA_params["POP_SIZE"])]
        if self.memory_trace: print(f"Memory after population creation: {get_memory_usage():.2f} MB")

        best_chrom, best_fit = None, float("-inf")
        with tqdm(range(self.GA_params["GENERATIONS"]), desc=f"Running GA", leave=False, postfix=f"best fitness: {best_fit:.2f}") as t:
            for _ in t:
                if self.memory_trace: print(f"Memory before fitness evaluation: {get_memory_usage():.2f} MB")
                fitness = self.evaluate_population(population)
                if self.memory_trace: print(f"Memory after fitness evaluation: {get_memory_usage():.2f} MB")

                # record global best
                gen_best_idx = max(range(self.GA_params["POP_SIZE"]), key=lambda i: fitness[i])
                if fitness[gen_best_idx] > best_fit:
                    best_chrom, best_fit = population[gen_best_idx][:], fitness[gen_best_idx]

                # ---------- elitism: copy ELITE_COUNT fittest unchanged
                elites_idx = sorted(range(self.GA_params["POP_SIZE"]), key=lambda i: fitness[i], reverse=True)[:self.GA_params["ELITE_COUNT"]]
                new_population = [population[i][:] for i in elites_idx]
                if self.memory_trace: print(f"Memory after elitism: {get_memory_usage():.2f} MB")

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
                if self.memory_trace: print(f"Memory after population update: {get_memory_usage():.2f} MB")

                t.set_postfix(best_fit=best_fit)
                t.update()

        self.seed_nodes = best_chrom
        self.fitness_score = best_fit
        return best_chrom, best_fit

    def get_seed_nodes(self) -> list[int]:
        return self.seed_nodes.tolist()
