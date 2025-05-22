import networkx as nx
from abc import ABC, abstractmethod
import random
from typing import List, Tuple
import networkx as nx
import tqdm

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
        child1 = parent1[:cut] + parent2[cut:]
        child2 = parent2[:cut] + parent1[cut:]
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

    def evaluate_chromosome(self, args):
        """Evaluate a single chromosome by running multiple independent cascade simulations."""
        i, chrom = args
        total = 0
        for _ in range(self.GA_params["N_SIM"]):
            total += self.independent_cascade(chrom, self.GA_params["IC_PROB"])
        return total / self.GA_params["N_SIM"]

    def evaluate_population(
            self,
            population: List[List[int]],
            num_processes: int = None
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
        from multiprocessing import Pool

        # Create process pool and map chromosomes to processes
        with Pool(processes=num_processes) as pool:
            # Pair each chromosome with its index
            chromosome_args = list(enumerate(population))
            # Map evaluation function across all chromosomes in parallel
            fitness = pool.map(self.evaluate_chromosome, chromosome_args)

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
        nodes = list(self.graph.nodes())

        # ----- initialise population randomly
        population = [random.sample(nodes, self.num_seed) for _ in range(self.GA_params["POP_SIZE"])]

        best_chrom, best_fit = None, float("-inf")

        for gen in range(1, self.GA_params["GENERATIONS"] + 1):
            fitness = self.evaluate_population(population, num_processes=num_processes)

            # record global best
            gen_best_idx = max(range(self.GA_params["POP_SIZE"]), key=lambda i: fitness[i])
            if fitness[gen_best_idx] > best_fit:
                best_chrom, best_fit = population[gen_best_idx][:], fitness[gen_best_idx]

            # ---------- elitism: copy ELITE_COUNT fittest unchanged
            elites_idx = sorted(range(self.GA_params["POP_SIZE"]), key=lambda i: fitness[i], reverse=True)[:self.GA_params["ELITE_COUNT"]]
            new_population = [population[i][:] for i in elites_idx]

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

            # optional: progress log
            # if gen % 10 == 0 or gen == 1:
            print(f"Gen {gen:3d}  |  best fitness so far = {best_fit:.2f}")

        return best_chrom, best_fit
