from influence_analysis.graphs import GraphGenerator
from influence_analysis.propogation import IndependentCascadeModel
from influence_analysis.influence_algorithm import GreedyAlgorithm, CostEffectiveLazyForwardAlgorithm, GeneticAlgorithm
from influence_analysis.simulator import Simulator

from argparse import ArgumentParser
from enum import Enum
from time import time

class GraphType(Enum):
    SOCIAL = "social"
    COMMUNICATION = "comm"
    COLLABORATION = "collab"
    RANDOM = "random"
    SCALE_FREE = "scale_free"

    def get(self):
        if self == GraphType.SOCIAL:
            return GraphGenerator.get_social_graph()
        elif self == GraphType.COMMUNICATION:
            return GraphGenerator.get_comm_graph()
        elif self == GraphType.COLLABORATION:
            return GraphGenerator.get_collab_graph()
        elif self == GraphType.RANDOM:
            return GraphGenerator.get_random_graph()
        elif self == GraphType.SCALE_FREE:
            return GraphGenerator.get_scale_free_graph()

    def __str__(self):
        return self.value[0]

class SeedAlgorithm(Enum):
    GREEDY = "greedy"
    CELF = "celf"
    GENETIC = "genetic"

    def get(self):
        if self == SeedAlgorithm.GREEDY:
            return GreedyAlgorithm
        elif self == SeedAlgorithm.CELF:
            return CostEffectiveLazyForwardAlgorithm
        elif self == SeedAlgorithm.GENETIC:
            return GeneticAlgorithm

    def __str__(self):
        return self.value[0]

if __name__ == '__main__':

    parser = ArgumentParser(
        description="Simulate influence dynamics on a given graph and find optimal seeds based on specified parameters."
    )

    # Argument for graph type
    parser.add_argument(
        '--graph_type',
        type=GraphType,
        choices=list(GraphType),
        default=GraphType.COLLABORATION,
        help=f"Type of graph to generate. Choices: {[str(x) for x in GraphType]}. (default: collab)"
    )

    # Argument for probability
    parser.add_argument(
        '-p',
        '--prop_prob',
        type=float,
        default=0.3,
        help="Probability for propagation between nodes every timestep. (default: 0.3)"
    )

    # Argument for number of timesteps
    parser.add_argument(
        '-N',
        '--num_timesteps',
        type=int,
        default=100,
        help="Number of simulation timesteps. (default: 100)"
    )

    # Argument for number of seeds to find
    parser.add_argument(
        '-k',
        '--num_seeds',
        type=int,
        default=3,
        help="Number of seeds (nodes) to identify using the specified algorithm. (default: 3)"
    )

    # Argument for seed finding algorithm
    parser.add_argument(
        '--seed_alg',
        type=SeedAlgorithm,
        choices=list(SeedAlgorithm),
        default=SeedAlgorithm.GREEDY,
        help=f"Algorithm to use for finding seeds. Choices: {[str(x) for x in SeedAlgorithm]}. (default: greedy)"
    )

    args = parser.parse_args()

    graph = args.graph_type.get()
    prob = args.prop_prob
    num_timestep = args.num_timesteps
    num_seeds = args.num_seeds
    prop_alg = IndependentCascadeModel(graph, prob)
    simulator = Simulator(graph, prop_alg, num_timestep)
    seed_alg = args.seed_alg.get()(graph, simulator, num_seeds) # This might have to change if we use genetic
    seed_start_time = time()
    seed_alg.run()
    seed_end_time = time()

    seeds = seed_alg.get_seed_nodes()
    simulator.reset()
    sim_start_time = time()
    num_active_nodes = simulator.estimate_spread(seeds)
    sim_end_time = time()

    print(f"Seed Nodes found in {seed_end_time-seed_start_time} seconds: {seeds}")
    print(f"Post-Simulation Results after {num_timestep} time steps")
    print(f"{num_active_nodes} nodes activated in {sim_end_time-sim_start_time} seconds")
    print(f"Percent Active: {(num_active_nodes/graph.number_of_nodes())*100:.2f}%")