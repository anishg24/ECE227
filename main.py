from influence_analysis.graphs import GraphGenerator
from influence_analysis.propogation import IndependentCascadeModel
from influence_analysis.influence_algorithm import DegreeCentralityAlgorithm, EigenVectorCentralityAlgorithm, PageRankCentralityAlgorithm, GreedyAlgorithm, CostEffectiveLazyForwardAlgorithm
from influence_analysis.simulator import Simulator
import logging

if __name__ == '__main__':
    graph = GraphGenerator.get_social_graph()
    # graph = GraphGenerator.get_random_graph(num_nodes=50)
    prob = 0.3
    num_timestep = 20
    num_seeds = 5
    num_trials = 100
    logging.basicConfig(
    filename="./logging/social_graph.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("comm_graph")

    logger.info(f"number of trials {num_trials}")

    prop_alg = IndependentCascadeModel(graph, prob)
    simulator = Simulator(graph, prop_alg)
    # influence_algorithm = GreedyAlgorithm(graph, simulator, num_seeds)
    influence_algorithm = CostEffectiveLazyForwardAlgorithm(graph, simulator, num_seeds)

    influence_algorithm.run(logger, num_trials)
    seeds = influence_algorithm.get_seed_nodes()
    # num_active_nodes = simulator.average_spread(seeds)
    # simulator.reset()

    # print(f"{num_active_nodes} nodes activated")
    # print(f"Percent Active: {(num_active_nodes/graph.number_of_nodes())*100:.2f}%")