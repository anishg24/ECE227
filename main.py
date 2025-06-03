from influence_analysis.graphs import GraphGenerator
from influence_analysis.propogation import IndependentCascadeModel
from influence_analysis.influence_algorithm import DegreeCentralityAlgorithm, EigenVectorCentralityAlgorithm, PageRankCentralityAlgorithm, GreedyAlgorithm, CostEffectiveLazyForwardAlgorithm
from influence_analysis.simulator import Simulator

if __name__ == '__main__':
    graph = GraphGenerator.get_collab_graph()
    # graph = GraphGenerator.get_random_graph(num_nodes=5000)
    prob = 0.3
    num_timestep = 100
    num_seeds = 5

    prop_alg = IndependentCascadeModel(graph, prob)
    simulator = Simulator(graph, prop_alg, num_timestep)
    # influence_algorithm = GreedyAlgorithm(graph, simulator, num_seeds)
    influence_algorithm = CostEffectiveLazyForwardAlgorithm(graph, simulator, num_seeds)

    influence_algorithm.run()
    seeds = influence_algorithm.get_seed_nodes()
    num_active_nodes = simulator.estimate_spread(seeds)
    simulator.reset()
    

    print(f"Post-Simulation Results after {num_timestep} time steps")
    print(f"{num_active_nodes} nodes activated")
    print(f"Percent Active: {(num_active_nodes/graph.number_of_nodes())*100:.2f}%")
    #print(simulator.get_active_nodes())