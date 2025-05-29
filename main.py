from influence_analysis.graphs import GraphGenerator
from influence_analysis.influence_algorithm import DegreeCentralityAlgorithm, EigenVectorCentralityAlgorithm, PageRankCentralityAlgorithm, GreedyAlgorithm

from graph_tool.dynamics import SIState

if __name__ == '__main__':
    graph = GraphGenerator.get_collab_graph()
    prob = 0.3
    num_timestep = 1000
    num_seeds = 5

    state = SIState(graph, prob)
    # state.iterate_sync(niter=num_timestep)
    # print(state.get_active())
    influence_algorithm = GreedyAlgorithm(graph, state, num_seeds)
    #
    influence_algorithm.run()
    # seeds = influence_algorithm.get_seed_nodes()
    # state.reset_active()
    # state.set_active(seeds)
    # num_active_nodes = state.iterate_sync(num_timestep)
    

    print(f"Post-Simulation Results after {num_timestep} time steps")
    print(f"{num_active_nodes} nodes activated")
    print(f"Percent Active: {(num_active_nodes/graph.get_vertices().size)*100:.2f}%")