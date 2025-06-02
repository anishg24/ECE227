from influence_analysis.graphs import GraphGenerator
from influence_analysis.influence_algorithm import DegreeCentralityAlgorithm, EigenVectorCentralityAlgorithm, \
    PageRankCentralityAlgorithm, GreedyAlgorithm, InfluenceAlgorithm

from graph_tool.dynamics import SIState
from graph_tool.draw import graphviz_draw
from graph_tool.collection import data
from matplotlib import pyplot as plt

from influence_analysis.propogation import IndependentCascadeModel
from influence_analysis.simulator import Simulator

if __name__ == '__main__':
    graph = GraphGenerator.get_collab_graph()
    prob = 0.3
    num_timestep = 1000
    num_seeds = 5

    # inf_state = graph.new_vertex_property("bool")
    # inf_state.set_value(False)
    # active_states = [graph.vertex(i) for i in range(num_seeds)]
    # for v in active_states:
    #     inf_state[v] = True
    #
    # state = SIState(graph, beta=prob, s=inf_state)
    # state.reset_active()
    # state.set_active(active_states)
    # print(list(state.get_state()))
    # num_active_nodes = state.iterate_sync(num_timestep)
    prop_alg = IndependentCascadeModel(graph, prob)
    simulator = Simulator(graph, prop_alg, num_timestep)
    influence_algorithm = GreedyAlgorithm(graph, simulator, num_seeds)
    influence_algorithm.run()
    seeds = influence_algorithm.get_seed_nodes()
    # state.reset_active()
    # state.set_active(seeds)
    num_active_nodes = simulator.estimate_spread(seeds)
    

    print(f"Post-Simulation Results after {num_timestep} time steps")
    print(f"{num_active_nodes} nodes activated")
    print(f"Percent Active: {(num_active_nodes/graph.get_vertices().size)*100:.2f}%")