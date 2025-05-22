from influence_analysis.graphs import GraphGenerator
from influence_analysis.influence_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    graph = GraphGenerator.get_social_graph()
    # print(graph.nodes['214328887']['active'])

    GA_params = {
        "POP_SIZE": 8,  # population size
        "GENERATIONS": 100,  # number of generations
        "ELITE_COUNT": 2,  # number of elites
        "TOUR_SIZE": 4,  # tournament size
        "MUT_RATE": 0.2,  # mutation rate
        "N_SIM": 100,  # number of simulations when evaluating fitness
        "IC_PROB": 0.05,  # default edge activation probability
    }
    ga = GeneticAlgorithm(graph, 10, GA_params)
    ga.get_seed_nodes()