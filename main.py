from influence_analysis.graphs import GraphGenerator
from influence_analysis.influence_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    graph = GraphGenerator.get_collab_graph()
    print(f"Graph size - Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    GA_params = {
        "POP_SIZE": 20,  # population size
        "GENERATIONS": 100,  # number of generations
        "ELITE_COUNT": 3,  # number of elites
        "TOUR_SIZE": 5,  # tournament size
        "MUT_RATE": 0.1,  # mutation rate
        "N_SIM": 100,  # number of simulations when evaluating fitness
        "IC_PROB": 0.3,  # default edge activation probability
    }
    ga = GeneticAlgorithm(graph, 5, GA_params)
    best_chrom, best_fit = ga.get_seed_nodes()
    print(f"Best chromosome: {best_chrom}")
    print(f"Best fitness: {best_fit}")

