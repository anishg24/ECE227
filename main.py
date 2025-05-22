from influence_analysis.graphs import GraphGenerator
from influence_analysis.influence_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    graph = GraphGenerator.get_social_graph()
    # print(graph.nodes['214328887']['active'])
    ga = GeneticAlgorithm(graph, 10)
    print(ga.get_seed_nodes())