from influence_analysis.graphs import GraphGenerator


if __name__ == '__main__':
    graph = GraphGenerator.get_social_graph()
    print(graph.nodes['214328887']['active'])