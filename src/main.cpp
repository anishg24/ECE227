#include <iostream>
#include <chrono>

#include "propagation_algorithms.h"
#include "simulator.h"
#include "seed_algorithms.h"
#include "graph_generator.cpp"
#include "properties.h"

using namespace boost;

int main() {
    const auto file_path = filesystem::path("./data/collab.txt");
    UndirectedGraph graph;
    populate_graph(file_path, graph);
    independent_cascade_algorithm prop_alg(graph, 30);
    simulator sim(graph, prop_alg);
    greedy_algorithm alg(graph, sim, 5);

    const auto start = std::chrono::high_resolution_clock::now();
    alg.run(100);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    for (auto v : alg.get_seed_nodes()) {
        std::cout << v << std::endl;
    }

    sim.seed_nodes(alg.get_seed_nodes());
    // auto v = vertex(5, graph);
    // sim.seed(v);
    sim.simulate(100);

    std::cout << "Percent activated: " << (static_cast<float>(sim.get_active_vertices().size())/static_cast<float>(num_vertices(graph))) * 100 << "%" << std::endl;

    return 0;
}
