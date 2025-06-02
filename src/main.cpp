#include <iostream>

#include "propagation_algorithms.h"
#include "simulator.h"
#include "seed_algorithms.h"
#include "graph_generator.cpp"
#include "properties.h"

using namespace boost;

int main() {
    srand(277); // NOLINT(*-msc51-cpp)

    const auto file_path = filesystem::path("./data/social.txt");
    UndirectedGraph graph;
    populate_graph(file_path, graph);
    independent_cascade_algorithm prop_alg(graph, 30);
    simulator sim(graph, prop_alg);
    greedy_algorithm alg(graph, sim, 5);

    alg.run(1000);

    for (auto v : alg.get_seed_nodes()) {
        std::cout << v << std::endl;
    }

    sim.seed_nodes(alg.get_seed_nodes());
    sim.simulate(100);

    std::cout << "Percent activated: " << (static_cast<float>(sim.get_active_vertices().size())/static_cast<float>(num_vertices(graph))) * 100 << "%" << std::endl;

    return 0;
}
