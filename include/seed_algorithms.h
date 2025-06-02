//
// Created by anish on 6/2/25.
//

#ifndef SEED_ALGORITHMS_H
#define SEED_ALGORITHMS_H

#include <cstdint>
#include <vector>
#include <boost/graph/graph_traits.hpp>
#include "simulator.h"
#include "properties.h"

using namespace boost;

template <typename G>
class greedy_algorithm {
private:
    G& graph;
    std::vector<typename graph_traits<G>::vertex_descriptor> seed_nodes;
    simulator<G> sim;
    uint32_t num_seed_nodes;

public:
    greedy_algorithm(G& g, simulator<G> sim, const uint32_t num_seed_nodes) : graph(g), seed_nodes({}), sim(sim), num_seed_nodes(num_seed_nodes) {}

    std::vector<typename graph_traits<G>::vertex_descriptor> get_seed_nodes();

    void run(uint32_t niter);
};



#endif //SEED_ALGORITHMS_H
