//
// Created by anish on 6/2/25.
//

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "propagation_algorithms.h"

template <typename G>
class simulator {
private:
    G& graph;
    propagation_algorithm<G>& prop_alg;
public:
    simulator(G& graph, propagation_algorithm<G>& propagation_algorithm);

    void seed(typename graph_traits<G>::vertex_descriptor v);

    void seed_nodes(std::vector<typename graph_traits<G>::vertex_descriptor> v_vec);

    void reset();

    std::vector<typename graph_traits<G>::vertex_descriptor> get_active_vertices();

    uint32_t simulate(uint32_t n_iter);
};

#endif