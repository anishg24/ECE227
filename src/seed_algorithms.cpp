//
// Created by anish on 6/2/25.
//

#include "seed_algorithms.h"

#include <iostream>

template <typename G>
std::vector<typename graph_traits<G>::vertex_descriptor> greedy_algorithm<G>::get_seed_nodes() {
    return this->seed_nodes;
}

template <typename G>
void greedy_algorithm<G>::run(uint32_t niter) {
    this->sim.reset();

    typename graph_traits<G>::vertex_iterator vi, v_end;
    std::set<typename graph_traits<G>::vertex_descriptor> candidate_vertices;

    for (tie(vi, v_end) = vertices(this->graph); vi != v_end; ++vi) {
        candidate_vertices.insert(*vi);
    }

    for (int i = 0; i < this->num_seed_nodes; i++) {
        typename graph_traits<G>::vertex_descriptor best_vertex;
        uint32_t best_score = 0;
        for (auto v : candidate_vertices) {
            auto trial_seeds = this->seed_nodes;
            trial_seeds.push_back(v);
            this->sim.seed_nodes(trial_seeds);
            auto gain = this->sim.simulate(niter);
            this->sim.reset();
            if (gain >= best_score) {
                best_vertex = v;
                best_score = gain;
            }
        }
        this->seed_nodes.push_back(best_vertex);
        candidate_vertices.erase(best_vertex);
    }
}

template class greedy_algorithm<UndirectedGraph>;
template class greedy_algorithm<DirectedGraph>;
