//
// Created by anish on 6/2/25.
//

#include "properties.h"
#include "propagation_algorithms.h"

using namespace boost;

template <typename G>
independent_cascade_algorithm<G>::independent_cascade_algorithm(G& graph, uint8_t probability) : graph(graph) {
    if (probability < 0 || probability > 100) {
        throw std::invalid_argument("probability must be between 0 and 100");
    }

    this->probability = probability;
}

template <typename G>
void independent_cascade_algorithm<G>::propagate(const typename graph_traits<G>::vertex_descriptor& u) {
    typename graph_traits<G>::adjacency_iterator ni, n_end;
    for (tie(ni, n_end) = adjacent_vertices(u, this->graph); ni != n_end; ++ni) {
        if (!(this->graph[*ni].active || this->graph[*ni].propagated)) {
            uint32_t random = this->dist(this->rng);
            if (random <= this->probability) {
                this->graph[*ni].propagated = true;
            }
        }
    }
}

template class independent_cascade_algorithm<UndirectedGraph>;
template class independent_cascade_algorithm<DirectedGraph>;