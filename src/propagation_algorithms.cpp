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
std::set<typename graph_traits<G>::vertex_descriptor> independent_cascade_algorithm<G>::propagate(
    typename graph_traits<G>::vertex_descriptor& u,
    std::pair<typename graph_traits<G>::adjacency_iterator, typename graph_traits<G>::adjacency_iterator> neighbors) {
    std::set<typename graph_traits<G>::vertex_descriptor> activated;
    typename graph_traits<G>::adjacency_iterator ni, n_end;
    for (tie(ni, n_end) = neighbors; ni != n_end; ++ni) {
        if (!this->graph[*ni].active) {
            bool random = rand() % 100; // NOLINT(*-msc50-cpp)
            if (random < this->probability) {
                this->graph[*ni].active = true;
                activated.insert(*ni);
            }
        }
    }
    return activated;
}

template class independent_cascade_algorithm<UndirectedGraph>;
template class independent_cascade_algorithm<DirectedGraph>;