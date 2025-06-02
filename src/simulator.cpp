//
// Created by anish on 6/2/25.
//

#include "simulator.h"
#include "propagation_algorithms.h"

template <typename G>
simulator<G>::simulator(G& graph, propagation_algorithm<G>& propagation_algorithm) : graph(graph), prop_alg(propagation_algorithm) {}

template <typename G>
void simulator<G>::seed(typename graph_traits<G>::vertex_descriptor v) {
    this->graph[v].active = true;
}

template <typename G>
void simulator<G>::seed_nodes(std::vector<typename graph_traits<G>::vertex_descriptor> v_vec) {
    for (auto v : v_vec) {
        this->seed(v);
    }
}

template <typename G>
void simulator<G>::reset() {
    typename graph_traits<G>::vertex_iterator vi, v_end;
    for (tie(vi, v_end) = vertices(this->graph); vi != v_end; ++vi) {
        this->graph[*vi].active = false;
        this->graph[*vi].already_spread = false;
    }
}

template <typename G>
std::set<typename graph_traits<G>::vertex_descriptor> simulator<G>::get_active_vertices() {
    std::set<typename graph_traits<G>::vertex_descriptor> result;
    typename graph_traits<G>::vertex_iterator vi, v_end;
    for (tie(vi, v_end) = vertices(this->graph); vi != v_end; ++vi) {
        if (this->graph[*vi].active) {
            result.insert(*vi);
        }
    }
    return result;
}

template <typename G>
uint32_t simulator<G>::simulate(uint32_t n_iter) {
    auto active_nodes = this->get_active_vertices();
    for (uint32_t i = 0; i < n_iter; i++) {
        std::set<typename graph_traits<G>::vertex_descriptor> new_activations;
        // #pragma omp parallel for
        for (auto v : active_nodes) {
            if (!this->graph[v].already_spread) {
                new_activations = this->prop_alg.propagate(v, adjacent_vertices(v, this->graph));
                this->graph[v].already_spread = true;
            }
        }
        active_nodes.merge(new_activations);
        new_activations.clear();
    }

    return active_nodes.size();
}

template class simulator<UndirectedGraph>;
template class simulator<DirectedGraph>;