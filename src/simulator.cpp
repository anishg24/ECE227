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
std::vector<typename graph_traits<G>::vertex_descriptor> simulator<G>::get_active_vertices() {
    std::vector<typename graph_traits<G>::vertex_descriptor> result;
    typename graph_traits<G>::vertex_iterator vi, v_end;
    for (tie(vi, v_end) = vertices(this->graph); vi != v_end; ++vi) {
        if (this->graph[*vi].active) {
            result.push_back(*vi);
        }
    }
    return result;
}

template <typename G>
uint32_t simulator<G>::simulate(const uint32_t n_iter) {
    auto active_nodes = this->get_active_vertices();
    for (uint32_t i = 0; i < n_iter; i++) {
        // 1. Iterate through all active vertices
        // 2. Set vertices to "propagated" according to propagation algorithm
        // 3. Iterate through all propagated vertices and update them to active for next iteration

        // #pragma omp parallel for
        for (auto v : active_nodes) {
            if (!this->graph[v].already_spread) {
                this->prop_alg.propagate(v);
                this->graph[v].already_spread = true;
            }
        }

        typename graph_traits<G>::vertex_iterator vi, v_end;
        for (tie(vi, v_end) = vertices(this->graph); vi != v_end; ++vi) {
            if (this->graph[*vi].propagated) {
                this->graph[*vi].active = true;
                active_nodes.push_back(*vi);
            }
        }
    }

    std::sort(active_nodes.begin(), active_nodes.end());
    active_nodes.erase(std::unique(active_nodes.begin(), active_nodes.end()), active_nodes.end());

    return active_nodes.size();
}

template class simulator<UndirectedGraph>;
template class simulator<DirectedGraph>;