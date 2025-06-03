//
// Created by anish on 6/2/25.
//

#include "seed_algorithms.h"
#include "indicators.hpp"

#include <iostream>

template <typename G>
std::vector<typename graph_traits<G>::vertex_descriptor> greedy_algorithm<G>::get_seed_nodes() {
    return this->seed_nodes;
}

template <typename G>
void greedy_algorithm<G>::run(uint32_t niter) {
    using namespace indicators;

    this->sim.reset();

    typename graph_traits<G>::vertex_iterator vi, v_end;
    std::vector<typename graph_traits<G>::vertex_descriptor> candidate_vertices;

    for (tie(vi, v_end) = vertices(this->graph); vi != v_end; ++vi) {
        candidate_vertices.push_back(*vi);
    }

    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Running Simulations"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::FontStyles{std::vector{FontStyle::bold}}
    };

    float step_size = 1/this->num_seed_nodes;
    for (int i = 0; i < this->num_seed_nodes; i++) {
        typename graph_traits<G>::vertex_descriptor best_vertex;
        uint32_t best_score = 0;
        // #pragma omp parallel for
        for (auto v : candidate_vertices) {
            this->sim.seed_nodes(this->seed_nodes);
            this->sim.seed(v);
            auto gain = this->sim.simulate(niter);
            this->sim.reset();
            if (gain >= best_score) {
                best_vertex = v;
                best_score = gain;
            }
        }
        this->seed_nodes.push_back(best_vertex);
        candidate_vertices.erase(std::remove(candidate_vertices.begin(), candidate_vertices.end(), best_vertex), candidate_vertices.end());
        bar.set_progress(i + step_size);
    }
}

template class greedy_algorithm<UndirectedGraph>;
template class greedy_algorithm<DirectedGraph>;
