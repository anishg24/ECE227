//
// Created by anish on 6/2/25.
//

#ifndef PROPAGATION_ALGORITHMS_H
#define PROPAGATION_ALGORITHMS_H

#include <cstdint>
#include <vector>
#include <boost/graph/graph_traits.hpp>
#include "properties.h"

using namespace boost;

template <typename G>
class propagation_algorithm {
public:
    virtual ~propagation_algorithm() = default;
    virtual std::set<typename graph_traits<G>::vertex_descriptor> propagate(
            typename graph_traits<G>::vertex_descriptor& u,
            std::pair<typename graph_traits<G>::adjacency_iterator, typename graph_traits<G>::adjacency_iterator> neighbors
        ) = 0;
};

template <typename G>
class independent_cascade_algorithm final : public propagation_algorithm<G> {
private:
    G& graph;
    uint8_t probability;

public:

    independent_cascade_algorithm(G& graph, uint8_t probability);

    std::set<typename graph_traits<G>::vertex_descriptor> propagate(
            typename graph_traits<G>::vertex_descriptor& u,
            std::pair<typename graph_traits<G>::adjacency_iterator, typename graph_traits<G>::adjacency_iterator> neighbors
    ) override;
};

#endif