//
// Created by anish on 6/2/25.
//

#ifndef PROPAGATION_ALGORITHMS_H
#define PROPAGATION_ALGORITHMS_H

#include <cstdint>
#include <random>
#include <boost/graph/graph_traits.hpp>
#include "properties.h"

using namespace boost;

template <typename G>
class propagation_algorithm {
public:
    virtual ~propagation_algorithm() = default;
    virtual void propagate(const typename graph_traits<G>::vertex_descriptor& u) = 0;
};

template <typename G>
class independent_cascade_algorithm final : public propagation_algorithm<G> {
private:
    G& graph;
    uint8_t probability;
    std::mt19937 rng = std::mt19937(RANDOM_SEED);
    std::uniform_int_distribution<uint8_t> dist = std::uniform_int_distribution<uint8_t>(1, 100);

public:

    independent_cascade_algorithm(G& graph, uint8_t probability);

    void propagate(const typename graph_traits<G>::vertex_descriptor& u) override;
};

#endif