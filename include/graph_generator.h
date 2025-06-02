//
// Created by anish on 6/2/25.
//

#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include <iostream>
#include <string>

#include "properties.h"

template <typename G>
typename graph_traits<G>::vertex_descriptor get_or_add_vertex(
    G &g,
    int id,
    std::map<VERTEX_TYPE, typename graph_traits<G>::vertex_descriptor> &vertex_map
    );

template <typename G>
bool populate_graph(const filesystem::path& data_file_path, G& g);

#endif //GRAPH_GENERATOR_H
