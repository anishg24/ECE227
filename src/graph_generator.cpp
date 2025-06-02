//
// Created by anish on 6/1/25.
//

#include "graph_generator.h"

using namespace boost;

template <typename G>
typename graph_traits<G>::vertex_descriptor get_or_add_vertex(
    G &g,
    int id,
    std::map<VERTEX_TYPE, typename graph_traits<G>::vertex_descriptor> &vertex_map
    ) {

    auto it = vertex_map.find(id);
    if (it != vertex_map.end()) {
        return it->second;
    }

    auto v = add_vertex(g);
    g[v].original_id = id;
    g[v].active = false;
    g[v].already_spread = false;

    vertex_map[id] = v;
    return v;
}

template <typename G>
bool populate_graph(const filesystem::path& data_file_path, G& g) {
    filesystem::ifstream file(data_file_path);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return false;
    }

    std::map<VERTEX_TYPE, typename graph_traits<G>::vertex_descriptor> vertex_map;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        VERTEX_TYPE from_node, to_node;
        if (!(iss >> from_node >> to_node)) {
            std::cerr << "Error parsing line: " << line << std::endl;
        }

        auto u = get_or_add_vertex(g, from_node, vertex_map);
        auto v = get_or_add_vertex(g, to_node, vertex_map);

        add_edge(u, v, g);
    }
    file.close();

    return true;
}


