//
// Created by anish on 6/2/25.
//

#ifndef PROPERTIES_H
#define PROPERTIES_H

#include <boost/filesystem/fstream.hpp>
#include <boost/graph/adjacency_list.hpp>

#define VERTEX_TYPE uint32_t

using namespace boost;

struct SIProperty {
    VERTEX_TYPE original_id;
    bool active;
    bool already_spread;
};

typedef adjacency_list<vecS, vecS, undirectedS, SIProperty> UndirectedGraph;
typedef adjacency_list<vecS, vecS, directedS, SIProperty> DirectedGraph;

#endif //PROPERTIES_H
