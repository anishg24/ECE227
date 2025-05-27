#include <iostream>
#include <vector>
#include <tuple>
#include "ic.cpp"

int main() {
    // Create a small test graph
    // 0 -> 1 -> 2
    //      |    ^
    //      v    |
    //      3 ---+
    std::vector<std::tuple<int, int>> edges = {
        {0, 1},
        {1, 2},
        {1, 3},
        {3, 2}
    };
    Graph g(4, edges);

    // Test case 1: Start from node 0
    std::vector<int> seeds1 = {0};
    std::cout << "Test 1 - Starting from node 0:" << std::endl;
    for (int i = 0; i < 5; i++) {
        int spread = independent_cascade(g, seeds1, 0.5);
        std::cout << "Run " << i + 1 << ": " << spread << " nodes activated" << std::endl;
    }

    // Test case 2: Start from node 1
    std::vector<int> seeds2 = {1};
    std::cout << "\nTest 2 - Starting from node 1:" << std::endl;
    for (int i = 0; i < 5; i++) {
        int spread = independent_cascade(g, seeds2, 0.5);
        std::cout << "Run " << i + 1 << ": " << spread << " nodes activated" << std::endl;
    }

    // Test case 3: Start from multiple nodes
    std::vector<int> seeds3 = {0, 3};
    std::cout << "\nTest 3 - Starting from nodes 0 and 3:" << std::endl;
    for (int i = 0; i < 5; i++) {
        int spread = independent_cascade(g, seeds3, 0.5);
        std::cout << "Run " << i + 1 << ": " << spread << " nodes activated" << std::endl;
    }

    return 0;
}
