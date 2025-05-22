
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

#include <random>
#include <vector>
#include <tuple>
#include <unordered_set>

// namespace py = pybind11;

/* ------------------------------------------------------------------ */
/*  A very small adjacency-list graph                                 */
/* ------------------------------------------------------------------ */
class Graph {
public:
    /*
       edges : list[tuple(src, dst, weight | None)]
       num_nodes is needed so we can allocate the adjacency vector once,
       not search for the maximum id.
    */
    Graph(std::size_t num_nodes,
          const std::vector<std::tuple<int, int>>& edges)
        : adj_(num_nodes)
    {
        for (const auto& e : edges) {
            int    u = std::get<0>(e);
            int    v = std::get<1>(e);
            adj_[u].push_back({v});
        }
    }

    const std::vector<int>& neighbors(int u) const {
        return adj_[u];
    }

private:
    std::vector<std::vector<int>> adj_;
};


/* ------------------------------------------------------------------ */
/*  Independent Cascade (one run)                                     */
/* ------------------------------------------------------------------ */
int independent_cascade(const Graph&              G,
                         const std::vector<int>&   seeds,
                         double                    default_p = 0.1)
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::unordered_set<int> active(seeds.begin(), seeds.end());
    std::vector<int>        frontier(seeds.begin(), seeds.end());

    while (!frontier.empty()) {
        std::vector<int> next;
        next.reserve(frontier.size() * 2);

        for (int u : frontier) {
            for (const auto& v : G.neighbors(u)) {
                if (active.find(v) != active.end()) continue;

                if (uni(rng) < default_p) {
                    active.insert(v);
                    next.push_back(v);
                }
            }
        }
        frontier.swap(next);
    }
    return static_cast<int>(active.size());
}
