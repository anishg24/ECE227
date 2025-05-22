// independent_cascade.cpp   (full, drop-in replacement)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <algorithm>
#include <cassert>

namespace py = pybind11;

/* ----------------------------------------------------------- */
/*  Small adjacency-list graph  (no edge weights)              */
/* ----------------------------------------------------------- */
class Graph {
public:
    /*  Variant 1 – let the ctor deduce `num_nodes`  */
    Graph(const std::vector<std::tuple<std::size_t,std::size_t>>& edges)
    {
        std::size_t max_id = 0;
        for (auto&& e : edges)
            max_id = std::max({max_id,
                               std::get<0>(e),
                               std::get<1>(e)});
        init(max_id + 1, edges);
    }

    /*  Variant 2 – caller gives num_nodes explicitly  */
    Graph(std::size_t num_nodes,
          const std::vector<std::tuple<std::size_t,std::size_t>>& edges)
    {
        init(num_nodes, edges);
    }

    const std::vector<std::size_t>& neighbors(std::size_t u) const
    {
        if (u >= adj_.size()) return empty_;          // tolerate bad seed
        return adj_[u];
    }

    std::size_t num_nodes() const { return adj_.size(); }

private:
    std::vector<std::vector<std::size_t>> adj_;
    static inline const std::vector<std::size_t> empty_{};

    void init(std::size_t num_nodes,
              const std::vector<std::tuple<std::size_t,std::size_t>>& edges)
    {
        adj_.assign(num_nodes, {});          // outer vector
        for (auto&& e : edges) {
            std::size_t u = std::get<0>(e);
            std::size_t v = std::get<1>(e);
            assert(u < num_nodes && v < num_nodes);   // debug-only
            adj_[u].push_back(v);
        }
    }
};

/* ----------------------------------------------------------- */
/*  Independent Cascade                                       */
/* ----------------------------------------------------------- */
std::size_t independent_cascade(const Graph&              G,
                                const std::vector<std::size_t>& seeds,
                                double default_p = 0.1)
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    const std::size_t N = G.num_nodes();
    std::vector<uint8_t> active_bitmap(N, 0);
    std::vector<std::size_t> frontier;

    frontier.reserve(seeds.size());
    for (auto s : seeds)
        if (s < N && !active_bitmap[s]) {   // ignore bad or duplicate seeds
            active_bitmap[s] = 1;
            frontier.push_back(s);
        }

    while (!frontier.empty()) {
        std::vector<std::size_t> next;
        next.reserve(frontier.size() * 2);

        for (std::size_t u : frontier) {
            for (std::size_t v : G.neighbors(u)) {
                if (active_bitmap[v]) continue;
                if (uni(rng) < default_p) {
                    active_bitmap[v] = 1;
                    next.push_back(v);
                }
            }
        }
        frontier.swap(next);
    }

    // count ones
    return std::count(active_bitmap.begin(), active_bitmap.end(), 1);
}

double evaluate_chromosome(const Graph&                       G,
                           const std::vector<std::size_t>&    seeds,
                           std::size_t                        n_sim     = 100,
                           double                             default_p = 0.1)
{
    std::size_t total = 0;
    for (std::size_t i = 0; i < n_sim; ++i)
        total += independent_cascade(G, seeds, default_p);

    return static_cast<double>(total) / static_cast<double>(n_sim);
}

/* ----------------------------------------------------------- */
/*  pybind11 glue                                              */
/* ----------------------------------------------------------- */
PYBIND11_MODULE(fast_ic, m)
{
    m.doc() = "Fast Independent Cascade (robust, no edge weights)";

    py::class_<Graph>(m, "Graph")
        .def(py::init<const std::vector<std::tuple<std::size_t,std::size_t>>&>(),
             py::arg("edges"),
             R"pbdoc(
                 Graph(edges)

                 Parameters
                 ----------
                 edges : list[ (src, dst) ]
                     Directed edges with 0-based node IDs.  The
                     constructor infers the number of nodes automatically.
             )pbdoc")
        .def(py::init<std::size_t,
                      const std::vector<std::tuple<std::size_t,std::size_t>>&>(),
             py::arg("num_nodes"),
             py::arg("edges"),
             R"pbdoc(
                 Graph(num_nodes, edges)

                 Use this variant if your node IDs are a dense 0…N-1
                 range but you want an explicit size.
             )pbdoc");

    m.def("independent_cascade",
          &independent_cascade,
          py::arg("graph"),
          py::arg("seeds"),
          py::arg("default_p") = 0.1);

    m.def("evaluate_chromosome",
          &evaluate_chromosome,
          py::arg("graph"),
          py::arg("seeds"),
          py::arg("n_sim")     = 100,
          py::arg("default_p") = 0.1,
          R"pbdoc(
              Average cascade size over `n_sim` repetitions.

              Parameters
              ----------
              graph : fast_ic.Graph
              seeds : list[int]
                  Seed set (chromosome) to evaluate.
              n_sim : int, optional
                  Number of stochastic simulations.  Default 100.
              default_p : float, optional
                  Activation probability for every edge.

              Returns
              -------
              float
                  Mean number of activated nodes.
          )pbdoc");
}
