
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <vector>
#include <tuple>
#include <unordered_set>

namespace py = pybind11;

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
          const std::vector<std::tuple<int, int, py::object>>& edges)
        : adj_(num_nodes)
    {
        for (const auto& e : edges) {
            int    u = std::get<0>(e);
            int    v = std::get<1>(e);
            double w = std::get<2>(e).is_none()    ? -1.0
                      : std::get<2>(e).cast<double>();
            adj_[u].push_back({v, w});
        }
    }

    const std::vector<std::pair<int, double>>& neighbors(int u) const {
        return adj_[u];
    }

private:
    std::vector<std::vector<std::pair<int, double>>> adj_;
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
            for (const auto& [v, w] : G.neighbors(u)) {
                if (active.find(v) != active.end()) continue;

                double prob = (w >= 0.0) ? w : default_p;
                if (uni(rng) < prob) {
                    active.insert(v);
                    next.push_back(v);
                }
            }
        }
        frontier.swap(next);
    }
    return static_cast<int>(active.size());
}

/* ------------------------------------------------------------------ */
/*  pybind11 glue                                                     */
/* ------------------------------------------------------------------ */
PYBIND11_MODULE(fast_ic, m)
{
    m.doc() = "Fast Independent Cascade (C++/pybind11)";

    py::class_<Graph>(m, "Graph")
        .def(py::init<std::size_t,
                      const std::vector<std::tuple<int,int,py::object>>&>(),
             py::arg("num_nodes"),
             py::arg("edges"),
             R"pbdoc(
                 Lightweight adjacency-list graph.

                 Parameters
                 ----------
                 num_nodes : int
                     Total number of nodes (0…num_nodes-1).
                 edges : list[ (src, dst, weight | None) ]
                     Directed edges.  Pass `None` if you want the
                     default activation probability for that edge.
             )pbdoc");

    m.def("independent_cascade",
          &independent_cascade,
          py::arg("graph"),
          py::arg("seeds"),
          py::arg("default_p") = 0.1,
          R"pbdoc(
              One stochastic IC simulation.

              Parameters
              ----------
              graph : fast_ic.Graph
              seeds : list[int]
              default_p : float, optional
                  Used when an edge has `weight is None`.

              Returns
              -------
              int
                  Total number of active nodes at cascade end.
          )pbdoc");
}