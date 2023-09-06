#include "bddenv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libbddenv, m)
{
    py::class_<BDDEnv>(m, "BDDEnv")
        .def(py::init<>())
        .def("reset", &BDDEnv::reset)
        .def("compute_pareto_frontier", &BDDEnv::compute_pareto_frontier)
        .def_readwrite("status", &BDDEnv::status)
        .def_readwrite("initial_width", &BDDEnv::initial_width)
        .def_readwrite("initial_node_count", &BDDEnv::initial_node_count)
        .def_readwrite("initial_arcs_count", &BDDEnv::initial_arcs_count)
        .def_readwrite("reduced_width", &BDDEnv::reduced_width)
        .def_readwrite("reduced_node_count", &BDDEnv::reduced_node_count)
        .def_readwrite("reduced_arcs_count", &BDDEnv::reduced_arcs_count)
        .def_readwrite("initial_avg_in_degree", &BDDEnv::initial_avg_in_degree)
        .def_readwrite("reduced_avg_in_degree", &BDDEnv::reduced_avg_in_degree)
        .def_readwrite("nnds", &BDDEnv::nnds)
        .def_readwrite("num_comparisons", &BDDEnv::num_comparisons)
        .def_readwrite("num_pareto_sol_per_layer", &BDDEnv::num_pareto_sol_per_layer)
        .def_readwrite("x_sol", &BDDEnv::x_sol)
        .def_readwrite("z_sol", &BDDEnv::z_sol);
}