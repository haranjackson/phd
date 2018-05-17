#include "../src/etc/types.h"
#include "../src/options.h"

#include "../include/pybind11/stl.h"
#include "../include/pybind11/stl_bind.h"

PYBIND11_MAKE_OPAQUE(std::vector<Vec>);
PYBIND11_MAKE_OPAQUE(std::vector<bVec>);

#include "../include/pybind11/eigen.h"
#include "../include/pybind11/functional.h"
#include "../include/pybind11/pybind11.h"

#include "../src/etc/globals.h"
#include "../src/etc/grid.h"

#include "../src/multi/fill.h"
#include "../src/multi/pfmm.h"

#include "../src/scipy/lgmres.h"
#include "../src/scipy/poly.h"

#include "../src/solvers/basis.h"
#include "../src/solvers/iterator.h"
#include "../src/solvers/steppers.h"

#include "../src/solvers/weno/weno.h"
#include "../src/solvers/weno/weno_matrices.h"

#include "../src/solvers/dg/dg.h"
#include "../src/solvers/dg/dg_matrices.h"

#include "../src/solvers/split/homogeneous.h"
#include "../src/solvers/split/ode.h"

#include "../src/solvers/fv/fluxes.h"
#include "../src/solvers/fv/fv.h"

#include "../src/system/analytic.h"
#include "../src/system/eig.h"
#include "../src/system/equations.h"
#include "../src/system/jacobians.h"

#include "../src/system/multi/eigenvecs.h"
#include "../src/system/multi/riemann.h"

#include "../src/system/variables/derivatives.h"

namespace py = pybind11;

PYBIND11_MODULE(GPRpy, m) {
  m.doc() = "Python bindings to the GPRcpp library";

  py::bind_vector<std::vector<Vec>>(m, "VectorVec");
  py::bind_vector<std::vector<bVec>>(m, "VectorbVec");

  m.def("N", []() { return N; });
  m.def("NV", []() { return V; });
  m.def("VISCOUS", []() { return VISCOUS; });
  m.def("THERMAL", []() { return THERMAL; });
  m.def("REACTIVE", []() { return REACTIVE; });
  m.def("MULTI", []() { return MULTI; });
  m.def("LSET", []() { return LSET; });

  pybind11::module m_classes =
      m.def_submodule("classes", "Classes used by GPRpy");

  pybind11::module m_multi =
      m.def_submodule("multi", "Generic multimaterial functions");

  pybind11::module m_scipy = m.def_submodule("scipy", "SciPy functions");

  pybind11::module m_system =
      m.def_submodule("system", "System vectors and matrices");
  pybind11::module m_system_multi =
      m_system.def_submodule("multi", "Functions related to multimaterial GPR");

  pybind11::module m_solvers = m.def_submodule("solvers", "Solver functions");
  pybind11::module m_solvers_common =
      m_solvers.def_submodule("common", "Functions common to all solvers");
  pybind11::module m_solvers_weno =
      m_solvers.def_submodule("weno", "WENO functions");
  pybind11::module m_solvers_dg = m_solvers.def_submodule("dg", "DG functions");
  pybind11::module m_solvers_split =
      m_solvers.def_submodule("split", "Operator splitting functions");
  pybind11::module m_solvers_fv = m_solvers.def_submodule("fv", "FV functions");

  pybind11::class_<Par>(m_classes, "Par")
      .def(py::init<>())
      .def_readwrite("Rc", &Par::Rc)
      .def_readwrite("EOS", &Par::EOS)
      .def_readwrite("ρ0", &Par::ρ0)
      .def_readwrite("p0", &Par::p0)
      .def_readwrite("T0", &Par::T0)
      .def_readwrite("Tref", &Par::Tref)
      .def_readwrite("cv", &Par::cv)
      .def_readwrite("pINF", &Par::pINF)
      .def_readwrite("Γ0", &Par::Γ0)
      .def_readwrite("c02", &Par::c02)
      .def_readwrite("s", &Par::s)
      .def_readwrite("e0", &Par::e0)
      .def_readwrite("α", &Par::α)
      .def_readwrite("β", &Par::β)
      .def_readwrite("γ", &Par::γ)
      .def_readwrite("A", &Par::A)
      .def_readwrite("B", &Par::B)
      .def_readwrite("R1", &Par::R1)
      .def_readwrite("R2", &Par::R2)
      .def_readwrite("B0", &Par::B0)
      .def_readwrite("τ1", &Par::τ1)
      .def_readwrite("PLASTIC", &Par::PLASTIC)
      .def_readwrite("σY", &Par::σY)
      .def_readwrite("n", &Par::n)
      .def_readwrite("cα2", &Par::cα2)
      .def_readwrite("τ2", &Par::τ2)
      .def_readwrite("Qc", &Par::Qc)
      .def_readwrite("δp", &Par::δp);

  pybind11::class_<poly>(m_classes, "poly")
      .def(pybind11::init<Vec>())
      .def_readwrite("coef", &poly::coef)
      .def("intt", &poly::intt)
      .def("diff", &poly::diff)
      .def("eval", &poly::eval);

  m_multi.def("find_interface_cells", &find_interface_cells);
  m_multi.def("fill_boundary_cells", &fill_boundary_cells);
  m_multi.def("fill_neighbor_cells", &fill_neighbor_cells);
  m_multi.def("fill_ghost_cells", &fill_ghost_cells);

  m_scipy.def("lgmres_wrapper", &lgmres_wrapper, py::arg("A"), py::arg("b"),
              py::arg("x0") = Vec(0), py::arg("M") = Mat(0, 0),
              py::arg("tol") = 1e-5, py::arg("maxiter") = 1000,
              py::arg("inner_m") = 30, py::arg("outer_k") = 3,
              py::arg("outer_v") = std::vector<Vec>());
  m_scipy.def("newton_krylov", &nonlin_solve, py::arg("F"), py::arg("x"),
              py::arg("f_tol") = pow(mEPS, 1. / 3), py::arg("f_rtol") = INF,
              py::arg("x_tol") = INF, py::arg("x_rtol") = INF);

  m_system.def("flux", &flux);
  m_system.def("source", &source);
  m_system.def("block", &block);
  m_system.def("Bdot", &Bdot);
  m_system.def("system_matrix", &system_matrix);

  m_system.def("max_abs_eigs", &max_abs_eigs);
  m_system.def("thermo_acoustic_tensor", &thermo_acoustic_tensor);

  m_system.def("dFdP", &dFdP);
  m_system.def("dPdQ", &dPdQ);

  m_system_multi.def("eigen", &eigen);
  m_system_multi.def("riemann_constraints", &riemann_constraints);
  m_system_multi.def("star_stepper", &star_stepper);
  m_system_multi.def("star_states", &star_states);

  m_solvers_common.def("basis_polys", &basis_polys);
  m_solvers_common.def("scaled_nodes", &scaled_nodes);
  m_solvers_common.def("scaled_weights", &scaled_weights);
  m_solvers_common.def("end_values", &end_values);
  m_solvers_common.def("derivative_values", &derivative_values);

  m_solvers.def("extended_dimensions", &extended_dimensions);
  m_solvers.def("ader_stepper", &ader_stepper);
  m_solvers.def("split_stepper", &split_stepper);
  m_solvers.def("iterator", &iterator);

  m_solvers_weno.def("coefficient_matrices", &coefficient_matrices);
  m_solvers_weno.def("oscillation_indicator", &oscillation_indicator);
  m_solvers_weno.def("weno_launcher", &weno_launcher);

  m_solvers_dg.def("rhs1", &rhs1);
  m_solvers_dg.def("rhs2", &rhs2);
  m_solvers_dg.def("obj1", &obj1);
  m_solvers_dg.def("obj2", &obj2);
  m_solvers_dg.def("predictor", &predictor);

  m_solvers_split.def("midstepper", &midstepper);
  m_solvers_split.def("analyticSolver_distortion", &analyticSolver_distortion);
  m_solvers_split.def("analyticSolver_thermal", &analyticSolver_thermal);
  m_solvers_split.def("ode_launcher", &ode_launcher);

  m_solvers_fv.def("Bint", &Bint);
  m_solvers_fv.def("D_RUS", &D_RUS);
  m_solvers_fv.def("D_ROE", &D_ROE);
  m_solvers_fv.def("D_OSH", &D_OSH);

  m_solvers_fv.def("centers1", &centers1);
  m_solvers_fv.def("centers2", &centers2);
  m_solvers_fv.def("interfs1", &interfs1);
  m_solvers_fv.def("interfs2", &interfs2);
  m_solvers_fv.def("fv_launcher", &fv_launcher);
}
