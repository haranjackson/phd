from gpr.systems.conserved import F_cons, B_cons_lset, S_cons, M_cons
from gpr.systems.eigenvalues import max_eig
from gpr.tests.one import fluids, solids, multi, toro
from gpr.tests.two import validation
from gpr.misc.plot import *
from solver.solver import SolverPlus
from solver.gfm import MultiSolver


if __name__ == "__main__":

    # initial_grid, material_params, final_time, dX = fluids.heat_conduction_IC()
    initial_grid, material_params, final_time, dX = multi.heat_conduction_IC()

    nvar = initial_grid.shape[-1]
    ndim = initial_grid.ndim - 1

    m = len(material_params)

    def B_cons(Q, d, MP):
        return B_cons_lset(Q, d, MP, m-1)

    grids = []

    def callback(u, t, count):
        grids.append(u.copy())

    if m == 1:

        solver = SolverPlus(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                            model_params=material_params[0], M=M_cons,
                            max_eig=max_eig, order=2, ncore=1, split=False,
                            ode_solver=None, riemann_solver='rusanov')

        solver.solve(initial_grid, final_time, dX, cfl=0.6,
                     boundary_conditions='transitive', verbose=True,
                     callback=callback, cpp_level=0)

    else:

        solver = MultiSolver(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                             model_params=material_params, M=M_cons,
                             max_eig=max_eig, order=2, ncore=2, split=True,
                             ode_solver=None, riemann_solver='rusanov')

        solver.solve(initial_grid, final_time, dX, cfl=0.9,
                     boundary_conditions='transitive', verbose=True,
                     callback=callback, cpp_level=0)
