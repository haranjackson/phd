from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig
from gpr.tests.one import fluids, solids, multi, toro
from gpr.tests.two import validation
from gpr.misc.plot import *

from solver import SolverPlus
from solver.gfm import MultiSolver


if __name__ == "__main__":

    # u, MPs, tf, dX = fluids.first_stokes_problem_IC()
    # u, MPs, tf, dX = multi.barton1_multi_IC()
    # u, MPs, tf, dX = multi.heat_conduction_multi_IC()
    # u, MPs, tf, dX = multi.helium_bubble_IC()
    # u, MPs, tf, dX = multi.water_water_IC()
    u, MPs, tf, dX = multi.water_gas_IC()
    # u, MPs, tf, dX = multi.gas_solid_IC()

    nvar = u.shape[-1]
    ndim = u.ndim - 1

    grids = []

    def callback(u, t, count):
        grids.append(u.copy())

    if len(MPs) == 1:

        solver = SolverPlus(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                            model_params=MPs[0], M=M_cons, max_eig=max_eig,
                            order=2, ncore=1, split=True, ode_solver=None,
                            riemann_solver='rusanov')

        solver.solve(u, tf, dX, cfl=0.6, boundary_conditions='transitive',
                     verbose=True, callback=callback, cpp_level=2)

    else:

        solver = MultiSolver(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                             model_params=MPs, M=M_cons, max_eig=max_eig,
                             order=2, ncore=1, split=True, ode_solver=None,
                             riemann_solver='rusanov')

        solver.solve(u, tf, dX, cfl=0.9, boundary_conditions='transitive',
                     verbose=True, callback=callback, cpp_level=0)
