from gpr.systems.eigenvalues import max_eig
from gpr.tests.one import fluids, solids, multi, toro
from gpr.tests.two import validation
from gpr.misc.plot import *

from solver import SolverPlus
from solver.gfm import MultiSolver


if __name__ == "__main__":

    # u, MPs, tf, dX, sys = fluids.heat_conduction_IC()
    u, MPs, tf, dX, sys = multi.heat_conduction_IC()

    nvar = u.shape[-1]
    ndim = u.ndim - 1

    grids = []

    def callback(u, t, count):
        grids.append(u.copy())

    if len(MPs) == 1:

        solver = SolverPlus(nvar, ndim, F=sys.F, B=sys.B, S=sys.S,
                            model_params=MPs[0], M=sys.M, max_eig=max_eig,
                            order=2, ncore=1, split=True, ode_solver=None,
                            riemann_solver='rusanov')

        solver.solve(u, tf, dX, cfl=0.6, boundary_conditions='transitive',
                     verbose=True, callback=callback, cpp_level=1)

    else:

        solver = MultiSolver(nvar, ndim, F=sys.F, B=sys.B, S=sys.S,
                             model_params=MPs, M=sys.M, max_eig=max_eig,
                             order=2, ncore=1, split=False, ode_solver=None,
                             riemann_solver='rusanov')

        solver.solve(u, tf, dX, cfl=0.9, boundary_conditions='transitive',
                     verbose=True, callback=callback, cpp_level=0)
