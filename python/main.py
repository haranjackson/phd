from time import time

from numpy import array

from gpr.tests.one import fluids, solids
from gpr.tests.two import validation
from gpr.misc.plot import *
from solver.gfm import add_ghost_cells, make_u
from solver.solver import SolverPlus


""" GFM Options """

LSET = 0                    # Number of level sets
RGFM = 0                    # Use Riemann GFM


def run(solver, u, dX, t, tf, count, data, MPs):

    tStart = time()
    u = data[count].grid

    while t < tf:

        t0 = time()
        dt = timestep(u, count, t, tf, dX, *args)

        mats = array([u for i in range(m)])
        if RGFM:
            add_ghost_cells(mats, dt, *args)

        u = make_u(mats)

        if RGFM:
            # reinitialize level sets
            pass

        t += dt
        count += 1
        print('Iteration Time:', time() - t0, '\n')

    print('TOTAL RUNTIME:', time() - tStart)


if __name__ == "__main__":

    u, MPs, tf, dX = solids.elastic1_IC()
    nvar = u.shape[-1]
    ndim = u.ndim
    m = len(MPs)

    from gpr.systems.conserved import F_cons, B_cons, S_cons, M_cons
    from gpr.systems.eigenvalues import max_eig

    solver = SolverPlus(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                        model_params=MPs[0], M=M_cons, max_eig=max_eig,
                        order=2, ncore=1, split=False, ode_solver=None)

    solver.solve(u, tf, dX, cfl=0.9, boundary_conditions='transitive',
                 verbose=True, callback=None, cpp_level=0)
