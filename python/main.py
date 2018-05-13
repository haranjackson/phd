from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig
from gpr.tests.one import fluids, solids, multi as multi1, toro
from gpr.tests.two import validation, multi as multi2
from gpr.misc.plot import *

from solver import SolverPlus
from solver.gfm import MultiSolver


# u, MPs, tf, dX = fluids.first_stokes_problem_IC()
# u, MPs, tf, dX = multi1.heat_conduction_multi_IC()
# u, MPs, tf, dX = multi1.helium_bubble_IC()
# u, MPs, tf, dX = multi1.water_gas_IC()
# u, MPs, tf, dX = multi1.gas_solid_IC()
u, MPs, tf, dX = multi2.water_gas_IC()

CPP_LVL = 0
N = 3


if CPP_LVL > 0:
    import GPRpy
    from gpr.opts import NV
    assert(GPRpy.NV() == NV)
    if CPP_LVL == 1:
        assert(GPRpy.N() == N)


nvar = u.shape[-1]
ndim = u.ndim - 1


grids = []
def callback(u, t, count):
    grids.append(u.copy())


if len(MPs) == 1:

    solver = SolverPlus(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                        model_params=MPs[0], M=M_cons, max_eig=max_eig,
                        order=N, ncore=1, split=True, ode_solver=None,
                        riemann_solver='rusanov')

    solver.solve(u, tf, dX, cfl=0.5, boundary_conditions='transitive',
                 verbose=True, callback=callback, cpp_level=CPP_LVL)

else:

    solver = MultiSolver(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                         model_params=MPs, M=M_cons, max_eig=max_eig,
                         order=N, ncore=1, split=True, ode_solver=None,
                         riemann_solver='rusanov')

    solver.solve(u, tf, dX, cfl=0.5, boundary_conditions='transitive',
                 verbose=True, callback=callback, cpp_level=CPP_LVL)
