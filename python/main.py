from numpy import array

from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig
from gpr.tests.one import fluids, solids, multi as multi1, toro
from gpr.tests.two import validation, louisa, multi as multi2
from gpr.misc.plot import *

from solver.gfm import MultiSolver


# u, MPs, tf, dX = fluids.first_stokes_problem_IC()
# u, MPs, tf, dX = multi1.heat_conduction_multi_IC()
# u, MPs, tf, dX = multi1.helium_bubble_IC()
# u, MPs, tf, dX = multi1.air_copper_IC()
# u, MPs, tf, dX = multi1.water_air_IC()
# u, MPs, tf, dX = multi1.aluminium_vacuum_IC()
# u, MPs, tf, dX = solids.piston_IC()
u, MPs, tf, dX = louisa.aluminium_plate_impact_IC()

BC = 'transitive'
# BC = solids.piston_BC


CPP_LVL = 2
N = 3
CFL = 0.5
SPLIT = True


if CPP_LVL > 0:
    import GPRpy
    from gpr.opts import NV
    assert(GPRpy.NV() == NV)
    if CPP_LVL == 1:
        assert(GPRpy.N() == N)


nvar = u.shape[-1]
ndim = u.ndim - 1
dX = array(dX)


grids = []
def callback(u, t, count):
    grids.append(u.copy())


solver = MultiSolver(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                     model_params=MPs, M=M_cons, max_eig=max_eig, order=N,
                     ncore=1, split=SPLIT, ode_solver=None,
                     riemann_solver='rusanov')

solver.solve(u, tf, dX, cfl=CFL, boundary_conditions=BC, verbose=True,
             callback=callback, cpp_level=CPP_LVL)
