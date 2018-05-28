from numpy import array, save

from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig
from gpr.tests.one import fluids, solids, multi as multi1, toro
from gpr.tests.two import validation, impact, multi as multi2
from gpr.misc.plot import *

from solver.gfm import MultiSolver


# u0, MPs, tf, dX = fluids.first_stokes_problem_IC()
# u0, MPs, tf, dX = multi1.heat_conduction_multi_IC()
# u0, MPs, tf, dX = multi1.water_air_IC()
# u0, MPs, tf, dX = multi1.helium_bubble_IC()
# u0, MPs, tf, dX = multi1.pbx_copper_IC()
# u0, MPs, tf, dX = multi1.aluminium_vacuum_IC()
# u0, MPs, tf, dX = solids.piston_IC()
u0, MPs, tf, dX = impact.aluminium_plates_IC()

bcs = 'transitive'
# bcs = solids.piston_BC


cpp_level = 2
N = 2
cfl = 0.3
SPLIT = True
SOLVER = 'roe'


if cpp_level > 0:
    import GPRpy
    from gpr.opts import NV
    assert(GPRpy.NV() == NV)
    if cpp_level == 1:
        assert(GPRpy.N() == N)


nvar = u0.shape[-1]
ndim = u0.ndim - 1
dX = array(dX)
verbose = True


uList = []
gridList = []
maskList = []
def callback(u, grids, masks):
    uList.append(u.copy())
    gridList.append(deepcopy(grids))
    maskList.append(deepcopy(masks))

solver = MultiSolver(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                     model_params=MPs, M=M_cons, max_eig=max_eig, order=N,
                     ncore=1, split=SPLIT, ode_solver=None,
                     riemann_solver=SOLVER)

solver.solve(u0, tf, dX, cfl=cfl, bcs=bcs, verbose=verbose, callback=callback,
             cpp_level=cpp_level)

save('results', array(uList))
