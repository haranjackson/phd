from copy import deepcopy

from numpy import array

from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig
from gpr.tests.one import newtonian, elastic, plastic, multimaterial, \
    non_newtonian as non_newtonian1
from gpr.tests.two import impact, non_newtonian as non_newtonian2
from gpr.misc.plot import *

from solver.gfm import MultiSolver


u0, MPs, tf, dX = non_newtonian1.poiseuille_IC()

#bcs = 'transitive'
bcs = 'stick'


cpp_level = 2
N = 3
cfl = 0.5
SPLIT = True
SOLVER = 'rusanov'


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
    # TODO: add distortion resetting
    uList.append(u.copy())
    gridList.append(deepcopy(grids))
    maskList.append(deepcopy(masks))


solver = MultiSolver(nvar, ndim, F=F_cons, B=B_cons, S=S_cons,
                     model_params=MPs, M=M_cons, max_eig=max_eig, order=N,
                     ncore=1, split=SPLIT, ode_solver=None,
                     riemann_solver=SOLVER)

solver.solve(u0, tf, dX, cfl=cfl, bcs=bcs, verbose=verbose, callback=callback,
             cpp_level=cpp_level)
