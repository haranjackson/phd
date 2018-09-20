import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from copy import deepcopy
from time import time

from numpy import array, save

from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig

from test.fluid.newtonian1 import heat_conduction, stokes, viscous_shock
from test.fluid.non_newtonian import poiseuille, poiseuille_bc, \
    lid_driven_cavity
from test.fluid.reactive import steady_znd, shock_detonation, \
    heating_deflagration
from test.impact.inert import aluminium_plates, rod_penetration
from test.multi.material import water_air, helium_bubble, pbx_copper, \
    aluminium_vacuum, heat_conduction_multi
from test.solid.elastic import barton, pure_elastic
from test.solid.plastic import piston, piston_bc, cylindrical_shock

from gpr.misc.plot import *

from solver.gfm import MultiSolver


ics = lid_driven_cavity
bcs = 'lid_driven'
#bcs = 'transitive'
#bcs = 'stick'
#bcs = 'symmetric'

cpp_level = 2
N = 3
cfl = 0.5
SPLIT = True
SOLVER = 'rusanov'
contorted_tol = 1


if cpp_level > 0:
    import GPRpy
    from gpr.opts import NV
    assert(GPRpy.NV() == NV)
    if cpp_level == 1:
        assert(GPRpy.N() == N)


u0, MPs, tf, dX = ics()
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

solver.solve(u0, tf, dX, contorted_tol, cfl=cfl, bcs=bcs, verbose=verbose,
             callback=callback, cpp_level=cpp_level)

save('results/' + ics.__name__ + str(int(time())), uList[-1])
