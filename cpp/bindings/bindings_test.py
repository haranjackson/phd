# Some scripts to test the Python bindings for the C++ GPR implementation
# Make sure Git/GPR-cpp and Git/GPR/python are in PYTHONPATH
import GPRpy

from ader.dg.dg import DGSolver
from ader.fv.fv import FVSolver
from ader.weno.weno import WENOSolver

from gpr.opts import VISCOUS, THERMAL, REACTIVE, MULTI, LSET
from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons
from gpr.sys.eigenvalues import max_eig
from gpr.tests.one import fluids, multi
from gpr.tests.two import validation

from solver.split import SplitSolver

from bindings_tests.system_tests import F_test, S_test, B_test, M_test

from bindings_tests.solver_tests import lgmres_test, newton_krylov_test
from bindings_tests.solver_tests import weno_test, rhs_test, obj_test, dg_test
from bindings_tests.solver_tests import midstepper_test, ode_test

from bindings_tests.fv_tests import TAT_test, Bint_test, D_RUS_test, \
    D_ROE_test, D_OSH_test, FVc_test, FVi_test, FV_test


""" OPTIONS """

# IC = validation.hagen_poiseuille_IC
IC = multi.water_water_IC
d = 0
dt = 0.0001

""" /OPTIONS """


u, MPs, _, dX = IC()
NDIM = u.ndim - 1
NV = u.shape[-1]
MP = MPs[0]
dx = dX[0]


N = GPRpy.N()
print('N =', N)

assert(VISCOUS == GPRpy.VISCOUS() and THERMAL == GPRpy.THERMAL() and
       REACTIVE == GPRpy.REACTIVE() and MULTI == GPRpy.MULTI() and
       LSET == GPRpy.LSET())


wenoSolver = WENOSolver(N, NV, NDIM)

dgSolver = DGSolver(N, NV, NDIM, F=F_cons, S=S_cons, B=B_cons, M=M_cons,
                    pars=MP, stiff=False, stiff_guess=False,
                    newton_guess=False, tol=1e-6, max_iter=50, max_size=1e16)

splitSolver = SplitSolver(N, NV, NDIM, F=F_cons, S=S_cons, B=B_cons, M=M_cons,
                          dSdQ=None, ode_solver=None, model_params=MP)

fvSolver = FVSolver(N, NV, NDIM, F=F_cons, S=S_cons, B=B_cons, M=M_cons,
                    max_eig=max_eig, pars=MP, riemann_solver='rusanov',
                    time_rec=True)


F_cp, F_py = F_test(d, MP)
S_cp, S_py = S_test(d, MP)
Bx_cp, Bx_py = B_test(d, MP)
M_cp, M_py = M_test(d, MP)

lgmres_cp, lgmres_py = lgmres_test()
nk_cp, nk_py = newton_krylov_test(u, dt, dX, wenoSolver, dgSolver)

wh_cp, wh_py = weno_test(wenoSolver)

TAT_cp, TAT_py = TAT_test(d, MP)
Bint_cp, Bint_py = Bint_test(d, fvSolver)
D_RUS_cp, D_RUS_py = D_RUS_test(d, fvSolver)
D_ROE_cp, D_ROE_py = D_ROE_test(d, fvSolver)
D_OSH_cp, D_OSH_py = D_OSH_test(d, fvSolver)

mid_cp, mid_py = midstepper_test(u, dX, dt, wenoSolver, splitSolver)
ode_cp, ode_py = ode_test(dt, MP)
qh_py = wenoSolver.solve(u)

rhs_cp, rhs_py = rhs_test(u, dX, dt, wenoSolver, dgSolver)
obj_cp, obj_py = obj_test(u, dX, dt, wenoSolver, dgSolver)
qh_cp, qh_py = dg_test(u, dX, dt, wenoSolver, dgSolver)

FVc_cp, FVc_py = FVc_test(qh_py, dX, dt, fvSolver)
FVi_cp, FVi_py = FVi_test(qh_py, dX, dt, fvSolver)
FV_cp, FV_py = FV_test(qh_py, dX, dt, fvSolver)
