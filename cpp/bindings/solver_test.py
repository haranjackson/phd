import GPRpy

from numpy import array, dot, int32, zeros
from numpy.random import rand
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import lgmres

from gpr.misc.structures import Cvec_to_Pclass
from gpr.systems.eigenvalues import Xi1, Xi2

from solvers.basis import GAPS
from solvers.dg.dg import DG_W, DG_U, predictor, rhs
from solvers.dg.dg import standard_initial_guess, hidalgo_initial_guess
from solvers.fv.fluxes import Bint, D_OSH, D_ROE, D_RUS
from solvers.fv.fv import interfaces, endpoints, extend_dimensions, fv_terms, centers
from solvers.split.split import weno_midstepper
from solvers.split.analytical import ode_stepper_analytical
from solvers.weno.weno import weno_launcher

from test_functions import check, generate_vector

from options import NDIM, NV, N, NT
from options import SPLIT, STIFF, HIDALGO, FLUX, PERR_FROB, DG_TOL


### NEWTON-KRYLOV ###


def lgmres_test():
    A = rand(30, 30)
    b = rand(30)
    lgmres_cp = GPRpy.scipy.lgmres_wrapper(A, b)
    lgmres_py = lgmres(A, b)[0]
    print("LGMRES", check(lgmres_cp, lgmres_py))
    return lgmres_cp, lgmres_py


def newton_krylov_test(u, dt, dX, MP):

    nx, ny = u.shape[:2]
    wh = weno_launcher(u)
    w = wh[int(nx / 2), int(ny/2), 0].reshape([N**NDIM, NV])
    Ww = dot(DG_W, w)

    if HIDALGO:
        dtGAPS = dt * GAPS
        q = hidalgo_initial_guess(w, dtGAPS, MP)
    else:
        q = standard_initial_guess(w)

    def obj(X): return dot(DG_U, X) - rhs(X, Ww, dt, dX, MP)

    def obj_cp(X):
        X2 = X.reshape([NT, NV])
        ret = obj(X2)
        return ret.ravel()

    nk_cp = GPRpy.scipy.newton_krylov(obj_cp, q.copy().ravel(), f_tol=DG_TOL)
    nk_py = newton_krylov(obj, q, f_tol=DG_TOL).ravel()
    print("N-K   ", check(nk_cp, nk_py))
    return nk_cp, nk_py


### WENO ###


def weno_test():

    nx = 20
    ny = 20 if NDIM > 1 else 1
    nz = 20 if NDIM > 2 else 1

    upy = rand(nx + 2, ny + 2 * (NDIM > 1), nz + 2 * (NDIM > 2), NV)
    ucp = upy.ravel()

    wh_py = weno_launcher(upy).ravel()
    wh_cp = zeros((nx + 2) * (ny + 2 * (NDIM > 1))
                  * (nz + 2 * (NDIM > 2)) * N**NDIM * NV)
    GPRpy.solvers.weno.weno_launcher(wh_cp, ucp, NDIM,
                                     array([nx, ny, nz], dtype=int32))
    print("WENO  ", check(wh_cp, wh_py))
    return wh_cp, wh_py


### DISCONTINUOUS GALERKIN ###


def rhs_test(u, dX, dt, MP):
    wh_py = weno_launcher(u)
    Q = wh_py[0, 0, 0]
    Q_py = array([Q] * N).reshape([N**(NDIM + 1), NV])
    Q_cp = Q_py[:, :NV]

    w = wh_py[0, 0, 0].reshape([N**NDIM, NV])
    Ww_py = dot(DG_W, w)
    Ww_cp = Ww_py[:, :NV]

    rhs_py = rhs(Q_py, Ww_py, dt, dX, MP)

    if NDIM == 1:
        rhs_cp = GPRpy.solvers.dg.rhs1(Q_cp, Ww_cp, dt, dX[0], MP)
    else:
        rhs_cp = GPRpy.solvers.dg.rhs2(Q_cp, Ww_cp, dt, dX[0], dX[1], MP)

    print("RHS   ", check(rhs_cp, rhs_py))
    return rhs_cp, rhs_py


def obj_test(u, dX, dt, MP):
    nx = u.shape[0]
    wh_py = weno_launcher(u)
    Q = wh_py[int(nx / 2), 0, 0]
    Q_py = array([Q] * N).reshape([N**(NDIM + 1), NV])
    Q_cp = Q_py[:, :NV]

    Ww_py = rand(N**(NDIM + 1), NV)
    Ww_py[:, -1] = 0
    Ww_cp = Ww_py[:, :NV]

    rhs_py = rhs(Q_py, Ww_py, dt, dX, MP)

    if NDIM == 1:
        obj_cp = GPRpy.solvers.dg.obj1(Q_cp.ravel(), Ww_cp, dt, dX[0], MP)
    else:
        obj_cp = GPRpy.solvers.dg.obj2(
            Q_cp.ravel(), Ww_cp, dt, dX[0], dX[1], MP)

    obj_cp = obj_cp.reshape([N**(NDIM + 1), NV])
    obj_py = rhs_py - dot(DG_U, Q_py)

    print("obj   ", check(obj_cp, obj_py))
    return obj_cp, obj_py


def dg_test(u, dX, dt, MP):
    wh_py = weno_launcher(u)
    wh_cp = wh_py.ravel()

    qh_py = predictor(wh_py, dt, dX, MP)

    qh_cp = zeros(len(wh_cp) * N)
    GPRpy.solvers.dg.predictor(qh_cp, wh_cp, NDIM, dt, dX, STIFF, HIDALGO, MP)
    qh_cp = qh_cp.reshape(qh_py.shape)

    print("DG    ", check(qh_cp, qh_py))
    return qh_cp, qh_py


### FLUXES ###


def TAT_test(d, MP):
    Q = generate_vector(MP)
    P = Cvec_to_Pclass(Q, MP)

    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    TAT_py = dot(Ξ1, Ξ2)
    TAT_cp = GPRpy.system.thermo_acoustic_tensor(Q, d, MP)

    print("TAT   ", check(TAT_cp, TAT_py))
    return TAT_cp, TAT_py


def Bint_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    Bint_cp = GPRpy.solvers.fv.Bint(Q1, Q2, d, MP)
    Bint_py = Bint(Q1, Q2, d, MP)

    print("Bint  ", check(Bint_cp, Bint_py))
    return Bint_cp, Bint_py


def D_RUS_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    D_RUS_cp = GPRpy.solvers.fv.D_RUS(Q1, Q2, d, PERR_FROB, MP)
    D_RUS_py = -D_RUS(Q1, Q2, d, MP)

    print("D_RUS ", check(D_RUS_cp, D_RUS_py))
    return D_RUS_cp, D_RUS_py


def D_ROE_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    D_ROE_cp = GPRpy.solvers.fv.D_ROE(Q1, Q2, d, MP)
    D_ROE_py = -D_ROE(Q1, Q2, d, MP)

    print("D_ROE ", check(D_ROE_cp, D_ROE_py))
    return D_ROE_cp, D_ROE_py


def D_OSH_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    D_OSH_cp = GPRpy.solvers.fv.D_OSH(Q1, Q2, d, MP)
    D_OSH_py = -D_OSH(Q1, Q2, d, MP)

    print("D_OSH ", check(D_OSH_cp, D_OSH_py))
    return D_OSH_cp, D_OSH_py

### FINITE VOLUME (NDIM=1) ###


HOMOGENEOUS = SPLIT
TIME = not SPLIT
SOURCES = not SPLIT


def FVc_test(qh_py, dX, dt, MP):

    qh0 = extend_dimensions(qh_py)[0]
    nx, ny, nz = qh0.shape[:3]

    FVc_py = dt / dX[0] * \
        centers(qh0, nx - 2, ny - 2, nz - 2, dX, MP, HOMOGENEOUS)

    FVc_cp = zeros([(nx - 2) * (ny - 2) * NV])
    if NDIM == 1:
        GPRpy.solvers.fv.centers1(FVc_cp, qh_py.ravel(), nx - 2, dt, dX[0],
                                  SOURCES, TIME, MP)
    else:
        GPRpy.solvers.fv.centers2(FVc_cp, qh_py.ravel(), nx - 2, ny - 2, dt,
                                  dX[0], dX[1], SOURCES, TIME, MP)

    FVc_cp = FVc_cp.reshape([nx - 2, ny - 2, NV])
    FVc_py = FVc_py.reshape([nx - 2, ny - 2, NV])

    print("FVc   ", check(FVc_cp, FVc_py))
    return FVc_cp, FVc_py


def FVi_test(qh_py, dX, dt, MP):

    qh0 = extend_dimensions(qh_py)[0]
    nx, ny, nz = qh0.shape[:3]
    qEnd = endpoints(qh0)

    FVi_py = -0.5 * dt / dX[0] * interfaces(qEnd, MP)
    FVi_cp = zeros([(nx - 2) * (ny - 2) * NV])

    if NDIM == 1:
        GPRpy.solvers.fv.interfs1(FVi_cp, qh_py.ravel(), nx - 2, dt, dX[0],
                                  TIME, FLUX, PERR_FROB, MP)
    else:
        GPRpy.solvers.fv.interfs2(FVi_cp, qh_py.ravel(), nx - 2, ny - 2, dt, dX[0], dX[1],
                                  TIME, FLUX, PERR_FROB, MP)

    FVi_cp = FVi_cp.reshape([nx - 2, ny - 2, NV])
    FVi_py = FVi_py.reshape([nx - 2, ny - 2, NV])

    print("FVi   ", check(FVi_cp, FVi_py))
    return FVi_cp, FVi_py


def FV_test(qh_py, dX, dt, MP):

    nx, ny = qh_py.shape[:2]
    ny = max(ny, 3)
    FV_py = fv_terms(qh_py, dt, dX, MP, HOMOGENEOUS)
    FV_cp = zeros([(nx - 2) * (ny - 2) * NV])

    GPRpy.solvers.fv.fv_launcher(FV_cp, qh_py.ravel(), NDIM, array(
        [nx - 2, ny - 2, 1], dtype=int32), dt, dX, SOURCES, TIME, FLUX, PERR_FROB, MP)

    FV_cp = FV_cp.reshape([nx - 2, ny - 2, NV])
    FV_py = FV_py.reshape([nx - 2, ny - 2, NV])

    print("FV    ", check(FV_cp, FV_py))
    return FV_cp, FV_py


### SPLIT (NDIM=1) ###


def midstepper_test(u, dX, dt, MP):

    nx, ny = u.shape[:2]
    wh = weno_launcher(u)
    mid_py = wh.reshape([nx, ny, 1] + [N] * NDIM + [NV])
    mid_cp = mid_py.ravel()

    weno_midstepper(mid_py, dt, dX, MP)
    GPRpy.solvers.split.midstepper(mid_cp, NDIM, dt, dX, MP)
    mid_cp = mid_cp.reshape([nx, ny] + [N] * NDIM + [NV])
    mid_py = mid_py.reshape([nx, ny] + [N] * NDIM + [NV])

    print("Step  ", check(mid_cp, mid_py))
    return mid_cp, mid_py


def ode_test(dt, MP):
    ode_py = generate_vector(MP)
    ode_cp = ode_py.copy()
    u = zeros([1, 1, 1, NV])
    u[0] = ode_py
    GPRpy.solvers.split.ode_launcher(ode_cp, dt, MP)
    ode_stepper_analytical(u, dt, MP)
    ode_py = u[0, 0, 0]

    print("ODEs  ", check(ode_cp, ode_py))
    return ode_cp, ode_py
