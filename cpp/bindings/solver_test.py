import GPRpy

from numpy import array, dot, zeros
from numpy.random import rand
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import lgmres

from gpr.misc.structures import Cvec_to_Pclass
from gpr.systems.eigenvalues import thermo_acoustic_tensor

from solvers.basis import GAPS
from solvers.dg.dg import DG_W, DG_U, predictor, rhs
from solvers.dg.dg import standard_initial_guess, hidalgo_initial_guess
from solvers.fv.fluxes import Bint, D_OSH, D_ROE, D_RUS
from solvers.fv.fv import interfaces, endpoints, extend_dimensions, fv_terms, centers
from solvers.split.homogeneous import weno_midstepper
from solvers.split.ode import ode_stepper_analytical
from solvers.weno.weno import weno_launcher

from test_functions import check, generate_vector

from options import ndim, nV, nx, ny, nz, N, NT
from options import SPLIT, STIFF, HIDALGO, FLUX, PERR_FROB, DG_TOL


### NEWTON-KRYLOV ###


def lgmres_test():
    A = rand(30, 30)
    b = rand(30)
    lgmres_cp = GPRpy.scipy.lgmres_wrapper(A, b)
    lgmres_py = lgmres(A, b)[0]
    print("LGMRES", check(lgmres_cp, lgmres_py))
    return lgmres_cp, lgmres_py


def newton_krylov_test(u, dt, MP):

    wh = weno_launcher(u)
    w = wh[int(nx / 2), 0, 0]
    HOMOGENEOUS = SPLIT
    Ww = dot(DG_W, w)
    dtGAPS = dt * GAPS
    if HIDALGO:
        q = hidalgo_initial_guess(w, dtGAPS, MP, HOMOGENEOUS)
    else:
        q = standard_initial_guess(w)

    def obj(X): return dot(DG_U, X) - rhs(X, Ww, dt, MP, HOMOGENEOUS)

    def obj_cp(X):
        X2 = X.reshape([NT, nV])
        ret = obj(X2)
        return ret.ravel()

    nk_cp = GPRpy.scipy.newton_krylov(obj_cp, q.copy().ravel(), f_tol=DG_TOL)
    nk_py = newton_krylov(obj, q, f_tol=DG_TOL).ravel()
    print("N-K   ", check(nk_cp, nk_py))
    return nk_cp, nk_py


### WENO ###


def weno_test():
    upy = rand(nx + 2, ny + 2 * (ndim > 1), nz + 2 * (ndim > 2), nV)
    ucp = upy.ravel()

    if ndim == 1:
        wh_py = weno_launcher(upy).ravel()
        wh_cp = zeros((nx + 2) * (ny) * (nz) * N * nV)
    elif ndim == 2:
        wh_py = weno_launcher(upy).ravel()
        wh_cp = zeros((nx + 2) * (ny + 2) * (nz) * N * N * nV)
    else:
        wh_py = weno_launcher(upy).ravel()
        wh_cp = zeros((nx + 2) * (ny + 2) * (nz + 2) * N * N * N * nV)

    GPRpy.solvers.weno.weno_launcher(wh_cp, ucp, ndim, nx, ny, nz)
    print("WENO  ", check(wh_cp, wh_py))
    return wh_cp, wh_py


### DISCONTINUOUS GALERKIN ###


def rhs_test(u, dx, dt, MP):
    wh_py = weno_launcher(u)
    Q = wh_py[100, 0, 0]
    Q_py = array([Q] * N).reshape([N * N, nV])
    Q_cp = Q_py[:, :nV]

    Ww_py = rand(N * N, nV)
    Ww_py[:, -1] = 0
    Ww_cp = Ww_py[:, :nV]

    rhs_py = rhs(Q_py, Ww_py, dt, MP, 0)
    rhs_cp = GPRpy.solvers.dg.rhs1(Q_cp, Ww_cp, dt, dx, MP)

    print("RHS   ", check(rhs_cp, rhs_py))
    return rhs_cp, rhs_py


def obj_test(u, dx, dt, MP):
    wh_py = weno_launcher(u)
    Q = wh_py[100, 0, 0]
    Q_py = array([Q] * N).reshape([N * N, nV])
    Q_cp = Q_py[:, :nV]

    Ww_py = rand(N * N, nV)
    Ww_py[:, -1] = 0
    Ww_cp = Ww_py[:, :nV]

    rhs_py = rhs(Q_py, Ww_py, dt, MP, 0)

    obj_cp = GPRpy.solvers.dg.obj1(Q_cp.ravel(), Ww_cp, dt, dx, MP)
    obj_cp = obj_cp.reshape([N * N, nV])
    obj_py = rhs_py - dot(DG_U, Q_py)

    print("obj   ", check(obj_cp, obj_py))
    return obj_cp, obj_py


def dg_test(u, dx, dt, MP):
    wh_py = weno_launcher(u)
    wh_cp = wh_py.ravel()

    qh_py = predictor(wh_py, dt, MP)

    qh_cp = zeros(len(wh_cp) * N)
    GPRpy.solvers.dg.predictor(qh_cp, wh_cp, ndim, dt, dx, dx, dx,
                               STIFF, HIDALGO, MP)
    qh_cp = qh_cp.reshape(qh_py.shape)

    print("DG    ", check(qh_cp, qh_py))
    return qh_cp, qh_py


### FLUXES ###


def TAT_test(d, MP):
    Q = generate_vector(MP)
    P = Cvec_to_Pclass(Q, MP)

    TAT_py = thermo_acoustic_tensor(P, d)
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

### FINITE VOLUME (ndim=1) ###


HOMOGENEOUS = SPLIT
TIME = not SPLIT
SOURCES = not SPLIT


def FVc_test(qh_py, dx, dt, MP):
    qh0 = extend_dimensions(qh_py)[0]
    FVc_py = dt / dx * centers(qh0, nx - 2, ny, nz, MP, HOMOGENEOUS)
    FVc_cp = zeros([(nx - 2) * nV])
    GPRpy.solvers.fv.centers1(FVc_cp, qh_py.ravel(), nx - 2, dt, dx,
                              SOURCES, TIME, MP)

    FVc_cp = FVc_cp.reshape([(nx - 2), nV])
    FVc_py = FVc_py.reshape([(nx - 2), nV])

    print("FVc   ", check(FVc_cp, FVc_py))
    return FVc_cp, FVc_py


def FVi_test(qh_py, dx, dt, MP):
    qh0 = extend_dimensions(qh_py)[0]
    qEnd = endpoints(qh0)
    FVi_py = -0.5 * dt / dx * interfaces(qEnd, MP)
    FVi_cp = zeros([(nx - 2) * nV])
    GPRpy.solvers.fv.interfs1(FVi_cp, qh_py.ravel(), nx - 2, dt, dx,
                              TIME, FLUX, PERR_FROB, MP)

    FVi_cp = FVi_cp.reshape([(nx - 2), nV])
    FVi_py = FVi_py.reshape([(nx - 2), nV])

    print("FVi   ", check(FVi_cp, FVi_py))
    return FVi_cp, FVi_py


def FV_test(qh_py, dx, dt, MP):
    FV_py = fv_terms(qh_py, dt, MP, HOMOGENEOUS)
    FV_cp = zeros([(nx - 2) * nV])
    GPRpy.solvers.fv.fv_launcher(FV_cp, qh_py.ravel(), 1, nx - 2, 1, 1, dt, dx,
                                 1, 1, SOURCES, TIME, FLUX, PERR_FROB, MP)

    FV_cp = FV_cp.reshape([(nx - 2), nV])
    FV_py = FV_py.reshape([(nx - 2), nV])

    print("FV    ", check(FV_cp, FV_py))
    return FV_cp, FV_py


### SPLIT (ndim=1) ###


def midstepper_test(u, dx, dt, MP):
    wh = weno_launcher(u)
    mid_py = wh.reshape([nx, 1, 1, N, nV])
    mid_cp = mid_py.ravel()

    weno_midstepper(mid_py, dt, MP)
    GPRpy.solvers.split.midstepper(mid_cp, 1, dt, dx, dx, dx, MP)
    mid_cp = mid_cp.reshape([nx, N, nV])
    mid_py = mid_py.reshape([nx, N, nV])

    print("Step  ", check(mid_cp, mid_py))
    return mid_cp, mid_py


def ode_test(dt, MP):
    ode_py = generate_vector(MP)
    ode_cp = ode_py.copy()
    u = zeros([1, 1, 1, nV])
    u[0] = ode_py
    GPRpy.solvers.split.ode_launcher(ode_cp, dt, MP)
    ode_stepper_analytical(u, dt, MP)
    ode_py = u[0, 0, 0]

    print("ODEs  ", check(ode_cp, ode_py))
    return ode_cp, ode_py