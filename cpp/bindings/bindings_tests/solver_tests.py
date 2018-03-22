import GPRpy

from numpy import array, dot, int32, zeros
from numpy.random import rand
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import lgmres

from solvers.basis import GAPS
from solvers.dg.dg import DG_W, DG_U, predictor, rhs
from solvers.dg.initial_guess import standard_initial_guess, stiff_initial_guess
from solvers.split.homogeneous import weno_midstepper
from models.gpr.systems.analytical import ode_solver_analytical
from solvers.weno.weno import weno_launcher

from bindings_tests.test_functions import check, generate_vector

from options import NDIM, NV, N
from options import STIFF, STIFF_IG, DG_TOL


### NEWTON-KRYLOV ###


def lgmres_test():
    A = rand(30, 30)
    b = rand(30)
    lgmres_cp = GPRpy.scipy.lgmres_wrapper(A, b)
    lgmres_py = lgmres(A, b)[0]
    print("LGMRES", check(lgmres_cp, lgmres_py))
    return lgmres_cp, lgmres_py


def newton_krylov_test(u, dt, dX, MP):

    NT = N**(NDIM + 1)
    nx, ny = u.shape[:2]
    wh = weno_launcher(u)

    if NDIM == 1:
        w = wh[int(nx / 2)].reshape([N**NDIM, NV])
    elif NDIM == 2:
        w = wh[int(nx / 2), int(ny / 2)].reshape([N**NDIM, NV])
    Ww = dot(DG_W, w)

    if STIFF_IG:
        dtGAPS = dt * GAPS
        q = stiff_initial_guess(w, dtGAPS, dX, MP)
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

    uBCpy = rand(nx + 2 * N,
                 ny + 2 * N * (NDIM > 1),
                 nz + 2 * N * (NDIM > 2),
                 NV)
    uBCcp = uBCpy.ravel()

    wh_py = weno_launcher(uBCpy)
    wh_cp = zeros((nx + 2) * (ny + 2 * (NDIM > 1)) *
                  (nz + 2 * (NDIM > 2)) * N**NDIM * NV)
    GPRpy.solvers.weno.weno_launcher(wh_cp, uBCcp, NDIM,
                                     array([nx, ny, nz], dtype=int32))
    wh_cp = wh_cp.reshape(wh_py.shape)

    print("WENO  ", check(wh_cp, wh_py))
    return wh_cp, wh_py


### DISCONTINUOUS GALERKIN ###


def rhs_test(u, dX, dt, MP):

    wh = weno_launcher(u)
    if NDIM == 1:
        Q = wh[0]
    elif NDIM == 2:
        Q = wh[0, 0]

    Q_py = array([Q] * N).reshape([N**(NDIM + 1), NV])
    Q_cp = Q_py[:, :NV]

    w = Q.reshape([N**NDIM, NV])
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

    nx, ny = u.shape[:2]
    wh = weno_launcher(u)

    if NDIM == 1:
        Q = wh[int(nx / 2)]
    elif NDIM == 2:
        Q = wh[int(nx / 2), int(ny / 2)]

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
    GPRpy.solvers.dg.predictor(qh_cp, wh_cp, NDIM, dt, dX, STIFF, STIFF_IG, MP)
    qh_cp = qh_cp.reshape(qh_py.shape)

    print("DG    ", check(qh_cp, qh_py))
    return qh_cp, qh_py


### SPLIT (NDIM=1) ###


def midstepper_test(u, dX, dt, MP):

    mid_py = weno_launcher(u)
    mid_cp = mid_py.ravel()

    weno_midstepper(mid_py, dt, dX, MP)
    GPRpy.solvers.split.midstepper(mid_cp, NDIM, dt, dX, MP)

    mid_cp = mid_cp.reshape(mid_py.shape)
    print("Step  ", check(mid_cp, mid_py))
    return mid_cp, mid_py


def ode_test(dt, MP):
    ode_py = generate_vector(MP)
    ode_cp = ode_py.copy()
    GPRpy.solvers.split.ode_launcher(ode_cp, dt, MP)
    ode_solver_analytical(ode_py, dt, MP)

    print("ODEs  ", check(ode_cp, ode_py))
    return ode_cp, ode_py
