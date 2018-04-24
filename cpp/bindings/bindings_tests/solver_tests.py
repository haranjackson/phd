import GPRpy

from numpy import array, dot, int32, zeros
from numpy.random import rand
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import lgmres

from ader.dg.initial_guess import stiff_initial_guess

from gpr.sys.analytical import ode_solver_cons

from bindings_tests.test_functions import check, generate_vector, cpp_dx


""" NEWTON-KRYLOV """


def lgmres_test():
    A = rand(30, 30)
    b = rand(30)
    lgmres_cp = GPRpy.scipy.lgmres_wrapper(A, b)
    lgmres_py = lgmres(A, b)[0]
    print("LGMRES", check(lgmres_cp, lgmres_py))
    return lgmres_cp, lgmres_py


def newton_krylov_test(u, dt, dX, wenoSolver, dgSolver):

    N = wenoSolver.N
    NV = wenoSolver.NV
    NDIM = wenoSolver.NDIM

    NT = N**(NDIM + 1)
    nx, ny = u.shape[:2]

    wh = wenoSolver.solve(u)

    if NDIM == 1:
        w = wh[int(nx / 2)].reshape([N**NDIM, NV])
    elif NDIM == 2:
        w = wh[int(nx / 2), int(ny / 2)].reshape([N**NDIM, NV])
    Ww = dot(dgSolver.DG_W, w)

    q = dgSolver.initial_guess(dgSolver, w, dt, dX)

    def obj(X): return dot(dgSolver.DG_U, X) - dgSolver.rhs(X, Ww, dt, dX)

    def obj_cp(X):
        X2 = X.reshape([NT, NV])
        ret = obj(X2)
        return ret.ravel()

    nk_cp = GPRpy.scipy.newton_krylov(obj_cp, q.copy().ravel(),
                                      f_tol=dgSolver.tol)
    nk_py = newton_krylov(obj, q, f_tol=dgSolver.tol).ravel()
    print("N-K   ", check(nk_cp, nk_py))
    return nk_cp, nk_py


""" WENO """


def weno_test(wenoSolver):

    N = wenoSolver.N
    NV = wenoSolver.NV
    NDIM = wenoSolver.NDIM

    nx = 20
    ny = 20 if NDIM > 1 else 1
    nz = 20 if NDIM > 2 else 1

    uBCpy = rand(nx + 2 * N,
                 ny + 2 * N * (NDIM > 1),
                 nz + 2 * N * (NDIM > 2),
                 NV)
    uBCcp = uBCpy.ravel()

    wh_py = wenoSolver.solve(uBCpy)

    wh_cp = zeros((nx + 2) * (ny + 2 * (NDIM > 1)) *
                  (nz + 2 * (NDIM > 2)) * N**NDIM * NV)
    GPRpy.solvers.weno.weno_launcher(wh_cp, uBCcp, NDIM,
                                     array([nx, ny, nz], dtype=int32))
    wh_cp = wh_cp.reshape(wh_py.shape)

    print("WENO  ", check(wh_cp, wh_py))
    return wh_cp, wh_py


""" DISCONTINUOUS GALERKIN """


def rhs_test(u, dX, dt, wenoSolver, dgSolver):

    N = wenoSolver.N
    NV = wenoSolver.NV
    NDIM = wenoSolver.NDIM

    wh = wenoSolver.solve(u)

    if NDIM == 1:
        Q = wh[0]
    elif NDIM == 2:
        Q = wh[0, 0]

    Q_py = array([Q] * N).reshape([N**(NDIM + 1), NV])
    Q_cp = Q_py[:, :NV]

    w = Q.reshape([N**NDIM, NV])
    Ww_py = dot(dgSolver.DG_W, w)
    Ww_cp = Ww_py[:, :NV]

    rhs_py = dgSolver.rhs(Q_py, Ww_py, dt, dX)

    if NDIM == 1:
        rhs_cp = GPRpy.solvers.dg.rhs1(Q_cp, Ww_cp, dt, dX[0], dgSolver.pars)
    else:
        rhs_cp = GPRpy.solvers.dg.rhs2(Q_cp, Ww_cp, dt, dX[0], dX[1],
                                       dgSolver.pars)

    print("RHS   ", check(rhs_cp, rhs_py))
    return rhs_cp, rhs_py


def obj_test(u, dX, dt, wenoSolver, dgSolver):

    N = wenoSolver.N
    NV = wenoSolver.NV
    NDIM = wenoSolver.NDIM

    nx, ny = u.shape[:2]
    wh = wenoSolver.solve(u)

    if NDIM == 1:
        Q = wh[int(nx / 2)]
    elif NDIM == 2:
        Q = wh[int(nx / 2), int(ny / 2)]

    Q_py = array([Q] * N).reshape([N**(NDIM + 1), NV])
    Q_cp = Q_py[:, :NV]

    Ww_py = rand(N**(NDIM + 1), NV)
    Ww_py[:, -1] = 0
    Ww_cp = Ww_py[:, :NV]

    rhs_py = dgSolver.rhs(Q_py, Ww_py, dt, dX)

    if NDIM == 1:
        obj_cp = GPRpy.solvers.dg.obj1(Q_cp.ravel(), Ww_cp, dt, dX[0],
                                       dgSolver.pars)
    else:
        obj_cp = GPRpy.solvers.dg.obj2(
            Q_cp.ravel(), Ww_cp, dt, dX[0], dX[1], dgSolver.pars)

    obj_cp = obj_cp.reshape([N**(NDIM + 1), NV])
    obj_py = rhs_py - dot(dgSolver.DG_U, Q_py)

    print("obj   ", check(obj_cp, obj_py))
    return obj_cp, obj_py


def dg_test(u, dX, dt, wenoSolver, dgSolver):

    stiff_guess = dgSolver.initial_guess == stiff_initial_guess

    N = wenoSolver.N
    NDIM = wenoSolver.NDIM

    wh_py = wenoSolver.solve(u)
    wh_cp = wh_py.ravel()

    qh_py = dgSolver.solve(wh_py, dt, dX)

    qh_cp = zeros(len(wh_cp) * N)
    GPRpy.solvers.dg.predictor(qh_cp, wh_cp, NDIM, dt, cpp_dx(dX),
                               dgSolver.stiff, stiff_guess, dgSolver.pars)
    qh_cp = qh_cp.reshape(qh_py.shape)

    print("DG    ", check(qh_cp, qh_py))
    return qh_cp, qh_py


""" SPLIT (NDIM=1) """


def midstepper_test(u, dX, dt, wenoSolver, splitSolver):

    NDIM = wenoSolver.NDIM

    mid_py = wenoSolver.solve(u)
    mid_cp = mid_py.ravel()

    splitSolver.weno_midstepper(mid_py, dt, dX)
    GPRpy.solvers.split.midstepper(mid_cp, NDIM, dt, cpp_dx(dX),
                                   splitSolver.pars)

    mid_cp = mid_cp.reshape(mid_py.shape)
    print("Step  ", check(mid_cp, mid_py))
    return mid_cp, mid_py


def ode_test(dt, MP):
    ode_py = generate_vector(MP)
    ode_cp = ode_py.copy()
    GPRpy.solvers.split.ode_launcher(ode_cp, dt, MP)
    ode_solver_cons(ode_py, dt, MP)

    print("ODEs  ", check(ode_cp, ode_py))
    return ode_cp, ode_py
