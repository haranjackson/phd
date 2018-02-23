from itertools import product

from joblib import delayed
from numpy import absolute, array, concatenate, dot, zeros
from scipy.linalg import solve
from scipy.optimize import newton_krylov

from solvers.dg.matrices import DG_W, DG_U, DG_V, DG_Z, DG_T
from solvers.basis import GAPS, DERVALS
from system import source, flux_ref, source_ref, Bdot, system
from options import NDIM, N, NT, NV
from options import STIFF, STIFF_IG, HIDALGO, DG_TOL, DG_IT, PARA_DG, NCORE


MAX_TOL = 1e16  # values above this level will cause an error


def rhs(q, Ww, dt, dX, MP):
    """ Returns the right-handside of the system governing coefficients of qh
    """
    ret = zeros([NT, NV])

    Tq = dot(DG_T, q)
    Fq = zeros([NDIM, NT, NV])
    Bq = zeros([NDIM, NT, NV])
    for b in range(NT):
        qb = q[b]
        source_ref(ret[b], qb, MP)
        for d in range(NDIM):
            flux_ref(Fq[d, b], qb, d, MP)
            Bdot(Bq[d, b], Tq[d, b], qb, d, MP)

    for d in range(NDIM):
        ret -= Bq[d] / dX[d]

    ret *= DG_Z
    for d in range(NDIM):
        ret -= dot(DG_V[d], Fq[d]) / dX[d]

    return dt * ret + Ww


def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(N)])
    return ret.reshape([NT, NV])


def hidalgo_initial_guess(w, dtGAPS, dX, MP):
    """ Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6
        NOTE: Only in 1D
    """
    q = zeros([N] * (NDIM + 1) + [NV])
    qt = w

    for t in range(N):
        dt = dtGAPS[t]
        dqdxj = dot(DERVALS, qt)

        for i in range(N):
            qi = qt[i]
            dqdxi = dqdxj[i]

            M = dot(system(qi, 0, MP), dqdxi)
            Sj = source(qi, MP)

            if STIFF_IG:
                def f(X): return (X - qi + dt / dX[0] * M
                                  - dt / 2 * (Sj + source(X, MP)))
                q[t, i] = newton_krylov(f, qi, f_tol=DG_TOL)
            else:
                q[t, i] = qi - dt / dX[0] * M + dt * Sj

        qt = q[t]
    return q.reshape([NT, NV])


def failed(w, f, dtGAPS, MP):
    #q = hidalgo_initial_guess(w, dtGAPS, dX, MP)
    q = standard_initial_guess(w)
    return newton_krylov(f, q, f_tol=DG_TOL, method='bicgstab')


def unconverged(q, qNew):
    """ Mixed convergence condition
    """
    return (absolute(q - qNew) > DG_TOL * (1 + absolute(q))).any()


def predictor(wh, dt, dX, MP):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    nx, ny, nz, = wh.shape[:3]
    wh = wh.reshape([nx, ny, nz, N**NDIM, NV])
    qh = zeros([nx, ny, nz, NT, NV])
    dtGAPS = dt * GAPS

    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        Ww = dot(DG_W, w)

        def obj(X): return dot(DG_U, X) - rhs(X, Ww, dt, dX, MP)

        if HIDALGO:
            q = hidalgo_initial_guess(w, dtGAPS, dX, MP)
        else:
            q = standard_initial_guess(w)

        if STIFF:
            qh[i, j, k] = newton_krylov(obj, q, f_tol=DG_TOL, method='bicgstab')

        else:
            for count in range(DG_IT):

                qNew = solve(DG_U, rhs(q, Ww, dt, dX, MP), check_finite=False)

                if (absolute(qNew) > MAX_TOL).any():
                    qh[i, j, k] = failed(w, obj, dtGAPS, MP)
                    break
                elif unconverged(q, qNew):
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break
            else:
                qh[i, j, k] = failed(w, obj, dtGAPS, MP)

    return qh


def dg_launcher(pool, wh, dt, dX, MP):
    """ Controls the parallel computation of the Galerkin predictor
    """
    if PARA_DG:
        nx = wh.shape[0]
        step = int(nx / NCORE)
        chunk = array([i * step for i in range(NCORE)] + [nx + 1])
        n = len(chunk) - 1
        qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i + 1]], dt, dX, MP)
                      for i in range(n))
        return concatenate(qhList)
    else:
        return predictor(wh, dt, dX, MP)
