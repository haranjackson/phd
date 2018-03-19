from joblib import delayed
from numpy import absolute, array, concatenate, dot, prod, zeros
from scipy.linalg import solve
from scipy.optimize import newton_krylov

from solvers.dg.initial_guess import standard_initial_guess, stiff_initial_guess
from solvers.dg.matrices import DG_W, DG_U, DG_V, DG_M, DG_D
from solvers.basis import GAPS
from system import flux_ref, source_ref, Bdot
from options import NDIM, N, NT, NV
from options import STIFF, STIFF_IG, DG_TOL, DG_IT, PARA_DG, NCORE


MAX_TOL = 1e16  # values above this level will cause an error


def rhs(q, Ww, dt, dX, MP):
    """ Returns the right-handside of the system governing coefficients of qh
    """
    ret = zeros([NT, NV])

    # Dq[d,i]: dq/dx at position i in direction d
    # Fq[d,i]: F(q) at position i in direction d
    # Bq[d,i]: B.dq/dx at position i in direction d
    Dq = dot(DG_D, q)
    Fq = zeros([NDIM, NT, NV])
    Bq = zeros([NDIM, NT, NV])

    for i in range(NT):
        qi = q[i]
        source_ref(ret[i], qi, MP)
        for d in range(NDIM):
            flux_ref(Fq[d, i], qi, d, MP)
            Bdot(Bq[d, i], Dq[d, i], qi, d, MP)

    for d in range(NDIM):
        ret -= Bq[d] / dX[d]

    ret *= DG_M
    for d in range(NDIM):
        ret -= dot(DG_V[d], Fq[d]) / dX[d]

    return dt * ret + Ww


def failed(w, f, dtGAPS, dX, MP):
    """ Attempts to find DG coefficients with Newton-Krylov, if standard
        method has failed
    """
    if STIFF_IG:
        q = stiff_initial_guess(w, dtGAPS, dX, MP)
    else:
        q = standard_initial_guess(w)
    return newton_krylov(f, q, f_tol=DG_TOL, method='bicgstab')


def unconverged(q, qNew):
    """ Mixed convergence condition
    """
    return (absolute(q - qNew) > DG_TOL * (1 + absolute(q))).any()


def predictor(wh, dt, dX, MP):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    shape = wh.shape
    n = prod(shape[:NDIM])
    wh = wh.reshape(n, N**NDIM, NV)
    qh = zeros([n, NT, NV])
    dtGAPS = dt * GAPS

    for i in range(n):

        w = wh[i]
        Ww = dot(DG_W, w)

        def obj(X): return dot(DG_U, X) - rhs(X, Ww, dt, dX, MP)

        if STIFF_IG:
            q = stiff_initial_guess(w, dtGAPS, dX, MP)
        else:
            q = standard_initial_guess(w)

        if STIFF:
            qh[i] = newton_krylov(obj, q, f_tol=DG_TOL, method='bicgstab')

        else:
            for count in range(DG_IT):

                qNew = solve(DG_U, rhs(q, Ww, dt, dX, MP), check_finite=False)

                if (absolute(qNew) > MAX_TOL).any():
                    qh[i] = failed(w, obj, dtGAPS, dX, MP)
                    break

                elif unconverged(q, qNew):
                    q = qNew
                    continue

                else:
                    qh[i] = qNew
                    break
            else:
                qh[i] = failed(w, obj, dtGAPS, dX, MP)

    return qh.reshape(shape[:NDIM] + (N,) * (NDIM + 1) + (NV,))


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
