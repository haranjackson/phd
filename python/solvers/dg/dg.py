from itertools import product

from joblib import delayed
from numpy import absolute, array, concatenate, dot, zeros
from scipy.linalg import solve
from scipy.optimize import newton_krylov

from solvers.dg.matrices import DG_W, DG_U, DG_V, DG_Z, DG_T
from solvers.basis import GAPS, DERVALS
from system.system import source, flux_ref, source_ref, Bdot, system
from options import ndim, dx, N1, NT, nV
from options import STIFF, SUPER_STIFF, HIDALGO, DG_TOL, MAX_ITER, PARA_DG, NCORE


MAX_TOL = 1e16  # values above this level will cause an error


def rhs(q, Ww, dt, PAR, HOMOGENEOUS):
    """ Returns the right-handside of the system governing coefficients of qh
    """
    ret = zeros([NT, nV])

    Tq = dot(DG_T, q)
    Fq = zeros([ndim, NT, nV])
    Bq = zeros([ndim, NT, nV])
    for b in range(NT):
        qb = q[b]
        if not HOMOGENEOUS:
            source_ref(ret[b], qb, PAR)
        for d in range(ndim):
            flux_ref(Fq[d,b], qb, d, PAR)
            Bdot(Bq[d,b], Tq[d,b], qb, d, PAR)

    if not HOMOGENEOUS:
        ret *= dx

    for d in range(ndim):
        ret -= Bq[d]

    ret *= DG_Z
    for d in range(ndim):
        ret -= dot(DG_V[d], Fq[d])

    return (dt/dx) * ret + Ww

def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(N1)])
    return ret.reshape([NT, nV])

def hidalgo_initial_guess(w, dtGAPS, PAR, HOMOGENEOUS):
    """ Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6
    """
    q = zeros([N1]*(ndim+1) + [nV])
    qt = w

    for t in range(N1):
        dt = dtGAPS[t]
        dqdxj = dot(DERVALS, qt)

        for i in range(N1):
            qi = qt[i]
            dqdxi = dqdxj[i]

            M = dot(system(qi, 0, PAR), dqdxi)
            Sj = source(qi, PAR)

            if SUPER_STIFF and not HOMOGENEOUS:
                f = lambda X: X - qi + dt/dx * M - dt/2 * (Sj+source(X,PAR))
                q[t,i] = newton_krylov(f, qi, f_tol=DG_TOL)
            else:
                q[t,i] = qi - dt/dx * M + dt * Sj

        qt = q[t]
    return q.reshape([NT, nV])

def failed(w, f, dtGAPS, PAR, HOMOGENEOUS):
    q = hidalgo_initial_guess(w, dtGAPS, PAR, HOMOGENEOUS)
    return newton_krylov(f, q, f_tol=DG_TOL, method='bicgstab')

def unconverged(q, qNew):
    """ Mixed convergence condition
    """
    return (absolute(q-qNew) > DG_TOL * (1 + absolute(q))).any()

def predictor(wh, dt, PAR, HOMOGENEOUS=0):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    nx, ny, nz, = wh.shape[:3]
    wh = wh.reshape([nx, ny, nz, N1**ndim, nV])
    qh = zeros([nx, ny, nz, NT, nV])
    dtGAPS = dt * GAPS

    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        Ww = dot(DG_W, w)
        obj = lambda X: dot(DG_U,X) - rhs(X, Ww, dt, PAR, HOMOGENEOUS)

        if HIDALGO:
            q = hidalgo_initial_guess(w, dtGAPS, PAR, HOMOGENEOUS)
        else:
            q = standard_initial_guess(w)

        if STIFF:
            qh[i, j, k] = newton_krylov(obj, q, f_tol=DG_TOL, method='bicgstab')

        else:
            for count in range(MAX_ITER):

                qNew = solve(DG_U, rhs(q, Ww, dt, PAR, HOMOGENEOUS),
                             check_finite=False)

                if (absolute(qNew) > MAX_TOL).any():
                    qh[i, j, k] = failed(w, obj, dtGAPS, PAR, HOMOGENEOUS)
                    break
                elif unconverged(q, qNew):
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break
            else:
                qh[i, j, k] = failed(w, obj, dtGAPS, PAR, HOMOGENEOUS)

    return qh

def dg_launcher(pool, wh, dt, PAR, HOMOGENEOUS=0):
    """ Controls the parallel computation of the Galerkin predictor
    """
    if PARA_DG:
        nx = wh.shape[0]
        step = int(nx / NCORE)
        chunk = array([i*step for i in range(NCORE)] + [nx+1])
        n = len(chunk) - 1
        qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i+1]], dt, PAR, HOMOGENEOUS)
                                        for i in range(n))
        return concatenate(qhList)
    else:
        return predictor(wh, dt, PAR, HOMOGENEOUS)
