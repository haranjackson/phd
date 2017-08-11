from itertools import product

from joblib import delayed
from numpy import absolute, array, concatenate, dot, isnan, zeros
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov

from solvers.dg.matrices import system_matrices, UinvDot1
from solvers.basis import quad, derivative_values
from gpr.variables.vectors import Cvec_to_Pvec
from gpr.matrices.conserved import source, flux_ref, source_ref, Bdot, system_conserved
from gpr.matrices.primitive import source_primitive_ref, Mdot_ref, source_primitive
from options import ndim, dx, N1, NT, reconstructPrim
from options import stiff, superStiff, hidalgo, TOL, MAX_ITER, failLim, paraDG, ncore
from options import VISCOUS


W, U, V, Z, T = system_matrices()
_, gaps, _ = quad()
derivs = derivative_values()
stiff = stiff


idxDer = [ndim] + [N1]*ndim + [1]*(3-ndim)


def rhs_conserved(q, Ww, dt, PAR, homogeneous):
    """ Returns the right handside of the linear system governing the coefficients of qh
    """
    Tq = dot(T, q)
    ret = zeros([NT, 18])
    Fq = zeros([ndim, NT, 18])
    Bq = zeros([ndim, NT, 18])
    for b in range(NT):
        qb = q[b]
        P = Cvec_to_Pvec(qb, PAR)
        E = qb[1] / qb[0]

        if not homogeneous:
            source_ref(ret[b], P, PAR)
        for d in range(ndim):
            flux_ref(Fq[d,b], P, E, d, PAR)
            if VISCOUS:
                Bdot(Bq[d,b], Tq[d,b], P[2:5], d)

    if not homogeneous:
        ret *= dx

    if VISCOUS:
        for d in range(ndim):
            ret -= Bq[d]

    ret *= Z
    for d in range(ndim):
        ret -= dot(V[d], Fq[d])

    return (dt/dx) * ret + Ww

def rhs_primitive(p, Ww, dt, PAR, homogeneous):
    """ Returns the right handside of the linear system governing the coefficients of ph
    """
    Tp = dot(T, p)
    Sp = zeros([NT, 18])
    Mp = zeros([ndim, NT, 18])
    for b in range(NT):
        P = p[b]
        if not homogeneous:
            source_primitive_ref(Sp[b], P, PAR)
        for d in range(ndim):
            Mdot_ref(Mp[d,b], P, Tp[d,b], d, PAR)

    ret = dx*Sp
    for d in range(ndim):
        ret -= Mp[d]
    ret *= Z

    return dt/dx * ret + Ww

def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(N1)])
    return ret.reshape([NT, 18])

def hidalgo_initial_guess(w, dtgaps, PAR, homogeneous):
    """ Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6
    """
    q = zeros([N1]*(ndim+1) + [18])
    qt = w

    for t in range(N1):
        dt = dtgaps[t]
        dqdxj = dot(derivs, qt)

        for i in range(N1):
            qi = qt[i]
            dqdxi = dqdxj[i]

            if reconstructPrim:
                M = zeros(18)
                Mdot_ref(M, qi, dqdxi, 0, PAR)
                Sj = source_primitive(qi, PAR)
            else:
                M = dot(system_conserved(qi, 0, PAR), dqdxi)
                Sj = source(qi, PAR)

            if superStiff and not homogeneous:
                if reconstructPrim:
                    f = lambda X: X - qi + dt/dx * M - dt/2 * (Sj+source_primitive(X,PAR))
                else:
                    f = lambda X: X - qi + dt/dx * M - dt/2 * (Sj+source(X,PAR))
                q[t,i] = newton_krylov(f, qi, f_tol=TOL)
            else:
                q[t,i] = qi - dt/dx * M + dt * Sj

        qt = q[t]
    return q.reshape([NT, 18])

def failed(w, qh, i, j, k, f, dtgaps, PAR, homogeneous):
    q = hidalgo_initial_guess(w, dtgaps, PAR, homogeneous)
    qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

def predictor(wh, dt, PAR, homogeneous=0):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    global stiff
    nx, ny, nz, = wh.shape[:3]
    wh = wh.reshape([nx, ny, nz, N1**ndim, 18])
    qh = zeros([nx, ny, nz, NT, 18])
    dtgaps = dt * gaps

    if reconstructPrim:
        rhs = lambda X, Ww: rhs_primitive(X, Ww, dt, PAR, homogeneous)
    else:
        rhs = lambda X, Ww: rhs_conserved(X, Ww, dt, PAR, homogeneous)

    failCount = 0
    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        Ww = dot(W, w)
        f = lambda X: X - spsolve(U, rhs(X, Ww))

        if hidalgo:
            q = hidalgo_initial_guess(w, dtgaps, PAR, homogeneous)
        else:
            q = standard_initial_guess(w)

        if stiff:
            qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

        else:
            for count in range(MAX_ITER):

                if N1==2:
                    qNew = UinvDot1(rhs(q,Ww))
                else:
                    qNew = spsolve(U, rhs(q, Ww))

                if isnan(qNew).any():
                    failed(w, qh, i, j, k, f, dtgaps, PAR, homogeneous)
                    failCount += 1
                    break
                elif (absolute(q-qNew) > TOL * (1 + absolute(q))).any():# Mixed convergence cond.
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break
            else:
                failed(w, qh, i, j, k, f, dtgaps, PAR, homogeneous)

    if failCount > failLim:
        stiff = 1
        print('Defaulting to Stiff Solver')
    return qh

def dg_launcher(pool, wh, dt, PAR, homogeneous=0):
    """ Controls the parallel computation of the Galerkin predictor
    """
    if paraDG:
        nx = wh.shape[0]
        step = int(nx / ncore)
        chunk = array([i*step for i in range(ncore)] + [nx+1])
        n = len(chunk) - 1
        qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i+1]], dt, PAR, homogeneous)
                                        for i in range(n))
        return concatenate(qhList)
    else:
        return predictor(wh, dt, PAR, homogeneous)
