from itertools import product

from numpy import absolute, array, dot, isnan, zeros
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov

from ader.dg_matrices import system_matrices, UinvDot1
from ader.basis import quad, derivative_values
from gpr.variables.vectors import Cvec_to_Pvec
from gpr.matrices.conserved import source, flux_ref, source_ref, Bdot, system_conserved
from gpr.matrices.primitive import source_primitive_ref, Mdot_ref, source_primitive
from options import ndim, dx, N1, NT, reconstructPrim
from options import stiff, superStiff, hidalgo, TOL, MAX_ITER, failLim


W, U, V, Z, T = system_matrices()
_, gaps, _ = quad()
derivs = derivative_values()
stiff = stiff


def rhs_conserved(q, Ww, dt, PAR, SYS, homogeneous):
    """ Returns the right handside of the linear system governing the coefficients of qh
    """
    Tq = dot(T, q)
    Sq = zeros([NT, 18])
    Fq = zeros([ndim, NT, 18])
    Bq = zeros([ndim, NT, 18])
    for b in range(NT):
        P = Cvec_to_Pvec(q[b], PAR, SYS)
        if not homogeneous:
            source_ref(Sq[b], P, PAR, SYS)
        for d in range(ndim):
            flux_ref(Fq[d,b], P, d, PAR, SYS)
            if SYS.viscous:
                Bdot(Bq[d,b], Tq[d,b], P[2:5], d)

    ret = dx*Sq
    for d in range(ndim):
        ret -= Bq[d]

    ret *= Z
    for d in range(ndim):
        ret -= dot(V[d], Fq[d])

    return dt/dx * ret + Ww

def rhs_primitive(p, Ww, dt, PAR, SYS, homogeneous):
    """ Returns the right handside of the linear system governing the coefficients of ph
    """
    Tp = dot(T, p)
    Sp = zeros([NT, 18])
    Mp = zeros([ndim, NT, 18])
    for b in range(NT):
        P = p[b]
        if not homogeneous:
            source_primitive_ref(Sp[b], P, PAR, SYS)
        for d in range(ndim):
            Mdot_ref(Mp[d,b], P, Tp[d,b], d, PAR, SYS)

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

def hidalgo_initial_guess(w, dtgaps, PAR, SYS, homogeneous):
    """ Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6
    """
    q = zeros([N1]*(ndim+1) + [18])
    qj = w

    for j in range(N1):
        dt = dtgaps[j]
        dqdxj = dot(derivs, qj)

        for i in range(N1):
            qij = qj[i]
            dqdxij = dqdxj[i]

            if reconstructPrim:
                M = zeros(18)
                Mdot_ref(M, qij, dqdxij, 0, PAR, SYS)
                Sj = source_primitive(qij, PAR, SYS)
            else:
                M = dot(system_conserved(qij, 0, PAR, SYS), dqdxij)
                Sj = source(qij, PAR, SYS)

            if superStiff:
                if homogeneous:
                    if reconstructPrim:
                        f = lambda X: X - qij + dt/dx * M
                    else:
                        f = lambda X: X - qij + dt/dx * M
                else:
                    if reconstructPrim:
                        f = lambda X: X - qij + dt/dx * M - dt/2 * (Sj+source_primitive(X,PAR,SYS))
                    else:
                        f = lambda X: X - qij + dt/dx * M - dt/2 * (Sj+source(X,PAR,SYS))
                q[j,i] = newton_krylov(f, qij, f_tol=TOL)
            else:
                q[j,i] = qij - dt/dx * M + dt * Sj

        qj = q[j]
    return q.reshape([NT, 18])

def failed(w, qh, i, j, k, f, dtgaps, PAR, SYS):
    q = hidalgo_initial_guess(w, dtgaps, PAR, SYS)
    qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

def predictor(wh, dt, PAR, SYS, homogeneous=0):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    global stiff
    nx, ny, nz, = wh.shape[:3]
    wh = wh.reshape([nx, ny, nz, N1**ndim, 18])
    qh = zeros([nx, ny, nz, NT, 18])
    dtgaps = dt * gaps

    if reconstructPrim:
        rhs = lambda X, Ww: rhs_primitive(X, Ww, dt, PAR, SYS, homogeneous)
    else:
        rhs = lambda X, Ww: rhs_conserved(X, Ww, dt, PAR, SYS, homogeneous)

    failCount = 0
    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        Ww = dot(W, w)
        f = lambda X: X - spsolve(U, rhs(X, Ww))

        if hidalgo:
            q = hidalgo_initial_guess(w, dtgaps, PAR, SYS)
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
                    failed(w, qh, i, j, k, f, dtgaps, PAR, SYS)
                    failCount += 1
                    break
                elif (absolute(q-qNew) > TOL * (1 + absolute(q))).any():# Mixed convergence cond.
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break
            else:
                failed(w, qh, i, j, k, f, dtgaps, PAR, SYS)

    if failCount > failLim:
        stiff = 1
        print('Defaulting to Stiff Solver')
    return qh
