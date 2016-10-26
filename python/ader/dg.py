from itertools import product

from numpy import absolute, array, dot, isnan, zeros
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov

from ader.dg_matrices import system_matrices
from ader.basis import quad, derivative_values
from gpr.variables.vectors import primitive
from gpr.matrices.conserved import source, flux_ref, source_ref, Bdot
from gpr.matrices.jacobians import jacobian
from options import stiff, superStiff, hidalgo, TOL, ndim, dx, MAX_ITER, N1, NT, failLim


W, U, V, Z, T = system_matrices()
_, gaps, _ = quad()
derivs = derivative_values()
stiff = stiff


def rhs(q, Ww, params, dt, subsystems):
    """ Returns the right handside of the linear system governing the coefficients of qh
    """
    Tq = dot(T, q)
    Sq = zeros([NT, 18])
    Fq = zeros([ndim, NT, 18])
    Bq = zeros([ndim, NT, 18])
    for b in range(NT):
        P = primitive(q[b], params, subsystems)
        source_ref(Sq[b], P, params, subsystems)
        for d in range(ndim):
            flux_ref(Fq[d,b], P, d, params, subsystems)
            Bdot(Bq[d,b], Tq[d,b], d, P.v, subsystems.viscous)

    ret = dx*Sq
    for d in range(ndim):
        ret -= Bq[d]

    ret *= Z
    for d in range(ndim):
        ret -= dot(V[d], Fq[d])

    return dt/dx * ret + Ww

def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    q = array([w for i in range(N1)])
    return q.reshape([NT, 18])

def hidalgo_initial_guess(w, params, dtgaps, subsystems):
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
            J = dot(jacobian(qij, 0, params, subsystems), dqdxij)
            Sj = source(qij, params, subsystems)
            if superStiff:
                f = lambda X: X - qij + dt/dx * J - dt/2 * (Sj + source(X, params, subsystems))
                q[j,i] = newton_krylov(f, qij, f_tol=TOL)
            else:
                q[j,i] = qij - dt/dx * J + dt * Sj
        qj = q[j]
    return q.reshape([NT, 18])

def predictor(wh, params, dt, subsystems):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    global stiff
    nx, ny, nz, = wh.shape[:3]
    qh = zeros([nx, ny, nz, NT, 18])
    dtgaps = dt * gaps

    def failed(w, qh, i, j, k, f):
        q = hidalgo_initial_guess(w, params, dtgaps, subsystems)
        qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

    failCount = 0
    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k].reshape([N1**ndim, 18])
        Ww = dot(W, w)
        f = lambda X: X - spsolve(U, rhs(X, Ww, params, dt, subsystems))

        if hidalgo:
            q = hidalgo_initial_guess(w, params, dtgaps, subsystems)
        else:
            q = standard_initial_guess(w)

        if stiff:
            qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

        else:
            for count in range(MAX_ITER):
                qNew = spsolve(U, rhs(q, Ww, params, dt, subsystems))

                if isnan(qNew).any():
                    failed(w, qh, i, j, k, f)
                    failCount += 1
                    break
                elif (absolute(q-qNew) > TOL * (1 + absolute(q))).any():
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break
            else:
                failed(w, qh, i, j, k, f)

    if failCount > failLim:
        stiff = 1
        print('Defaulting to Stiff Solver')
    return qh
