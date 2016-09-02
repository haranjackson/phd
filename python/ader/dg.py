from numpy import absolute, array, dot, isnan, zeros
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov

from ader.dg_matrices import system_matrices
from ader.basis import quad, derivative_values
from gpr.functions import primitive
from gpr.matrices import source, jacobian, flux_ref, source_ref, Bdot
from options import stiff, superStiff, hidalgo, TOL, ndim, dx, MAX_ITER, N1, NT, failLim


F, K0, K, M, Kt = system_matrices()
_, gaps, _ = quad()
derivs = derivative_values()
stiff = stiff


def rhs(q, Fw, params, dt, mechanical, viscous, thermal, reactive):
    """ Returns the right handside of the linear system governing the coefficients of qh
    """
    Tq = dot(Kt, q)
    Sq = zeros([NT, 18])
    Fq = zeros([ndim, NT, 18])
    Bq = zeros([ndim, NT, 18])
    for b in range(NT):
        P = primitive(q[b], params, viscous, thermal, reactive)
        source_ref(Sq[b], P, params, viscous, thermal, reactive)
        for d in range(ndim):
            flux_ref(Fq[d,b], P, d, params, mechanical, viscous, thermal, reactive)
            Bdot(Bq[d,b], Tq[d,b], d, P.v, viscous)

    ret = dx*Sq
    for d in range(ndim):
        ret -= Bq[d]

    ret *= M
    for d in range(ndim):
        ret -= dot(K[d], Fq[d])

    return dt/dx * ret + Fw

def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    q = array([w for i in range(N1)])
    return q.reshape([NT, 18])

def hidalgo_initial_guess(w, params, dtgaps, viscous, thermal, reactive):
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
            J = dot(jacobian(qij, 0, params, viscous, thermal, reactive), dqdxij)
            Sj = source(qij, params, viscous, thermal, reactive)
            if superStiff:
                f = lambda X: X - qij + dt/dx * J - dt/2 * (Sj + source(X, params, viscous, thermal, reactive))
                q[j,i] = newton_krylov(f, qij, f_tol=TOL)
            else:
                q[j,i] = qij - dt/dx * J + dt * Sj
        qj = q[j]
    return q.reshape([NT, 18])

def predictor(wh, params, dt, mechanical, viscous, thermal, reactive):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    global stiff
    nx, ny, nz, = wh.shape[:3]
    qh = zeros([nx, ny, nz, NT, 18])
    dtgaps = dt * gaps

    def failed(w, qh, i, j, k, f):
        q = hidalgo_initial_guess(w, params, dtgaps, viscous, thermal, reactive)
        qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

    failCount = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                w = wh[i, j, k]
                Fw = dot(F, w)
                f = lambda X: X - spsolve(K0, rhs(X, Fw, params, dt,
                                                  mechanical, viscous, thermal, reactive))

                if hidalgo:
                    q = hidalgo_initial_guess(w, params, dtgaps, viscous, thermal, reactive)
                else:
                    q = standard_initial_guess(w)

                if stiff:
                    qh[i, j, k] = newton_krylov(f, q, f_tol=TOL, method='bicgstab')

                else:
                    for count in range(MAX_ITER):
                        qNew = spsolve(K0, rhs(q, Fw, params, dt,
                                               mechanical, viscous, thermal, reactive))

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
