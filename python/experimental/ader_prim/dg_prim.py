from itertools import product

from numpy import absolute, array, dot, isnan, zeros
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov

from ader.dg_matrices import system_matrices
from ader.basis import quad, derivative_values
from experimental.ader_prim.matrices_prim import Mdot_ref, source_primitive_ref
from experimental.ader_prim.matrices_prim import  source_primitive_ret
from options import stiff, superStiff, hidalgo, TOL, ndim, dx, MAX_ITER, N1, NT, failLim


W, U, V, Z, T = system_matrices()
_, gaps, _ = quad()
derivs = derivative_values()
stiff = stiff


def rhs(p, Ww, dt, γ, pINF, cv, α2, viscous, thermal):
    """ Returns the right handside of the linear system governing the coefficients of ph
    """
    Tp = dot(T, p)
    Sp = zeros([NT, 18])
    Mp = zeros([ndim, NT, 18])
    for b in range(NT):
        source_primitive_ref(Sp[b], p[b], γ, pINF, cv, viscous, thermal)
        for d in range(ndim):
            Mdot_ref(Mp[d,b], p[b], Tp[d,b], d, γ, pINF, cv, α2, viscous, thermal)

    ret = dx*Sp
    for d in range(ndim):
        ret -= Mp[d]
    ret *= Z

    return dt/dx * ret + Ww

def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of p at t=0
    """
    p = array([w for i in range(N1)])
    return p.reshape([NT, 18])

def hidalgo_initial_guess(w, dtgaps, γ, pINF, cv, α2, viscous, thermal):
    """ Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6
    """
    p = zeros([N1]*(ndim+1) + [18])
    pj = w
    for j in range(N1):
        dt = dtgaps[j]
        dpdxj = dot(derivs, pj)
        for i in range(N1):
            pij = pj[i]
            dpdxij = dpdxj[i]
            M = zeros(18)
            Mdot_ref(M, pij, dpdxij, 0, γ, pINF, cv, α2, viscous, thermal)
            Sj = zeros(18)
            source_primitive_ref(Sj, pij, γ, pINF, cv, viscous, thermal)
            if superStiff:
                f = lambda X: X - pij + dt/dx * M - dt/2 * (Sj +
                                                            source_primitive_ret(X, γ, pINF, cv,
                                                                                 viscous, thermal))
                p[j,i] = newton_krylov(f, pij, f_tol=TOL)
            else:
                p[j,i] = pij - dt/dx * M + dt * Sj
        pj = p[j]
    return p.reshape([NT, 18])

def predictor_prim(wh, params, dt, subsystems):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    global stiff
    nx, ny, nz, = wh.shape[:3]
    ph = zeros([nx, ny, nz, NT, 18])
    dtgaps = dt * gaps

    γ, pINF, cv, α2 = params.γ, params.pINF, params.cv, params.α2
    viscous, thermal = subsystems.viscous, subsystems.thermal

    def failed(w, ph, i, j, k, f):
        p = hidalgo_initial_guess(w, dtgaps, γ, pINF, cv, α2, viscous, thermal)
        ph[i, j, k] = newton_krylov(f, p, f_tol=TOL, method='bicgstab')

    failCount = 0
    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k].reshape([N1**ndim, 18])
        Ww = dot(W, w)
        f = lambda X: X - spsolve(U, rhs(X, Ww, dt, γ, pINF, cv, α2, viscous, thermal))

        if hidalgo:
            p = hidalgo_initial_guess(w, dtgaps, γ, pINF, cv, α2, viscous, thermal)
        else:
            p = standard_initial_guess(w)

        if stiff:
            ph[i, j, k] = newton_krylov(f, p, f_tol=TOL, method='bicgstab')

        else:
            for count in range(MAX_ITER):
                pNew = spsolve(U, rhs(p, Ww, dt, γ, pINF, cv, α2, viscous, thermal))

                if isnan(pNew).any():
                    failed(w, ph, i, j, k, f)
                    failCount += 1
                    break
                elif (absolute(p-pNew) > TOL * (1 + absolute(p))).any():
                    p = pNew
                    continue
                else:
                    ph[i, j, k] = pNew
                    break
            else:
                failed(w, ph, i, j, k, f)

    if failCount > failLim:
        stiff = 1
        print('Defaulting to Stiff Solver')
    return ph
