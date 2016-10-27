from itertools import product

from numpy import dot, zeros, tensordot

from experimental.ader_prim.fv_fluxes_prim import Dos, Drus
from ader.basis import quad, end_values, derivative_values
from experimental.ader_prim.matrices_prim import source_primitive_ret
from gpr.matrices.conserved import B0dot, B1dot, B2dot
from options import ndim, dx, N1, method


nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()

if method == 'osher':
    D = Dos
elif method == 'rusanov':
    D = Drus


def endpoints(ph):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        pEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    nx, ny, nz = ph.shape[:3]
    ph0 = ph.reshape([nx, ny, nz] + [N1]*(ndim+1) + [18])
    pEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*ndim + [18])
    for d in range(ndim):
        pEnd[d] = tensordot(endVals, ph0, (0,4+d))
    return pEnd

def interface(pEndL, pEndM, pEndR, d, params, subsystems):
    """ Returns flux term and jump term in dth direction at the interface between states phL, phR
    """
    ret = zeros(18)
    for a in range(N1):
        if ndim > 1:
            for b in range(N1):
                if ndim > 2:
                    for c in range(N1):
                        pL1 = pEndL[d, 1, a, b, c]
                        pM0 = pEndM[d, 0, a, b, c]
                        pM1 = pEndM[d, 1, a, b, c]
                        pR0 = pEndR[d, 0, a, b, c]
                        weight = weights[a] * weights[b] * weights[c]
                        ret += weight * (D(pM1, pR0, d, 1, params, subsystems)
                                         + D(pM0, pL1, d, 0, params, subsystems))
                else:
                    pL1 = pEndL[d, 1, a, b]
                    pM0 = pEndM[d, 0, a, b]
                    pM1 = pEndM[d, 1, a, b]
                    pR0 = pEndR[d, 0, a, b]
                    weight = weights[a] * weights[b]
                    ret += weight * (D(pM1, pR0, d, 1, params, subsystems)
                                     + D(pM0, pL1, d, 0, params, subsystems))
        else:
            pL1 = pEndL[d, 1, a]
            pM0 = pEndM[d, 0, a]
            pM1 = pEndM[d, 1, a]
            pR0 = pEndR[d, 0, a]
            ret += weights[a] * (D(pM1, pR0, d, 1, params, subsystems)
                                 + D(pM0, pL1, d, 0, params, subsystems))

    return 0.5 * ret

def center(phijk, t, x, y, z, γ, pINF, cv, viscous, thermal):
    """ Returns the space-time averaged source term and non-conservative term in cell ijk
    """
    if ndim > 1:
        if ndim > 2:
            px = phijk[t, :, y, z]
            py = phijk[t, x, :, z]
            pz = phijk[t, x, y, :]
            p = phijk[t, x, y, z]
        else:
            px = phijk[t, :, y]
            py = phijk[t, x, :]
            p = phijk[t, x, y]
    else:
        px = phijk[t, :]
        p = phijk[t, x]

    term = dx * source_primitive_ret(p, γ, pINF, cv, viscous, thermal)

    dqdx = dot(derivs, qx)[x]
    v = p[2:5]

    if subsystems.viscous:
        temp = zeros(18)
        B0dot(temp, dqdx, v, 1)
        term -= temp
        if ndim > 1:
            dqdy = dot(derivs, qy)[y]
            temp = zeros(18)
            B1dot(temp, dqdy, v, 1)
            term -= temp
            if ndim > 2:
                dqdz = dot(derivs, qz)[z]
                temp = zeros(18)
                B2dot(temp, dqdz, v, 1)
                term -= temp

    return term

def fv_terms(ph, params, dt, subsystems):
    """ Returns the space-time averaged interface terms, jump terms, source terms, and
        non-conservative terms
    """
    if ndim < 3:
        ph0 = ph.repeat([3], axis=2)
    if ndim < 2:
        ph0 = ph.repeat([3], axis=1)
        ph0 = ph0.repeat([3], axis=2)

    nx = ph0.shape[0] - 2; ny = ph0.shape[1] - 2; nz = ph0.shape[2] - 2
    pEnd = endpoints(ph0)
    ph0 = ph0.reshape([nx+2, ny+2, nz+2] + [N1]*(ndim+1) + [18])

    s = zeros([nx, ny, nz, 18])
    f = zeros([nx, ny, nz, 18])
    g = zeros([nx, ny, nz, 18])
    h = zeros([nx, ny, nz, 18])

    interface_func = lambda pL, pM, pR, d: interface(pL, pM, pR, d, params, subsystems)
    center_func = lambda phijk, t, x, y, z: center(phijk, t, x, y, z, γ, pINF, cv, viscous, thermal)

    for i, j, k in product(range(nx), range(ny), range(nz)):

        phijk = ph0[i+1, j+1, k+1]
        for t in range(N1):
            for x in range(N1):
                if ndim > 1:
                    for y in range(N1):
                        if ndim > 2:
                            for z in range(N1):
                                weight = weights[t] * weights[x] * weights[y] * weights[z]
                                s[i, j, k] += weight * center_func(phijk, t, x, y, z)
                        else:
                            weight = weights[t] * weights[x] * weights[y]
                            s[i, j, k] += weight * center_func(phijk, t, x, y, 0)
                else:
                    weight = weights[t] * weights[x]
                    s[i, j, k] += weight * center_func(phijk, t, x, 0, 0)

        pEndM = pEnd[:, :, i+1, j+1, k+1]
        pEndL = pEnd[:, :, i,   j+1, k+1]
        pEndR = pEnd[:, :, i+2, j+1, k+1]
        f[i ,j, k] = interface_func(pEndL, pEndM, pEndR, 0)
        if ndim > 1:
            pEndL = pEnd[:, :, i+1, j,   k+1]
            pEndR = pEnd[:, :, i+1, j+2, k+1]
            g[i, j, k] = interface_func(pEndL, pEndM, pEndR, 1)
            if ndim > 2:
                pEndL = pEnd[:, :, i+1, j+1, k]
                pEndR = pEnd[:, :, i+1, j+1, k+2]
                h[i, j, k] = interface_func(pEndL, pEndM, pEndR, 2)

    return dt/dx * (s - f - g - h)
