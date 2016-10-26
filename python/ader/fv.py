from itertools import product

from numpy import dot, zeros, tensordot

from ader.fv_fluxes import Dos, Drus
from ader.basis import quad, end_values, derivative_values
from gpr.matrices.conserved import block, source
from options import ndim, dx, N1, method


nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()

if method == 'osher':
    D = Dos
elif method == 'rusanov':
    D = Drus


def endpoints(qh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        qEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    nx, ny, nz = qh.shape[:3]
    qh0 = qh.reshape([nx, ny, nz] + [N1]*(ndim+1) + [18])
    qEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*ndim + [18])
    for d in range(ndim):
        qEnd[d] = tensordot(endVals, qh0, (0,4+d))
    return qEnd

def interface(qEndL, qEndM, qEndR, d, params, subsystems):
    """ Returns flux term and jump term in dth direction at the interface between states qhL, qhR
    """
    ret = zeros(18)
    for a in range(N1):
        if ndim > 1:
            for b in range(N1):
                if ndim > 2:
                    for c in range(N1):
                        qL1 = qEndL[d, 1, a, b, c]
                        qM0 = qEndM[d, 0, a, b, c]
                        qM1 = qEndM[d, 1, a, b, c]
                        qR0 = qEndR[d, 0, a, b, c]
                        weight = weights[a] * weights[b] * weights[c]
                        ret += weight * (D(qM1, qR0, d, 1, params, subsystems)
                                         + D(qM0, qL1, d, 0, params, subsystems))
                else:
                    qL1 = qEndL[d, 1, a, b]
                    qM0 = qEndM[d, 0, a, b]
                    qM1 = qEndM[d, 1, a, b]
                    qR0 = qEndR[d, 0, a, b]
                    weight = weights[a] * weights[b]
                    ret += weight * (D(qM1, qR0, d, 1, params, subsystems)
                                     + D(qM0, qL1, d, 0, params, subsystems))
        else:
            qL1 = qEndL[d, 1, a]
            qM0 = qEndM[d, 0, a]
            qM1 = qEndM[d, 1, a]
            qR0 = qEndR[d, 0, a]
            ret += weights[a] * (D(qM1, qR0, d, 1, params, subsystems)
                                 + D(qM0, qL1, d, 0, params, subsystems))

    return 0.5 * ret

def center(qhijk, t, x, y, z, params, subsystems):
    """ Returns the space-time averaged source term and non-conservative term in cell ijk
    """
    if ndim > 1:
        if ndim > 2:
            qx = qhijk[t, :, y, z]
            qy = qhijk[t, x, :, z]
            qz = qhijk[t, x, y, :]
            q = qhijk[t, x, y, z]
        else:
            qx = qhijk[t, :, y]
            qy = qhijk[t, x, :]
            q = qhijk[t, x, y]
    else:
        qx = qhijk[t, :]
        q = qhijk[t, x]

    term = dx * source(q, params, subsystems)

    dqdx = dot(derivs, qx)[x]
    v = q[2:5] / q[0]
    term -= dot(block(v, 0, subsystems.viscous), dqdx)
    if ndim > 1:
        dqdy = dot(derivs, qy)[y]
        term -= dot(block(v, 1, subsystems.viscous), dqdy)
        if ndim > 2:
            dqdz = dot(derivs, qz)[z]
            term -= dot(block(v, 2, subsystems.viscous), dqdz)

    return term

def fv_terms(qh, params, dt, subsystems):
    """ Returns the space-time averaged interface terms, jump terms, source terms, and
        non-conservative terms
    """
    if ndim < 3:
        qh0 = qh.repeat([3], axis=2)
    if ndim < 2:
        qh0 = qh.repeat([3], axis=1)
        qh0 = qh0.repeat([3], axis=2)

    nx = qh0.shape[0] - 2; ny = qh0.shape[1] - 2; nz = qh0.shape[2] - 2
    qEnd = endpoints(qh0)
    qh0 = qh0.reshape([nx+2, ny+2, nz+2] + [N1]*(ndim+1) + [18])

    s = zeros([nx, ny, nz, 18])
    f = zeros([nx, ny, nz, 18])
    g = zeros([nx, ny, nz, 18])
    h = zeros([nx, ny, nz, 18])

    interface_func = lambda qL, qM, qR, d: interface(qL, qM, qR, d, params, subsystems)
    center_func = lambda qhijk, t, x, y, z: center(qhijk, t, x, y, z, params, subsystems)

    for i, j, k in product(range(nx), range(ny), range(nz)):

        qhijk = qh0[i+1, j+1, k+1]
        for t in range(N1):
            for x in range(N1):
                if ndim > 1:
                    for y in range(N1):
                        if ndim > 2:
                            for z in range(N1):
                                weight = weights[t] * weights[x] * weights[y] * weights[z]
                                s[i, j, k] += weight * center_func(qhijk, t, x, y, z)
                        else:
                            weight = weights[t] * weights[x] * weights[y]
                            s[i, j, k] += weight * center_func(qhijk, t, x, y, 0)
                else:
                    weight = weights[t] * weights[x]
                    s[i, j, k] += weight * center_func(qhijk, t, x, 0, 0)

        qEndM = qEnd[:, :, i+1, j+1, k+1]
        qEndL = qEnd[:, :, i,   j+1, k+1]
        qEndR = qEnd[:, :, i+2, j+1, k+1]
        f[i ,j, k] = interface_func(qEndL, qEndM, qEndR, 0)
        if ndim > 1:
            qEndL = qEnd[:, :, i+1, j,   k+1]
            qEndR = qEnd[:, :, i+1, j+2, k+1]
            g[i, j, k] = interface_func(qEndL, qEndM, qEndR, 1)
            if ndim > 2:
                qEndL = qEnd[:, :, i+1, j+1, k]
                qEndR = qEnd[:, :, i+1, j+1, k+2]
                h[i, j, k] = interface_func(qEndL, qEndM, qEndR, 2)

    return dt/dx * (s - f - g - h)
