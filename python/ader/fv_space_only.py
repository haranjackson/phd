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


def endpoints(wh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        qEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    nx, ny, nz = wh.shape[:3]
    wh0 = wh.reshape([nx, ny, nz] + [N1]*ndim + [18])
    qEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*(ndim-1) + [18])
    for d in range(ndim):
        qEnd[d] = tensordot(endVals, wh0, (0,3+d))
    return qEnd

def interface(qEndL, qEndM, qEndR, d, params, subsystems):
    """ Returns flux term and jump term in dth direction at the interface between states qhL, qhR
    """
    ret = zeros(18)
    if ndim > 1:
        for b in range(N1):
            if ndim > 2:
                for c in range(N1):
                    qL1 = qEndL[d, 1, b, c]
                    qM0 = qEndM[d, 0, b, c]
                    qM1 = qEndM[d, 1, b, c]
                    qR0 = qEndR[d, 0, b, c]
                    weight = weights[b] * weights[c]
                    ret += weight * (D(qM1, qR0, d, 1, params, subsystems)
                                     + D(qM0, qL1, d, 0, params, subsystems))
            else:
                qL1 = qEndL[d, 1, b]
                qM0 = qEndM[d, 0, b]
                qM1 = qEndM[d, 1, b]
                qR0 = qEndR[d, 0, b]
                weight = weights[b]
                ret += weight * (D(qM1, qR0, d, 1, params, subsystems)
                                 + D(qM0, qL1, d, 0, params, subsystems))
    else:
        qL1 = qEndL[d, 1]
        qM0 = qEndM[d, 0]
        qM1 = qEndM[d, 1]
        qR0 = qEndR[d, 0]
        ret += D(qM1, qR0, d, 1, params, subsystems) + D(qM0, qL1, d, 0, params, subsystems)

    return 0.5 * ret

def center(whijk, x, y, z, params, subsystems):
    """ Returns the space-time averaged source term and non-conservative term in cell ijk
    """
    if ndim > 1:
        if ndim > 2:
            qx = whijk[:, y, z]
            qy = whijk[x, :, z]
            qz = whijk[x, y, :]
            q = whijk[x, y, z]
        else:
            qx = whijk[:, y]
            qy = whijk[x, :]
            q = whijk[x, y]
    else:
        qx = whijk[:]
        q = whijk[x]

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

def fv_terms_space_only(wh, params, dt, subsystems):
    """ Returns the space-time averaged interface terms, jump terms, source terms, and
        non-conservative terms
    """
    if ndim < 3:
        wh0 = wh.repeat([3], axis=2)
    if ndim < 2:
        wh0 = wh.repeat([3], axis=1)
        wh0 = wh0.repeat([3], axis=2)

    nx = wh0.shape[0] - 2; ny = wh0.shape[1] - 2; nz = wh0.shape[2] - 2
    qEnd = endpoints(wh0)
    wh0 = wh0.reshape([nx+2, ny+2, nz+2] + [N1]*ndim + [18])

    s = zeros([nx, ny, nz, 18])
    f = zeros([nx, ny, nz, 18])
    g = zeros([nx, ny, nz, 18])
    h = zeros([nx, ny, nz, 18])

    interface_func = lambda qL, qM, qR, d: interface(qL, qM, qR, d, params, subsystems)
    center_func = lambda whijk, x, y, z: center(whijk, x, y, z, params, subsystems)

    for i, j, k in product(range(nx), range(ny), range(nz)):

        whijk = wh0[i+1, j+1, k+1]
        for x in range(N1):
            if ndim > 1:
                for y in range(N1):
                    if ndim > 2:
                        for z in range(N1):
                            weight = weights[x] * weights[y] * weights[z]
                            s[i, j, k] += weight * center_func(whijk, x, y, z)
                    else:
                        weight = weights[x] * weights[y]
                        s[i, j, k] += weight * center_func(whijk, x, y, 0)
            else:
                weight = weights[x]
                s[i, j, k] += weight * center_func(whijk, x, 0, 0)

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
