from itertools import product

from numpy import array, dot, einsum, tensordot, zeros

from ader.fv_fluxes import Dos, Drus
from ader.basis import quad, end_values, derivative_values
from gpr.matrices.conserved import Bdot, source_ref
from gpr.matrices.jacobians import dQdPdot
from gpr.variables.vectors import Cvec_to_Pvec
from options import ndim, dx, N1, method, approxInterface, reconstructPrim, timeDim


nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()

if method == 'osher':
    D = Dos
elif method == 'rusanov':
    D = Drus

weightList = [weights if timeDim else array([1])] + [weights]*ndim + [array([1])]*(3-ndim)
weightListEnd = [weights if timeDim else array([1])] + [weights]*(ndim-1) + [array([1])]*(3-ndim)

weight = einsum('t,x,y,z', weightList[0], weightList[1], weightList[2], weightList[3])
weightEnd = einsum('t,a,b', weightListEnd[0], weightListEnd[1], weightListEnd[2])

index = [N1 if timeDim else 1] + [N1]*ndim + [1]*(3-ndim)
indexEnd = [N1 if (timeDim and not approxInterface) else 1] + [N1]*(ndim-1) + [1]*(3-ndim)


def endpoints(qh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        qEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    nx, ny, nz = qh.shape[:3]
    qh0 = qh.reshape([nx, ny, nz] + index + [18])
    qEnd = zeros([ndim, 2, nx, ny, nz] + indexEnd + [18])
    for d in range(ndim):
        temp = tensordot(endVals, qh0, (0,4+d))
        if approxInterface:
            qEnd[d,:,:,:,:,0] = tensordot(weights, temp, (0,4))
        else:
            qEnd[d] = temp
    return qEnd

def interface(qEndL, qEndM, qEndR, d, PAR, SYS):
    """ Returns flux term and jump term in dth direction at the interface between states qhL, qhR
    """
    ret = zeros(18)
    for t, x1, x2 in product(range(indexEnd[0]), range(indexEnd[1]), range(indexEnd[2])):
        qL1 = qEndL[d, 1, t, x1, x2]
        qM0 = qEndM[d, 0, t, x1, x2]
        qM1 = qEndM[d, 1, t, x1, x2]
        qR0 = qEndR[d, 0, t, x1, x2]
        ret += weightEnd[t,x1,x2] * (D(qM1, qR0, d, 1, PAR, SYS) + D(qM0, qL1, d, 0, PAR, SYS))

    return 0.5 * ret

def center(qhijk, t, inds, PAR, SYS, homogeneous=0):
    """ Returns the space-time averaged source term and non-conservative term in cell ijk
    """
    qxi = zeros([ndim, N1, 18])
    if ndim > 1:
        if ndim > 2:
            qxi[0] = qhijk[t, :, inds[1], inds[2]]
            qxi[1] = qhijk[t, inds[0], :, inds[2]]
            qxi[2] = qhijk[t, inds[0], inds[1], :]
            q = qhijk[t, inds[0], inds[1], inds[2]]
        else:
            qxi[0] = qhijk[t, :, inds[1], 0]
            qxi[1] = qhijk[t, inds[0], :, 0]
            q = qhijk[t, inds[0], inds[1], 0]
    else:
        qxi[0] = qhijk[t, :, 0, 0]
        q = qhijk[t, inds[0], 0, 0]

    ret = zeros(18)

    if not homogeneous:
        if reconstructPrim:
            P = q
        else:
            P = Cvec_to_Pvec(q, PAR, SYS)
        source_ref(ret, P, PAR, SYS)
        ret *= dx

    if SYS.viscous:
        if reconstructPrim:
            v = q[2:5]
        else:
            v = q[2:5] / q[0]
        for d in range(ndim):
            dqdxi = dot(derivs[inds[d]], qxi[d])
            if reconstructPrim:
                dqdxi = dQdPdot(q, dqdxi, PAR, SYS)
            temp = zeros(18)
            Bdot(temp, dqdxi, v, d)
            ret -= temp

    return ret

def fv_terms(qh, dt, PAR, SYS, homogeneous=0):
    """ Returns the space-time averaged interface terms, jump terms, source terms, and
        non-conservative terms
    """
    if ndim < 3:
        qh0 = qh.repeat([3], axis=2)
    if ndim < 2:
        qh0 = qh.repeat([3], axis=1)
        qh0 = qh0.repeat([3], axis=2)

    nx, ny, nz = array(qh0.shape[:3]) - 2
    qEnd = endpoints(qh0)
    qh0 = qh0.reshape([nx+2, ny+2, nz+2] + index + [18])

    s = zeros([nx, ny, nz, 18])
    F = zeros([ndim, nx, ny, nz, 18])

    interface_func = lambda qL, qM, qR, d: interface(qL, qM, qR, d, PAR, SYS)
    center_func = lambda qhijk, t, inds: center(qhijk, t, inds, PAR, SYS, homogeneous)

    for i, j, k in product(range(nx), range(ny), range(nz)):

        qhijk = qh0[i+1, j+1, k+1]

        for t, x, y, z in product(range(index[0]),range(index[1]),range(index[2]),range(index[3])):
            s[i, j, k] += weight[t,x,y,z] * center_func(qhijk, t, [x, y, z])

        qEndM = qEnd[:, :, i+1, j+1, k+1]
        qEndL = qEnd[:, :, i,   j+1, k+1]
        qEndR = qEnd[:, :, i+2, j+1, k+1]
        F[0, i ,j, k] = interface_func(qEndL, qEndM, qEndR, 0)
        if ndim > 1:
            qEndL = qEnd[:, :, i+1, j,   k+1]
            qEndR = qEnd[:, :, i+1, j+2, k+1]
            F[1, i, j, k] = interface_func(qEndL, qEndM, qEndR, 1)
            if ndim > 2:
                qEndL = qEnd[:, :, i+1, j+1, k]
                qEndR = qEnd[:, :, i+1, j+1, k+2]
                F[2, i, j, k] = interface_func(qEndL, qEndM, qEndR, 2)

    for d in range(ndim):
        s -= F[d]
    return dt/dx * s
