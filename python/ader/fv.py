from itertools import product

from numpy import array, dot, einsum, tensordot, zeros

from ader.fv_fluxes import Dos, Drus, input_vectors, Bint, Aint, Smax
from ader.basis import quad, end_values, derivative_values
from gpr.matrices.conserved import Bdot, source_ref, flux_ref
from gpr.matrices.jacobians import dQdPdot
from gpr.variables.vectors import Cvec_to_Pvec
from options import ndim, dx, N1, method, approxInterface, reconstructPrim, timeDim


altInterfaces = 1


nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()

if method == 'osher':
    D = Dos
    s_func = Aint
elif method == 'rusanov':
    D = Drus
    s_func = Smax

weightList = [weights if timeDim else array([1])] + [weights]*ndim + [array([1])]*(3-ndim)
weightListEnd = [weights if timeDim else array([1])] + [weights]*(ndim-1) + [array([1])]*(3-ndim)

weight = einsum('t,x,y,z', weightList[0], weightList[1], weightList[2], weightList[3])
weightEnd = einsum('t,x,y', weightListEnd[0], weightListEnd[1], weightListEnd[2])

idx = [N1 if timeDim else 1] + [N1]*ndim + [1]*(3-ndim)
idxEnd = [N1 if (timeDim and not approxInterface) else 1] + [N1]*(ndim-1) + [1]*(3-ndim)


def endpoints(xh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        xEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    nx, ny, nz = xh.shape[:3]
    xh0 = xh.reshape([nx, ny, nz] + idx + [18])
    xEnd = zeros([ndim, 2, nx, ny, nz] + idxEnd + [18])
    for d in range(ndim):
        temp = tensordot(endVals, xh0, (0,4+d))
        if approxInterface:
            xEnd[d,:,:,:,:,0] = tensordot(weights, temp, (0,4))
        else:
            xEnd[d] = temp
    return xEnd

def alternative_interfaces(xEnd, PAR, SYS):
    nx, ny, nz = xEnd.shape[2:5]
    fEnd = zeros([ndim, nx-1, ny-1, nz-1, 18])
    BEnd = zeros([ndim, nx-1, ny-1, nz-1, 18])

    inpt_lam = lambda xL, xR: input_vectors(xL, xR, PAR, SYS)
    flux_lam = lambda ftemp, p, d: flux_ref(ftemp, p, d, PAR, SYS)
    s_lam = lambda pL, pR, qL, qR, d: s_func(pL, pR, qL, qR, d, PAR, SYS)

    for d in range(ndim):
        for i, j, k in product(range(nx-1), range(ny-1), range(nz-1)):

            xL0 = xEnd[d, 1, i, j, k]
            if d==0:
                xR0 = xEnd[d, 0, i+1, j, k]
            elif d==1:
                xR0 = xEnd[d, 0, i, j+1, k]
            else:
                xR0 = xEnd[d, 0, i, j, k+1]

            fEndTemp = zeros(18)
            BEndTemp = zeros(18)
            for t, x1, x2 in product(range(idxEnd[0]), range(idxEnd[1]), range(idxEnd[2])):
                pL, pR, qL, qR = inpt_lam(xL0[t, x1, x2], xR0[t, x1, x2])
                ftemp = zeros(18)
                weight0 = weightEnd[t, x1, x2]

                flux_lam(ftemp, pL, d)
                flux_lam(ftemp, pR, d)
                ftemp -= s_lam(pL, pR, qL, qR, d)
                fEndTemp += weight0 * ftemp
                BEndTemp += weight0 * Bint(qL, qR, d, SYS.viscous)

            fEnd[d, i, j, k] = fEndTemp
            BEnd[d, i, j, k] = BEndTemp

    ret = zeros([nx-2, ny-2, nz-2, 18])
    ret -= fEnd[0, :-1, 1:, 1:]
    ret += fEnd[0, 1:,  1:, 1:]
    ret += BEnd[0, :-1, 1:, 1:]
    ret += BEnd[0, 1:,  1:, 1:]
    if ndim > 1:
        ret -= fEnd[1, 1:, :-1, 1:]
        ret += fEnd[1, 1:,  1:, 1:]
        ret += BEnd[1, 1:, :-1, 1:]
        ret += BEnd[1, 1:,  1:, 1:]
    if ndim > 2:
        ret -= fEnd[2, 1:, 1:, :-1]
        ret += fEnd[2, 1:, 1:,  1:]
        ret += BEnd[2, 1:, 1:, :-1]
        ret += BEnd[2, 1:, 1:,  1:]
    return ret

def interface(ret, xEndL, xEndM, xEndR, d, PAR, SYS):
    """ Returns flux term and jump term in dth direction at the interface between states xhL, xhR
    """
    for t, x1, x2 in product(range(idxEnd[0]), range(idxEnd[1]), range(idxEnd[2])):
        xL1 = xEndL[d, 1, t, x1, x2]
        xM0 = xEndM[d, 0, t, x1, x2]
        xM1 = xEndM[d, 1, t, x1, x2]
        xR0 = xEndR[d, 0, t, x1, x2]
        ret += 0.5 * weightEnd[t,x1,x2] * (D(xM1,xR0,d,1,PAR,SYS) + D(xM0,xL1,d,0,PAR,SYS))

def center(xhijk, t, inds, PAR, SYS, homogeneous=0):
    """ Returns the space-time averaged source term and non-conservative term in cell ijk
    """
    xxi = zeros([ndim, N1, 18])
    if ndim > 1:
        if ndim > 2:
            xxi[0] = xhijk[t, :, inds[1], inds[2]]
            xxi[1] = xhijk[t, inds[0], :, inds[2]]
            xxi[2] = xhijk[t, inds[0], inds[1], :]
            x = xhijk[t, inds[0], inds[1], inds[2]]
        else:
            xxi[0] = xhijk[t, :, inds[1], 0]
            xxi[1] = xhijk[t, inds[0], :, 0]
            x = xhijk[t, inds[0], inds[1], 0]
    else:
        xxi[0] = xhijk[t, :, 0, 0]
        x = xhijk[t, inds[0], 0, 0]

    ret = zeros(18)

    if not homogeneous:
        if reconstructPrim:
            p = x
        else:
            p = Cvec_to_Pvec(x, PAR, SYS)
        source_ref(ret, p, PAR, SYS)
        ret *= dx

    if SYS.viscous:
        if reconstructPrim:
            v = x[2:5]
        else:
            v = x[2:5] / x[0]
        for d in range(ndim):
            dxdxi = dot(derivs[inds[d]], xxi[d])
            if reconstructPrim:
                dxdxi = dQdPdot(x, dxdxi, PAR, SYS)
            temp = zeros(18)
            Bdot(temp, dxdxi, v, d)
            ret -= temp

    return ret

def fv_terms(xh, dt, PAR, SYS, homogeneous=0):
    """ Returns the space-time averaged interface terms, jump terms, source terms, and
        non-conservative terms
    """
    if ndim < 3:
        xh0 = xh.repeat([3], axis=2)
    if ndim < 2:
        xh0 = xh.repeat([3], axis=1)
        xh0 = xh0.repeat([3], axis=2)

    nx, ny, nz = array(xh0.shape[:3]) - 2
    xEnd = endpoints(xh0)
    xh0 = xh0.reshape([nx+2, ny+2, nz+2] + idx + [18])

    interface_func = lambda ret, xL, xM, xR, d: interface(ret, xL, xM, xR, d, PAR, SYS)
    center_func = lambda xhijk, t, inds: center(xhijk, t, inds, PAR, SYS, homogeneous)

    s = zeros([nx, ny, nz, 18])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        xhijk = xh0[i+1, j+1, k+1]
        for t, x, y, z in product(range(idx[0]),range(idx[1]),range(idx[2]),range(idx[3])):
            s[i, j, k] += weight[t,x,y,z] * center_func(xhijk, t, [x, y, z])

    if altInterfaces:
        s -= 0.5 * alternative_interfaces(xEnd, PAR, SYS)

    else:
        F = zeros([ndim, nx, ny, nz, 18])
        for i, j, k in product(range(nx), range(ny), range(nz)):
            xEndM = xEnd[:, :, i+1, j+1, k+1]
            xEndL = xEnd[:, :, i,   j+1, k+1]
            xEndR = xEnd[:, :, i+2, j+1, k+1]
            interface_func(F[0,i,j,k], xEndL, xEndM, xEndR, 0)
            if ndim > 1:
                xEndL = xEnd[:, :, i+1, j,   k+1]
                xEndR = xEnd[:, :, i+1, j+2, k+1]
                interface_func(F[1,i,j,k], xEndL, xEndM, xEndR, 1)
                if ndim > 2:
                    xEndL = xEnd[:, :, i+1, j+1, k]
                    xEndR = xEnd[:, :, i+1, j+1, k+2]
                    interface_func(F[2,i,j,k], xEndL, xEndM, xEndR, 2)

        for d in range(ndim):
            s -= F[d]

    return dt/dx * s
