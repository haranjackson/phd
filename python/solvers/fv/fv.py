from itertools import product

from joblib import delayed
from numpy import array, concatenate, dot, einsum, tensordot, zeros

from solvers.fv.fluxes import input_vectors, Bint, Aint, Smax
from solvers.basis import quad, end_values, derivative_values
from gpr.matrices.conserved import Bdot, source_ref, flux_ref
from gpr.matrices.jacobians import dQdPdot
from gpr.variables.vectors import Cvec_to_Pvec
from options import ndim, dx, N1, method, approxInterface, reconstructPrim, timeDim, paraFV, ncore


nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()

if method == 'osher':
    s_func = Aint
elif method == 'rusanov':
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

def interfaces(xEnd, PAR, SYS):
    nx, ny, nz = xEnd.shape[2:5]
    fEnd = zeros([ndim, nx-1, ny-1, nz-1, 18])
    BEnd = zeros([ndim, nx-1, ny-1, nz-1, 18])

    inpt_lam = lambda xL, xR: input_vectors(xL, xR, PAR, SYS)
    flux_lam = lambda ftemp, p, E, d: flux_ref(ftemp, p, E, d, PAR, SYS)
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
                EL = qL[1] / qL[0]
                ER = qR[1] / qR[0]

                ftemp = zeros(18)
                weight0 = weightEnd[t, x1, x2]

                flux_lam(ftemp, pL, EL, d)
                flux_lam(ftemp, pR, ER, d)
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
    nx,ny,nz = xh.shape[:3]
    if nx==1:
        xh0 = xh.repeat([3], axis=2)
    if ny==1:
        xh0 = xh.repeat([3], axis=1)
        xh0 = xh0.repeat([3], axis=2)

    nx, ny, nz = array(xh0.shape[:3]) - 2
    xEnd = endpoints(xh0)
    xh0 = xh0.reshape([nx+2, ny+2, nz+2] + idx + [18])

    center_func = lambda xhijk, t, inds: center(xhijk, t, inds, PAR, SYS, homogeneous)

    s = zeros([nx, ny, nz, 18])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        xhijk = xh0[i+1, j+1, k+1]
        for t, x, y, z in product(range(idx[0]),range(idx[1]),range(idx[2]),range(idx[3])):
            s[i, j, k] += weight[t,x,y,z] * center_func(xhijk, t, [x, y, z])

    s -= 0.5 * interfaces(xEnd, PAR, SYS)

    return dt/dx * s

def fv_launcher(pool, qh, dt, PAR, SYS, homogeneous=0):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    if paraFV:
        nx = qh.shape[0]
        step = int(nx / ncore)
        chunk = array([i*step for i in range(ncore)] + [nx+1])
        chunk[0] += 1
        chunk[-1] -= 1
        n = len(chunk) - 1
        qhList = pool(delayed(fv_terms)(qh[chunk[i]-1:chunk[i+1]+1], dt, PAR, SYS, homogeneous)
                                       for i in range(n))
        return concatenate(qhList)
    else:
        return fv_terms(qh, dt, PAR, SYS, homogeneous)
