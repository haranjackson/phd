from numpy import tensordot, zeros

from solvers.basis import basis_polys, end_values

from options import N1, ndim


def derivative_end_values():
    """ Returns the value of the derivative of the ith basis function at 0 and 1
    """
    _, psiDer, _ = basis_polys()
    ret = zeros([N1, 2])
    for i in range(N1):
        ret[i,0] = psiDer[1][i](0)
        ret[i,1] = psiDer[1][i](1)
    return ret

def weno_endpoints(wh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        qEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    endVals = end_values()
    derEndVals = derivative_end_values()
    nx, ny, nz = wh.shape[:3]
    wh0 = wh.reshape([nx, ny, nz] + [N1]*ndim + [18])
    wEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*(ndim-1) + [18])
    wDerEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*(ndim-1) + [18])
    for d in range(ndim):
        wEnd[d] = tensordot(endVals, wh0, (0,3+d))
        wDerEnd[d] = tensordot(derEndVals, wh0, (0,3+d))
    return wEnd, wDerEnd
