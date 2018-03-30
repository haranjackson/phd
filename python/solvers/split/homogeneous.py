from itertools import product

from numpy import dot, zeros

from etc.grids import flat_index
from solvers.basis import DERVALS
from system import nonconservative_matrix, flux, system_matrix
from options import NV, N, NDIM


def weno_midstepper(wh, dt, dX, *args):
    """ Steps the WENO reconstruction forwards by dt/2, under the homogeneous system
    """
    USE_JACOBIAN = 0

    for coords in product(*[range(s) for s in wh.shape[:NDIM]]):

        w = wh[coords]

        if not USE_JACOBIAN:

            # calculate the flux at each node, in each direction
            F = [zeros([N] * NDIM + [NV])] * NDIM
            for d in range(NDIM):
                for inds in product(*[range(N)] * NDIM):
                    F[d][inds] = flux(w[inds], d, *args)

        for inds in product(*[range(N)] * NDIM):

            tmp = zeros(NV)

            # wi holds the coefficients at the nodes lying in a strip in the
            # dth direction, at the node given by inds
            for d in range(NDIM):
                i = flat_index(inds[:d])
                j = flat_index(inds[d + 1:])
                wi = w.reshape([N**d, N, N**(NDIM - d - 1), NV])[i, :, j]
                dwdx = dot(DERVALS[inds[d]], wi)

                if USE_JACOBIAN:
                    M = system_matrix(w[inds], d, *args)
                    tmp += dot(M, dwdx) / dX[d]

                else:
                    Fi = F[d].reshape([N**d, N, N**(NDIM - d - 1), NV])[i, :, j]
                    dFdx = dot(DERVALS[inds[d]], Fi)
                    B = nonconservative_matrix(w[inds], d, *args)
                    Bdwdx = dot(B, dwdx)
                    tmp += (dFdx + Bdwdx) / dX[d]

            w[inds] -= dt / 2 * tmp
