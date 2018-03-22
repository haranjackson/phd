from itertools import product

from numpy import dot, zeros

from etc.grids import flat_index
from solvers.basis import DERVALS
from solvers.split.analytical import ode_stepper_analytical
from solvers.split.numerical import ode_stepper_numerical
from system import nonconservative_product, flux, system_matrix
from options import NV, N, NDIM, NUM_ODE


def ode_launcher(u, dt, *args):
    if NUM_ODE:
        ode_stepper_numerical(u, dt, *args)
    else:
        ode_stepper_analytical(u, dt, *args)


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
                    flux(F[d][inds], w[inds], d, *args)

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
                    Bdwdx = zeros(NV)
                    nonconservative_product(Bdwdx, dwdx, w[inds], d, *args)
                    tmp += (dFdx + Bdwdx) / dX[d]

            w[inds] -= dt / 2 * tmp
