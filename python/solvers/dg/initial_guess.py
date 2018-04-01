from itertools import product

from numpy import array, dot, zeros
from scipy.optimize import newton_krylov

from etc.grids import flat_index
from solvers.basis import DERVALS, GAPS
from system import source, system_matrix


def standard_initial_guess(obj, w, *args):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(obj.N)])
    return ret.reshape(obj.NT, obj.NV)


def stiff_initial_guess(obj, w, dt, dX):
    """ Returns an initial guess based on the underlying equations
    """
    q = zeros([obj.N] * (obj.NDIM + 1) + [obj.NV])
    qt = w.reshape([obj.N] * obj.NDIM + [obj.NV])
    coordList = [range(obj.N)] * obj.NDIM

    for t in range(obj.N):

        dt_ = dt * GAPS[t]

        # loop over the coordinates of each spatial node
        for coords in product(*coordList):

            q_ = qt[coords]     # the value of q at the current spatial node

            # qi[d] holds the coefficients at the nodes lying in a strip in the
            # dth direction, at the current spatial node
            qi = []
            for d in range(obj.NDIM):
                i = flat_index(coords[:d])
                j = flat_index(coords[d + 1:])
                qi.append(qt.reshape(obj.N**d, obj.N, obj.N**(obj.NDIM - d - 1), obj.NV)[i, :, j])

            Mdqdx = zeros(obj.NV)
            for d in range(obj.NDIM):
                dqdxi = dot(DERVALS[coords[d]], qi[d])
                Mdqdx += dot(system_matrix(q_, d, obj.model_params), dqdxi) / dX[d]

            S0 = source(q_, obj.model_params)

            if obj.nk_ig:

                def f(X):

                    S = (S0 + source(X, obj.model_params)) / 2
                    return X - q_ + dt_ * (Mdqdx - S)

                q[(t,) + coords] = newton_krylov(f, q_, f_tol=obj.tol)

            else:
                q[(t,) + coords] = q_ - dt_ * (Mdqdx - S0)

        qt = q[t]

    return q.reshape(obj.NT, obj.NV)
