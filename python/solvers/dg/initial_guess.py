from itertools import product

from numpy import array, dot, zeros
from scipy.optimize import newton_krylov

from solvers.basis import DERVALS
from system import source, system
from options import N, NT, NV, NDIM, N_K_IG, DG_TOL


IDX = [N] * NDIM + [1] * (3 - NDIM)


def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(N)])
    return ret.reshape([NT, NV])


def stiff_initial_guess(w, dtGAPS, dX, MP):
    """ Returns an initial guess based on the underlying equations
    """
    q = zeros([N] + IDX + [NV])
    qt = w.reshape(IDX + [NV])

    for t in range(N):

        dt = dtGAPS[t]

        for x, y, z in product(range(IDX[0]), range(IDX[1]), range(IDX[2])):

            q_ = qt[x, y, z]
            qx = qt[:, y, z]
            qy = qt[x, :, z]
            qz = qt[x, y, :]
            qi = [qx, qy, qz]

            Mdqdx = zeros(NV)
            inds = [x, y, z]
            for d in range(NDIM):
                dqdxi = dot(DERVALS[inds[d]], qi[d])
                Mdqdx += dot(system(q_, d, MP), dqdxi) / dX[d]

            S = source(q_, MP)

            if N_K_IG:
                def f(X): return X - q_ + dt * (Mdqdx - (S + source(X, MP)) / 2)
                q[t, x, y, z] = newton_krylov(f, q_, f_tol=DG_TOL)
            else:
                q[t, x, y, z] = q_ - dt * (Mdqdx - S)

        qt = q[t]

    return q.reshape([NT, NV])
