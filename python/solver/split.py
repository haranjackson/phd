from itertools import product

from numpy import array, dot, zeros
from scipy.integrate import odeint

from ader.etc.basis import Basis, derivative


class SplitSolver():

    def __init__(self, N, NV, NDIM, F, S=None, B=None, M=None, dSdQ=None,
                 model_params=None):

        self.N = N
        self.NV = NV
        self.NDIM = NDIM
        self.F = F
        self.B = B
        self.S = S
        self.M = M
        self.dSdQ = dSdQ
        self.model_params = model_params

        basis = Basis(N)
        self.DERVALS = basis.DERVALS

    def weno_midstepper(self, wh, dt, dX):
        """ Steps the WENO reconstruction forwards by dt/2, under the homogeneous
            system
        """
        for coords in product(*[range(s) for s in wh.shape[:self.NDIM]]):

            w = wh[coords]

            if self.M is None:

                # calculate the flux at each node, in each direction
                F = [zeros(w.shape)] * self.NDIM
                for d in range(self.NDIM):
                    for inds in product(*[range(self.N)] * self.NDIM):
                        F[d][inds] = self.F(w[inds], d, self.model_params)

            for inds in product(*[range(self.N)] * self.NDIM):

                tmp = zeros(self.NV)

                # wi holds the coefficients at the nodes lying in a strip in
                # the dth direction, at the node given by inds
                for d in range(self.NDIM):
                    dwdx = derivative(self.N, self.NV, self.NDIM, w, inds, d,
                                      self.DERVALS)

                    if self.M is None:
                        dFdx = derivative(self.N, self.NV, self.NDIM, F, inds,
                                          d, self.DERVALS)
                        B = self.B(w[inds], d, self.model_params)
                        Bdwdx = dot(B, dwdx)
                        tmp += (dFdx + Bdwdx) / dX[d]

                    else:
                        M = self.M(w[inds], d, self.model_params)
                        tmp += dot(M, dwdx) / dX[d]

                w[inds] -= dt / 2 * tmp

        def f(self, y, t0):
            return self.S(y, self.model_params)

        def jac(self, y, t0):
            return self.dSdQ(y, self.model_params)

        def ode_solver_numerical(self, Q, dt):
            """ Full numerical solver for the ODE system
            """
            y0 = Q.copy()
            t = array([0, dt])

            if self.dSdQ is not None:
                Q[:] = odeint(f, y0, t, Dfun=jac)[1]
            else:
                Q[:] = odeint(f, y0, t)[1]

        def ode_launcher(self, u, dt):

            for coords in product(*[range(s) for s in u.shape[:-1]]):
                if self.ode_solver is not None:
                    self.ode_solver(u[coords], dt, self.model_params)
                else:
                    self.ode_solver_numerical(u[coords], dt)
