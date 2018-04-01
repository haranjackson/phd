from numpy import absolute, dot, prod, zeros
from scipy.linalg import solve
from scipy.optimize import newton_krylov

from solvers.dg.initial_guess import standard_initial_guess, stiff_initial_guess
from solvers.dg.matrices import galerkin_matrices


def blowup(qNew, max_size):
    """ Check whether qNew has blown up larger than max_size
    """
    return (absolute(qNew) > max_size).any()


def unconverged(q, qNew, TOL):
    """ Mixed convergence condition
    """
    return (absolute(q - qNew) > TOL * (1 + absolute(q))).any()


class DiscontinuousGalerkinSolver():

    def __init__(self, N, NV, NDIM, flux, source=None,
                 nonconservative_matrix=None, model_params=None,
                 stiff=False, stiff_ig=False, nk_ig=False,
                 tol=1e-6, max_iter=50, max_size=1e16):

        self.N = N
        self.NV = NV
        self.NDIM = NDIM
        self.NT = N**(NDIM + 1)

        self.flux = flux
        self.source = source
        self.nonconservative_matrix = nonconservative_matrix
        self.model_params = model_params

        self.max_iter = max_iter
        self.tol = tol
        self.max_size = max_size

        self.stiff = stiff
        if stiff_ig:
            self.initial_guess = stiff_initial_guess
        else:
            self.initial_guess = standard_initial_guess
        self.nk_ig = nk_ig

        self.DG_W, self.DG_V, self.DG_U, self.DG_M, self.DG_D = galerkin_matrices(
            N, NDIM, NV)

    def rhs(self, q, Ww, dt, dX):
        """ Returns the right-handside of the system governing coefficients of qh
        """
        ret = zeros([self.NT, self.NV])

        # Dq[d,i]: dq/dx at position i in direction d
        # Fq[d,i]: F(q) at position i in direction d
        # Bq[d,i]: B.dq/dx at position i in direction d
        Dq = dot(self.DG_D, q)
        Fq = zeros([self.NDIM, self.NT, self.NV])
        Bq = zeros([self.NDIM, self.NT, self.NV])

        for i in range(self.NT):
            qi = q[i]

            if self.source is not None:
                ret[i] = self.source(qi, self.model_params)

            for d in range(self.NDIM):
                Fq[d, i] = self.flux(qi, d, self.model_params)

                if self.nonconservative_matrix is not None:
                    B = self.nonconservative_matrix(qi, d, self.model_params)
                    Bq[d, i] = dot(B, Dq[d, i])

        if self.nonconservative_matrix is not None:
            for d in range(self.NDIM):
                ret -= Bq[d] / dX[d]

        ret *= self.DG_M
        for d in range(self.NDIM):
            ret -= dot(self.DG_V[d], Fq[d]) / dX[d]

        return dt * ret + Ww

    def root_find(self, w, Ww, dt, dX):
        """ Finds DG coefficients with Newton-Krylov, if iteration has failed
        """
        q = self.initial_guess(self, w, dt, dX)

        def f(x): return dot(self.DG_U, x) - self.rhs(x, Ww, dt, dX)
        return newton_krylov(f, q, f_tol=self.tol, method='bicgstab')

    def solve(self, wh, dt, dX):
        """ Returns the Galerkin predictor, given the WENO reconstruction at tn
        """
        shape = wh.shape
        n = prod(shape[:self.NDIM])
        wh = wh.reshape(n, self.N**self.NDIM, self.NV)
        qh = zeros([n, self.NT, self.NV])

        for i in range(n):

            w = wh[i]
            Ww = dot(self.DG_W, w)

            if self.stiff:
                qh[i] = self.root_find(w, Ww, dt, dX)

            else:
                q = self.initial_guess(self, w, dt, dX)

                for count in range(self.max_iter):

                    qNew = solve(self.DG_U, self.rhs(q, Ww, dt, dX))

                    if blowup(qNew, self.max_size):
                        qh[i] = self.root_find(w, Ww, dt, dX)
                        break

                    elif unconverged(q, qNew, self.tol):
                        q = qNew
                        continue

                    else:
                        qh[i] = qNew
                        break
                else:
                    qh[i] = self.root_find(w, Ww, dt, dX)

        return qh.reshape(shape[:self.NDIM] + (self.N,) * (self.NDIM + 1) + (self.NV,))
