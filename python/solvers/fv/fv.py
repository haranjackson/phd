from itertools import product

from numpy import array, dot, tensordot, zeros

from etc.grids import flat_index
from solvers.fv.fluxes import B_INT, D_OSH, D_RUS, D_ROE
from solvers.fv.matrices import quad_weights
from solvers.basis import ENDVALS, DERVALS


def endpoints(qh, NDIM):
    """ Returns tensor T where T[d,e,i1,...,in] is the set of DG coefficients in
        the dth direction, at end e (either 0 or 1), in cell (i1,...,in)
    """
    return array([tensordot(ENDVALS, qh, (0, NDIM + 1 + d))
                  for d in range(NDIM)])


class FiniteVolumeSolver():

    def __init__(self, N, NV, NDIM, flux, source=None,
                 nonconservative_matrix=None, system_matrix=None, max_eig=None,
                 model_params=None, riemann_solver='rusanov', time_rec=True):

        self.N = N
        self.NV = NV
        self.NDIM = NDIM

        self.flux = flux
        self.source = source
        self.nonconservative_matrix = nonconservative_matrix
        self.system_matrix = system_matrix
        self.max_eig = max_eig
        self.model_params = model_params

        if riemann_solver == 'rusanov':
            self.D_FUN = D_RUS
        elif riemann_solver == 'roe':
            self.D_FUN = D_ROE
        elif riemann_solver == 'osher':
            self.D_FUN = D_OSH
        else:
            raise ValueError("Choice of 'riemann_solver' choice not recognised.\n" +
                             "Choose from 'rusanov', 'roe', and 'osher'.")

        self.time_rec = time_rec
        self.TN, self.WGHT, self.WGHT_END = quad_weights(N, NDIM, time_rec)

    def interfaces(self, ret, qEnd, dX):
        """ Returns the contribution to the finite volume update coming from the
            fluxes at the interfaces
        """
        dims = ret.shape[:self.NDIM]
        nweights = len(self.WGHT_END)

        for d in range(self.NDIM):

            # dimensions of cells traversed when calculating fluxes in direction d
            dimensions = [range(1, dim + 1) for dim in dims[:d]] + \
                         [range(1, dims[d] + 2)] + \
                         [range(1, dim + 1) for dim in dims[d + 1:]]

            for coords in product(*dimensions):

                # qL,qR are the sets of polynomial coefficients for the DG
                # reconstruction at the left and right sides of the interface
                lcoords = (d, 1) + coords[:d] + \
                    (coords[d] - 1,) + coords[d + 1:]
                rcoords = (d, 0) + coords
                qL = qEnd[lcoords].reshape(nweights, self.NV)
                qR = qEnd[rcoords].reshape(nweights, self.NV)

                # integrate the flux over the surface normal to direction d
                fInt = zeros(self.NV)    # flux from conservative terms
                BInt = zeros(self.NV)    # flux from non-conservative terms
                for ind in range(nweights):
                    qL_ = qL[ind]
                    qR_ = qR[ind]

                    fL = self.flux(qL_, d, self.model_params)
                    fR = self.flux(qR_, d, self.model_params)

                    fInt += self.WGHT_END[ind] * (fL + fR - self.D_FUN(self, qL_, qR_, d))

                    if self.nonconservative_matrix is not None:
                        BInt += self.WGHT_END[ind] * B_INT(self, qL_, qR_, d)

                rcoords_ = tuple(c - 1 for c in rcoords[2:])
                lcoords_ = rcoords_[:d] + (rcoords_[d] - 1,) + rcoords_[d + 1:]

                if lcoords_[d] >= 0:
                    ret[lcoords_] -= (fInt + BInt) / dX[d]

                if rcoords_[d] < dims[d]:
                    ret[rcoords_] += (fInt - BInt) / dX[d]

    def centers(self, ret, qh, dX):
        """ Returns the space-time averaged source term and non-conservative terms
        """
        for coords in product(*[range(dim) for dim in ret.shape[:self.NDIM]]):

            qhi = qh[tuple(coord + 1 for coord in coords)]

            # Integrate across volume of spacetime cell
            for inds in product(*[range(s) for s in self.WGHT.shape]):

                q = qhi[inds]
                qi = []
                for d in range(self.NDIM):
                    t = inds[0]
                    i = flat_index(inds[1:d])
                    j = flat_index(inds[d + 2:])
                    qhi_r = qhi.reshape(self.TN, self.N**d, self.N,
                                        self.N**(self.NDIM - d - 1), self.NV)
                    qi.append(qhi_r[t, i, :, j])

                tmp = zeros(self.NV)

                if self.source is not None:
                    tmp = self.source(q, self.model_params)

                if self.nonconservative_matrix is not None:

                    for d in range(self.NDIM):
                        ind = inds[d + 1]
                        # derivative of q in direction d
                        dqdx = dot(DERVALS[ind], qi[d])

                        B = self.nonconservative_matrix(q, d, self.model_params)
                        Bdqdx = dot(B, dqdx)

                        tmp -= Bdqdx / dX[d]

                ret[coords] += self.WGHT[inds] * tmp

    def solve(self, qh, dt, dX):
        """ Returns the space-time averaged interface terms, jump terms,
            source terms, and non-conservative terms
        """
        if not self.time_rec:
            qh = qh.reshape(qh.shape[:self.NDIM] + (1,) + qh.shape[self.NDIM:])

        qEnd = endpoints(qh, self.NDIM)

        ret = zeros([s - 2 for s in qh.shape[:self.NDIM]] + [self.NV])

        if self.source is not None or self.nonconservative_matrix is not None:
            self.centers(ret, qh, dX)

        self.interfaces(ret, qEnd, dX)

        return dt * ret
