from itertools import product

from joblib import delayed
from numpy import array, concatenate, dot, tensordot, zeros

from etc.grids import flat_index
from solvers.fv.fluxes import B_INT, D_OSH, D_RUS, D_ROE, RUSANOV, OSHER, ROE
from solvers.fv.matrices import WGHT, WGHT_END, TN
from solvers.basis import ENDVALS, DERVALS
from system import nonconservative_matrix, source, flux
from options import NDIM, NV, FLUX, PARA_FV, NCORE, N


if FLUX == RUSANOV:
    D_FUN = D_RUS
elif FLUX == ROE:
    D_FUN = D_ROE
elif FLUX == OSHER:
    D_FUN = D_OSH


def endpoints(qh):
    """ Returns tensor T where T[d,e,i1,...,in] is the set of DG coefficients in
        the dth direction, at end e (either 0 or 1), in cell (i1,...,in)
    """
    return array([tensordot(ENDVALS, qh, (0, NDIM + 1 + d))
                  for d in range(NDIM)])


def interfaces(ret, qEnd, dX, *args):
    """ Returns the contribution to the finite volume update coming from the
        fluxes at the interfaces
    """
    dims = ret.shape[:NDIM]
    nweights = len(WGHT_END)

    for d in range(NDIM):

        # dimensions of cells traversed when calculating fluxes in direction d
        dimensions = [range(1, dim + 1) for dim in dims[:d]] + \
                     [range(1, dims[d] + 2)] + \
                     [range(1, dim + 1) for dim in dims[d + 1:]]

        for coords in product(*dimensions):

            # qL,qR are the sets of polynomial coefficients for the DG
            # reconstruction at the left and right sides of the interface
            lcoords = (d, 1) + coords[:d] + (coords[d] - 1,) + coords[d + 1:]
            rcoords = (d, 0) + coords
            qL = qEnd[lcoords].reshape([nweights, NV])
            qR = qEnd[rcoords].reshape([nweights, NV])

            # integrate the flux over the surface normal to direction d
            fInt = zeros(NV)    # flux from conservative terms
            BInt = zeros(NV)    # flux from non-conservative terms
            for ind in range(nweights):
                qL_ = qL[ind]
                qR_ = qR[ind]

                fL = flux(qL_, d, *args)
                fR = flux(qR_, d, *args)

                fInt += WGHT_END[ind] * (fL + fR - D_FUN(qL_, qR_, d, *args))
                BInt += WGHT_END[ind] * B_INT(qL_, qR_, d, *args)

            rcoords_ = tuple(c - 1 for c in rcoords[2:])
            lcoords_ = rcoords_[:d] + (rcoords_[d] - 1,) + rcoords_[d + 1:]

            if lcoords_[d] >= 0:
                ret[lcoords_] -= (fInt + BInt) / dX[d]

            if rcoords_[d] < dims[d]:
                ret[rcoords_] += (fInt - BInt) / dX[d]


def centers(ret, qh, dX, HOMOGENEOUS, *args):
    """ Returns the space-time averaged source term and non-conservative terms
    """
    for coords in product(*[range(dim) for dim in ret.shape[:NDIM]]):

        qhi = qh[tuple(coord + 1 for coord in coords)]

        # Integrate across volume of spacetime cell
        for inds in product(*[range(s) for s in WGHT.shape]):

            q = qhi[inds]
            qi = []
            for d in range(NDIM):
                t = inds[0]
                i = flat_index(inds[1:d])
                j = flat_index(inds[d + 2:])
                qhi_r = qhi.reshape([TN, N**d, N, N**(NDIM - d - 1), NV])
                qi.append(qhi_r[t, i, :, j])

            tmp = zeros(NV)

            if not HOMOGENEOUS:
                tmp = source(q, *args)

            for d in range(NDIM):
                ind = inds[d + 1]
                dqdx = dot(DERVALS[ind], qi[d]) # derivative of q in direction d

                B = nonconservative_matrix(q, d, *args)
                Bdqdx = dot(B, dqdx)

                tmp -= Bdqdx / dX[d]

            ret[coords] += WGHT[inds] * tmp


def fv_terms(qh, dt, dX, HOMOGENEOUS, *args):
    """ Returns the space-time averaged interface terms, jump terms,
        source terms, and non-conservative terms
    """
    if HOMOGENEOUS:
        qh = qh.reshape(qh.shape[:NDIM] + (1,) + qh.shape[NDIM:])

    dims = [s - 2 for s in qh.shape[:NDIM]]
    qEnd = endpoints(qh)

    ret = zeros(dims + [NV])
    centers(ret, qh, dX, HOMOGENEOUS, *args)
    interfaces(ret, qEnd, dX, *args)

    return dt * ret


def fv_launcher(pool, qh, dt, dX, HOMOGENEOUS, *args):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    if PARA_FV:

        nx = qh.shape[0]
        step = int(nx / NCORE)
        chunk = array([i * step for i in range(NCORE)] + [nx + 1])
        chunk[0] += 1
        chunk[-1] -= 1
        n = len(chunk) - 1

        qhList = pool(delayed(fv_terms)(qh[chunk[i] - 1:chunk[i + 1] + 1], dt,
                                        dX, HOMOGENEOUS, *args) for i in range(n))
        return concatenate(qhList)
    else:
        return fv_terms(qh, dt, dX, HOMOGENEOUS, *args)
