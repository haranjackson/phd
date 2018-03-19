from itertools import product

from joblib import delayed
from numpy import arange, array, concatenate, dot, tensordot, zeros

from etc.grids import flat_index
from solvers.fv.fluxes import B_INT, D_OSH, D_RUS, D_ROE, RUSANOV, OSHER, ROE
from solvers.fv.matrices import WGHT, WGHT_END, TN
from solvers.basis import ENDVALS, DERVALS
from system import Bdot, source_ref, flux_ref
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


def interfaces(qEnd, dX, MP):
    """ Returns the contribution to the finite volume update coming from the
        fluxes at the interfaces
    """
    dims = [dim - 2 for dim in qEnd.shape[2:NDIM + 2]]
    nweights = len(WGHT_END)
    ret = zeros(dims + [NV])

    for d in range(NDIM):

        dimensions = [arange(1, dim + 1) for dim in dims[:d]] + \
                     [arange(1, dims[d] + 2)] + \
                     [arange(1, dim + 1) for dim in dims[d + 1:]]

        for coords in product(*dimensions):

            # qL,qR are the sets of polynomial coefficients for the DG
            # reconstruction at the left and right sides of the interface
            lcoords = (d, 1) + coords[:d] + (coords[d] - 1,) + coords[d + 1:]
            rcoords = (d, 0) + coords
            qL = qEnd[lcoords].reshape([nweights, NV])
            qR = qEnd[rcoords].reshape([nweights, NV])

            # Integrate the flux over the interface
            fEnd = zeros(NV)
            BEnd = zeros(NV)
            for ind in range(nweights):
                qL_ = qL[ind]
                qR_ = qR[ind]

                ftemp = zeros(NV)
                flux_ref(ftemp, qL_, d, MP)
                flux_ref(ftemp, qR_, d, MP)
                ftemp -= D_FUN(qL_, qR_, d, MP).real
                Btemp = B_INT(qL_, qR_, d, MP)

                fEnd += WGHT_END[ind] * ftemp / dX[d]
                BEnd += WGHT_END[ind] * Btemp / dX[d]

            rcoords_ = tuple(c - 1 for c in rcoords[2:])
            lcoords_ = rcoords_[:d] + (rcoords_[d] - 1,) + rcoords_[d + 1:]

            if lcoords_[d] >= 0:
                ret[lcoords_] -= fEnd - BEnd

            if rcoords_[d] < dims[d]:
                ret[rcoords_] += fEnd + BEnd

    return ret


def centers(qh, dX, MP, HOMOGENEOUS):
    """ Returns the space-time averaged source term and non-conservative terms
    """
    dims = tuple(s - 2 for s in qh.shape[:NDIM])
    s = zeros(dims + (NV,))

    for coords in product(*[arange(dim) for dim in dims]):

        qhi = qh[tuple(coord + 1 for coord in coords)]

        # Integrate across volume of spacetime cell (i,j,k)
        for inds in product(*[arange(s) for s in WGHT.shape]):

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
                source_ref(tmp, q, MP)

            for d in range(NDIM):
                ind = inds[d + 1]
                dxdxi = dot(DERVALS[ind], qi[d])
                temp = zeros(NV)
                Bdot(temp, dxdxi, q, d, MP)
                tmp -= temp / dX[d]

            s[coords] += WGHT[inds] * tmp

    return s


def fv_terms(qh, dt, dX, MP, HOMOGENEOUS=0):
    """ Returns the space-time averaged interface terms, jump terms,
        source terms, and non-conservative terms
    """
    qEnd = endpoints(qh)
    s = centers(qh, dX, MP, HOMOGENEOUS)
    s -= 0.5 * interfaces(qEnd, dX, MP)
    return dt * s


def fv_launcher(pool, qh, dt, dX, MP, HOMOGENEOUS=0):
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
                                        dX, MP, HOMOGENEOUS) for i in range(n))
        return concatenate(qhList)
    else:
        return fv_terms(qh, dt, dX, MP, HOMOGENEOUS)
