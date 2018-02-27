from itertools import product

from joblib import delayed
from numpy import array, concatenate, dot, einsum, tensordot, zeros

from solvers.fv.fluxes import Bint, D_OSH, D_RUS, D_ROE, RUSANOV, OSHER, ROE
from solvers.basis import WGHTS, ENDVALS, DERVALS
from system import Bdot, source_ref, flux_ref
from options import NDIM, NV, N, FLUX, SPLIT, PARA_FV, NCORE


if FLUX == RUSANOV:
    D_FUN = D_RUS
elif FLUX == ROE:
    D_FUN = D_ROE
elif FLUX == OSHER:
    D_FUN = D_OSH

TWGHTS = array([1]) if SPLIT else WGHTS
TN = len(TWGHTS)

IDX = [N] * NDIM + [1] * (3 - NDIM)
WGHT_LIST = [WGHTS] * NDIM + [array([1])] * (3 - NDIM)

WGHT = einsum('t,x,y,z', TWGHTS, WGHT_LIST[0], WGHT_LIST[1], WGHT_LIST[2])
WGHT_END = einsum('t,x,y', TWGHTS, WGHT_LIST[1], WGHT_LIST[2])


def endpoints(qh0):
    """ Returns tensor T where T[d,e,i,j,k] is the set of DG coefficients at end
        e (either 0 or 1) in the dth direction, in cell (i,j,k)
    """
    return array([tensordot(ENDVALS, qh0, (0, 4 + d)) for d in range(NDIM)])


def interfaces(qEnd, dX, MP):
    nx, ny, nz = qEnd.shape[2:5]
    fEnd = zeros([NDIM, nx - 1, ny - 1, nz - 1, NV])
    BEnd = zeros([NDIM, nx - 1, ny - 1, nz - 1, NV])

    for d in range(NDIM):
        for i, j, k in product(range(nx - 1), range(ny - 1), range(nz - 1)):

            qL = qEnd[d, 1, i, j, k]
            if d == 0:
                qR = qEnd[d, 0, i + 1, j, k]
            elif d == 1:
                qR = qEnd[d, 0, i, j + 1, k]
            else:
                qR = qEnd[d, 0, i, j, k + 1]

            fEndTemp = zeros(NV)
            BEndTemp = zeros(NV)
            for t, x1, x2 in product(range(TN), range(IDX[1]), range(IDX[2])):
                qL_ = qL[t, x1, x2]
                qR_ = qR[t, x1, x2]

                ftemp = zeros(NV)
                flux_ref(ftemp, qL_, d, MP)
                flux_ref(ftemp, qR_, d, MP)
                ftemp -= D_FUN(qL_, qR_, d, MP).real
                fEndTemp += WGHT_END[t, x1, x2] * ftemp
                BEndTemp += WGHT_END[t, x1, x2] * Bint(qL_, qR_, d, MP)

            fEnd[d, i, j, k] = fEndTemp / dX[d]
            BEnd[d, i, j, k] = BEndTemp / dX[d]

    ret = zeros([nx - 2, ny - 2, nz - 2, NV])
    ret -= fEnd[0, :-1, 1:, 1:]
    ret += fEnd[0, 1:,  1:, 1:]
    ret += BEnd[0, :-1, 1:, 1:]
    ret += BEnd[0, 1:,  1:, 1:]
    if NDIM > 1:
        ret -= fEnd[1, 1:, :-1, 1:]
        ret += fEnd[1, 1:,  1:, 1:]
        ret += BEnd[1, 1:, :-1, 1:]
        ret += BEnd[1, 1:,  1:, 1:]
    if NDIM > 2:
        ret -= fEnd[2, 1:, 1:, :-1]
        ret += fEnd[2, 1:, 1:,  1:]
        ret += BEnd[2, 1:, 1:, :-1]
        ret += BEnd[2, 1:, 1:,  1:]
    return ret


def centers(qh0, nx, ny, nz, dX, MP, HOMOGENEOUS):
    """ Returns the space-time averaged source term and non-conservative terms
    """
    s = zeros([nx, ny, nz, NV])

    for i, j, k in product(range(nx), range(ny), range(nz)):

        qhi = qh0[i + 1, j + 1, k + 1]

        for t, x, y, z in product(range(TN), range(IDX[0]),
                                  range(IDX[1]), range(IDX[2])):

            q = qhi[t, x, y, z]
            qx = qhi[t, :, y, z]
            qy = qhi[t, x, :, z]
            qz = qhi[t, x, y, :]
            qi = [qx, qy, qz]

            tmp = zeros(NV)

            if not HOMOGENEOUS:
                source_ref(tmp, q, MP)

            inds = [x, y, z]
            for d in range(NDIM):
                dxdxi = dot(DERVALS[inds[d]], qi[d])
                temp = zeros(NV)
                Bdot(temp, dxdxi, q, d, MP)
                tmp -= temp / dX[d]

            s[i, j, k] += WGHT[t, x, y, z] * tmp

    return s


def extend_dimensions(qh):
    """ If the simulation is 1D or 2D, extends the array in the unused
        dimensions so that the 3D solver can be used
    """
    nx, ny, nz = qh.shape[:3]
    if nz == 1:
        qh0 = qh.repeat([3], axis=2)
    if ny == 1:
        qh0 = qh0.repeat([3], axis=1)
    nx, ny, nz = array(qh0.shape[:3]) - 2

    qh0 = qh0.reshape([nx + 2, ny + 2, nz + 2, TN] + IDX + [NV])

    return qh0, nx, ny, nz


def fv_terms(qh, dt, dX, MP, HOMOGENEOUS=0):
    """ Returns the space-time averaged interface terms, jump terms,
        source terms, and non-conservative terms
    """
    qh0, nx, ny, nz = extend_dimensions(qh)
    qEnd = endpoints(qh0)

    s = centers(qh0, nx, ny, nz, dX, MP, HOMOGENEOUS)
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
