import GPRpy

from numpy import array, dot, int32, zeros

from gpr.misc.structures import Cvec_to_Pclass
from gpr.systems.eigenvalues import Xi1, Xi2

from solvers.fv.fluxes import B_INT, D_OSH, D_ROE, D_RUS
from solvers.fv.fv import interfaces, endpoints, fv_terms, centers

from bindings_tests.test_functions import check, generate_vector

from options import SPLIT, FLUX, PERR_FROB, NV, NDIM


### FLUXES ###


def TAT_test(d, MP):
    Q = generate_vector(MP)
    P = Cvec_to_Pclass(Q, MP)

    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    TAT_py = dot(Ξ1, Ξ2)
    TAT_cp = GPRpy.system.thermo_acoustic_tensor(Q, d, MP)

    print("TAT   ", check(TAT_cp, TAT_py))
    return TAT_cp, TAT_py


def Bint_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    Bint_cp = GPRpy.solvers.fv.Bint(Q1, Q2, d, MP)
    Bint_py = B_INT(Q1, Q2, d, MP)

    print("Bint  ", check(Bint_cp, Bint_py))
    return Bint_cp, Bint_py


def D_RUS_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    D_RUS_cp = GPRpy.solvers.fv.D_RUS(Q1, Q2, d, PERR_FROB, MP)
    D_RUS_py = -D_RUS(Q1, Q2, d, MP)

    print("D_RUS ", check(D_RUS_cp, D_RUS_py))
    return D_RUS_cp, D_RUS_py


def D_ROE_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    D_ROE_cp = GPRpy.solvers.fv.D_ROE(Q1, Q2, d, MP)
    D_ROE_py = -D_ROE(Q1, Q2, d, MP)

    print("D_ROE ", check(D_ROE_cp, D_ROE_py))
    return D_ROE_cp, D_ROE_py


def D_OSH_test(d, MP):
    Q1 = generate_vector(MP)
    Q2 = generate_vector(MP)
    D_OSH_cp = GPRpy.solvers.fv.D_OSH(Q1, Q2, d, MP)
    D_OSH_py = -D_OSH(Q1, Q2, d, MP)

    print("D_OSH ", check(D_OSH_cp, D_OSH_py))
    return D_OSH_cp, D_OSH_py

### FINITE VOLUME (NDIM=1) ###


HOMOGENEOUS = SPLIT
TIME = not SPLIT
SOURCES = not SPLIT


def FVc_test(qh_py, dX, dt, MP):

    if HOMOGENEOUS:
        shape = qh_py.shape
        qh_py = qh_py.reshape(shape[:NDIM] + (1,) + shape[NDIM:])

    nx, ny = qh_py.shape[:2]

    if NDIM == 1:
        FVc_py = zeros([nx - 2, NV])
        FVc_cp = zeros((nx - 2) * NV)
        GPRpy.solvers.fv.centers1(FVc_cp, qh_py.ravel(), nx - 2, dt, dX[0],
                                  SOURCES, TIME, MP)

    elif NDIM == 2:
        FVc_py = zeros([nx - 2, ny - 2, NV])
        FVc_cp = zeros((nx - 2) * (ny - 2) * NV)
        GPRpy.solvers.fv.centers2(FVc_cp, qh_py.ravel(), nx - 2, ny - 2, dt,
                                  dX[0], dX[1], SOURCES, TIME, MP)

    centers(FVc_py, qh_py, dX, HOMOGENEOUS, MP)
    FVc_py *= dt

    FVc_cp = FVc_cp.reshape(FVc_py.shape)
    print("FVc   ", check(FVc_cp, FVc_py))
    return FVc_cp, FVc_py


def FVi_test(qh_py, dX, dt, MP):

    if HOMOGENEOUS:
        shape = qh_py.shape
        qh_py = qh_py.reshape(shape[:NDIM] + (1,) + shape[NDIM:])

    nx, ny = qh_py.shape[:2]
    qEnd = endpoints(qh_py)

    if NDIM == 1:
        FVi_py = zeros([nx - 2, NV])

        FVi_cp = zeros((nx - 2) * NV)
        GPRpy.solvers.fv.interfs1(FVi_cp, qh_py.ravel(), nx - 2, dt, dX[0],
                                  TIME, FLUX, PERR_FROB, MP)

    elif NDIM == 2:
        FVi_py = zeros([nx - 2, ny - 2, NV])

        FVi_cp = zeros((nx - 2) * (ny - 2) * NV)
        GPRpy.solvers.fv.interfs2(FVi_cp, qh_py.ravel(), nx - 2, ny - 2, dt,
                                  dX[0], dX[1], TIME, FLUX, PERR_FROB, MP)

    interfaces(FVi_py, qEnd, dX, MP)
    FVi_py *= dt

    FVi_cp = FVi_cp.reshape(FVi_py.shape)
    print("FVi   ", check(FVi_cp, FVi_py))
    return FVi_cp, FVi_py


def FV_test(qh_py, dX, dt, MP):

    nx, ny = qh_py.shape[:2]

    if NDIM == 1:
        FV_cp = zeros((nx - 2) * NV)
        nX = array([nx - 2, 1, 1], dtype=int32)

    elif NDIM == 2:
        FV_cp = zeros((nx - 2) * (ny - 2) * NV)
        nX = array([nx - 2, ny - 2, 1], dtype=int32)

    FV_py = fv_terms(qh_py, dt, dX, HOMOGENEOUS, MP)

    GPRpy.solvers.fv.fv_launcher(FV_cp, qh_py.ravel(), NDIM, nX, dt, dX,
                                 SOURCES, TIME, FLUX, PERR_FROB, MP)

    FV_cp = FV_cp.reshape(FV_py.shape)
    print("FV    ", check(FV_cp, FV_py))
    return FV_cp, FV_py
