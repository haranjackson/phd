import GPRpy

from numpy import array, int32, zeros

from etc.save import Data
from options import N, NDIM, NV, STRANG, HALF_STEP, FLUX, STIFF, SPLIT, CFL


def get_dimensions(arr):
    return array(arr.shape[:NDIM] + (1,)*(3-NDIM), dtype=int32)


def cpp_split_stepper(mat, matBC, dt, dX, *args):

    nX = get_dimensions(mat)
    extDims = GPRpy.solvers.extended_dimensions(nX, 1)
    wh = zeros(extDims * int(pow(N, NDIM)) * NV)

    matr = mat.ravel()
    matBCr = matBC.ravel()

    GPRpy.solvers.split_stepper(matr, matBCr, wh, NDIM, nX, dt, dX,
                                STRANG, HALF_STEP, FLUX, *args)

    mat = matr.reshape(mat.shape)


def cpp_ader_stepper(mat, matBC, dt, dX, *args):

    nX = get_dimensions(mat)
    extDims = GPRpy.solvers.extended_dimensions(nX, 1)
    wh = zeros(extDims * int(pow(N, NDIM)) * NV)
    qh = zeros(extDims * int(pow(N, NDIM + 1)) * NV)

    matr = mat.ravel()
    matBCr = matBC.ravel()

    GPRpy.solvers.ader_stepper(matr, matBCr, wh, qh, NDIM, nX, dt, dX,
                               STIFF, FLUX, *args)

    mat = matr.reshape(mat.shape)


def run_cpp(u, dX, t, tf, count, data, *args):
    MPs = args[0]
    nX = get_dimensions(u)
    u1 = u.ravel()
    GPRpy.solvers.iterator(u1, tf - t, nX, dX, CFL, False,
                           SPLIT, STRANG, HALF_STEP, STIFF, FLUX, MPs[0])
    data.append(Data(u1.reshape(u.shape), t))
