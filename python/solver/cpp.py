import GPRpy

from numpy import array, int32


def get_dimensions(arr):
    return array(arr.shape[:arr.ndim - 1], dtype=int32)


def cpp_split_stepper(obj, mat, matBC, dt, dX, maskBC):

    nX = get_dimensions(mat)
    matr = mat.ravel()
    matBCr = matBC.ravel()

    GPRpy.solvers.split_stepper(matr, matBCr, nX, dt, array(dX), obj.half_step,
                                obj.flux_type, obj.pars, maskBC.ravel())

    mat = matr.reshape(mat.shape)


def cpp_ader_stepper(obj, mat, matBC, dt, dX, maskBC):

    nX = get_dimensions(mat)
    matr = mat.ravel()
    matBCr = matBC.ravel()

    GPRpy.solvers.ader_stepper(matr, matBCr, nX, dt, array(dX), obj.stiff_dg,
                               obj.flux_type, obj.pars, maskBC.ravel())

    mat = matr.reshape(mat.shape)


def solve_full_cpp(obj, initial_grid, final_time, dX, cfl):

    nX = get_dimensions(initial_grid)
    u = initial_grid.ravel()

    GPRpy.solvers.iterator(u, final_time, nX, array(dX), cfl, False, obj.split,
                           obj.half_step, obj.stiff_dg, obj.flux_type,
                           obj.pars)
    return u.reshape(initial_grid.shape)
