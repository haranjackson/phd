import GPRpy

from numpy import array, int32


def get_dimensions(arr):
    return array(arr.shape[:arr.ndim - 1], dtype=int32)


def cpp_split_stepper(obj, mat, matBC, dt, dX, maskBC):

    nX = get_dimensions(mat)
    matr = mat.ravel()
    matBCr = matBC.ravel()

    # GPRpy.solvers.split_stepper(matr, matBCr, nX, dt, array(dX), obj.half_step,
    #                            obj.flux_type, obj.pars, maskBC.ravel())

    GPRpy.solvers.split_stepper_para(matr, matBCr, nX, dt, array(dX), obj.half_step,
                                     obj.flux_type, obj.pars, maskBC.ravel())

    mat = matr.reshape(mat.shape)


def cpp_ader_stepper(obj, mat, matBC, dt, dX, maskBC):

    nX = get_dimensions(mat)
    matr = mat.ravel()
    matBCr = matBC.ravel()

    # GPRpy.solvers.ader_stepper(matr, matBCr, nX, dt, array(dX), obj.stiff_dg,
    #                           obj.flux_type, obj.pars, maskBC.ravel())

    GPRpy.solvers.ader_stepper_para(matr, matBCr, nX, dt, array(dX), obj.stiff_dg,
                                    obj.flux_type, obj.pars, maskBC.ravel())

    mat = matr.reshape(mat.shape)


def solve_full_cpp(obj, initial_grid, final_time, dX, cfl, nOut, callback, bcs):

    ndim = initial_grid.ndim - 1
    if bcs == 'transitive':
        bcs = array([0] * ndim, dtype=int32)
    elif bcs == 'periodic':
        bcs = array([1] * ndim, dtype=int32)
    elif bcs == 'slip':
        bcs = array([2] * ndim, dtype=int32)
    elif bcs == 'stick':
        bcs = array([3] * ndim, dtype=int32)

    nX = get_dimensions(initial_grid)
    u = initial_grid.ravel()

    uOut = GPRpy.solvers.iterator(u, final_time, nX, array(dX), cfl, bcs,
                                  obj.split, obj.half_step, obj.stiff_dg,
                                  obj.flux_type, obj.pars, nOut)

    if callback is not None:
        shape = initial_grid.shape
        for out, count in zip(uOut, range(nOut)):
            callback(out.reshape(shape), (count+1) / nOut * final_time, count)

    return u.reshape(initial_grid.shape)
