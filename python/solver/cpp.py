import GPRpy

from numpy import array, int32, zeros


def get_dimensions(arr, NDIM):
    return array(arr.shape[:NDIM] + (1,)*(3-NDIM), dtype=int32)


def cpp_split_stepper(obj, mat, matBC, dt, dX):

    nX = get_dimensions(mat, obj.NDIM)
    extDims = GPRpy.solvers.extended_dimensions(nX, 1)
    wh = zeros(extDims * int(pow(obj.N, obj.NDIM)) * obj.NV)

    matr = mat.ravel()
    matBCr = matBC.ravel()

    GPRpy.solvers.split_stepper(matr, matBCr, wh, obj.NDIM, nX, dt, dX,
                                obj.strang, obj.half_step, obj.flux_type,
                                obj.model_params)

    mat = matr.reshape(mat.shape)


def cpp_ader_stepper(obj, mat, matBC, dt, dX):

    nX = get_dimensions(mat, obj.NDIM)
    extDims = GPRpy.solvers.extended_dimensions(nX, 1)
    wh = zeros(extDims * int(pow(obj.N, obj.NDIM)) * obj.NV)
    qh = zeros(extDims * int(pow(obj.N, obj.NDIM + 1)) * obj.NV)

    matr = mat.ravel()
    matBCr = matBC.ravel()

    GPRpy.solvers.ader_stepper(matr, matBCr, wh, qh, obj.NDIM, nX, dt, dX,
                               obj.stiff, obj.flux_type, obj.model_params)

    mat = matr.reshape(mat.shape)


def solve_full_cpp(obj, initial_grid, final_time, dX, cfl):
    nX = get_dimensions(initial_grid)
    u = initial_grid.ravel()
    GPRpy.solvers.iterator(u, final_time, nX, dX, cfl, False, obj.split,
                           obj.strang, obj.half_step, obj.stiff_dg,
                           obj.flux_type, obj.model_params)
    return u.reshape(initial_grid.shape)
