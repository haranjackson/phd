import GPRpy

from time import time

from joblib import Parallel
from numpy import array, int32, zeros

from etc import boundaries
from tests_1d import fluids, solids, multi, toro
from tests_2d import validation
from gpr.misc.plot import *

from etc.iterator import timestep, make_u
from etc.save import Data, print_stats, save_all
from multi.gfm import add_ghost_cells
from solvers.solvers import ader_stepper, split_stepper

from options import NV, NDIM, N, NCORE, CFL, RGFM
from options import SPLIT, CPP_LVL, STRANG, HALF_STEP, STIFF, FLUX, PERR_FROB


### CHECK ARGUMENTS ###
IC = solids.elastic1_IC
BC = boundaries.standard_BC


u, MPs, tf, dX = IC()
data = [Data(u, 0)]
m = len(MPs)

pool = Parallel(n_jobs=NCORE)


if CPP_LVL > 0:

    nX = array(u.shape[:NDIM] + (1,)*(3-NDIM), dtype=int32)
    extDims = GPRpy.solvers.extended_dimensions(nX, 1)
    wh = zeros(extDims * int(pow(N, NDIM)) * NV)
    qh = zeros(extDims * int(pow(N, NDIM + 1)) * NV)

    cppIterator = GPRpy.solvers.iterator
    cppSplitStepper = GPRpy.solvers.split_stepper
    cppAderStepper = GPRpy.solvers.ader_stepper


def stepper(mat, matBC, dt, *args):

    if CPP_LVL == 1:

        matr = mat.ravel()
        matBCr = matBC.ravel()

        if SPLIT:
            cppSplitStepper(matr, matBCr, wh, NDIM, nX, dt, dX,
                            STRANG, HALF_STEP, FLUX, PERR_FROB, *args)
        else:
            cppAderStepper(matr, matBCr, wh, qh, NDIM, nX, dt, dX,
                           STIFF, FLUX, PERR_FROB, *args)

        mat = matr.reshape([nX[0], nX[1], nX[2], NV])

    else:
        if SPLIT:
            split_stepper(pool, mat, matBC, dt, dX, *args)
        else:
            ader_stepper(pool, mat, matBC, dt, dX, *args)


def main(t, tf, count, data):

    tStart = time()
    u = data[count].grid

    if CPP_LVL == 2:
        u1 = u.ravel()
        cppIterator(u1, tf, nX, dX, CFL, False, SPLIT, STRANG, HALF_STEP, STIFF,
                    FLUX, PERR_FROB, MPs[0])
        data.append(Data(u1.reshape(u.shape), t))

    else:
        while t < tf:

            t0 = time()
            dt = timestep(u, count, t, tf, dX, MPs)

            mats = array([u for i in range(m)])
            if RGFM:
                add_ghost_cells(mats, MPs, dt)

            print_stats(count, t, dt)

            for i in range(m):
                mat = mats[i]
                matBC = BC(mat)
                MP = MPs[i]
                stepper(mat, matBC, dt, MP)

            u = make_u(mats)
            data.append(Data(u, t))

            if RGFM:
                # reinitialize level sets
                pass

            t += dt
            count += 1
            print('Total Time:', time() - t0, '\n')

    print('TOTAL RUNTIME:', time() - tStart)


if __name__ == "__main__":
    main(0, tf, 0, data)
    # save_all(data)
