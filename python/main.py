import GPRpy

from time import time

from joblib import Parallel
from numpy import array, zeros

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
IC = solids.piston_IC
BC = solids.piston_BC


u, MPs, tf, dX = IC()
data = [Data(u, 0)]
m = len(MPs)

pool = Parallel(n_jobs=NCORE)


if CPP_LVL > 0:

    nx, ny, nz = u.shape[:3]
    extDims = GPRpy.solvers.extended_dimensions(nx, ny, nz)
    wh = zeros(extDims * int(pow(N, NDIM)) * NV)
    qh = zeros(extDims * int(pow(N, NDIM + 1)) * NV)

    cppIterator = GPRpy.solvers.iterator
    cppSplitStepper = GPRpy.solvers.split_stepper
    cppAderStepper = GPRpy.solvers.ader_stepper


def main(t, tf, count, data):

    tStart = time()
    u = data[count].grid

    if CPP_LVL == 2:
        u1 = u.ravel()
        cppIterator(u1, tf, nx, ny, nz, dX[0], dX[1], dX[2], CFL, False,
                    SPLIT, STRANG, HALF_STEP, STIFF, FLUX, PERR_FROB, MPs[0])
        data.append(Data(u1.reshape(u.shape), t))

    else:
        while t < tf:

            t0 = time()

            mats = array([u for i in range(m)])
            dt = timestep(mats, count, t, tf, dX, MPs)

            if RGFM:
                add_ghost_cells(mats, MPs, dt)

            print_stats(count, t, dt)

            for i in range(m):

                mat = mats[i]
                matBC = BC(mat)
                MP = MPs[i]

                if CPP_LVL == 1:
                    matr = mat.ravel()
                    matBCr = matBC.ravel()

                    if SPLIT:
                        cppSplitStepper(matr, matBCr, wh, NDIM, nx, ny, nz,
                                        dt, dX[0], dX[1], dX[2],
                                        STRANG, HALF_STEP, FLUX, PERR_FROB, MP)
                    else:
                        cppAderStepper(matr, matBCr, wh, qh, NDIM, nx, ny, nz,
                                       dt, dX[0], dX[1], dX[2],
                                       STIFF, FLUX, PERR_FROB, MP)

                    mat = matr.reshape([nx, ny, nz, NV])

                else:
                    if SPLIT:
                        split_stepper(pool, mat, matBC, dt, dX, MP)
                    else:
                        ader_stepper(pool, mat, matBC, dt, dX, MP)

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
    save_all(data)
