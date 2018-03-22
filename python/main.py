from time import time

from joblib import Parallel
from numpy import array

from etc import boundaries
from etc.cpp import cpp_ader_stepper, cpp_split_stepper, run_cpp
from etc.iterator import timestep, make_u
from etc.save import Data, print_stats, save_all
from multi.gfm import add_ghost_cells
from solvers.solvers import ader_stepper, split_stepper

from options import NV, NDIM, NCORE, RGFM, SPLIT, CPP_LVL

from models.gpr.tests.one import fluids, solids
from models.gpr.tests.two import validation
from models.gpr.misc.plot import *


### CHECK ARGUMENTS ###
IC = solids.elastic1_IC
BC = boundaries.standard_BC


def stepper(pool, mat, matBC, dt, dX, *args):

    if CPP_LVL == 1:

        if SPLIT:
            cpp_split_stepper(mat, matBC, dt, dX, *args)
        else:
            cpp_ader_stepper(mat, matBC, dt, dX, *args)

    else:
        if SPLIT:
            split_stepper(pool, mat, matBC, dt, dX, *args)
        else:
            ader_stepper(pool, mat, matBC, dt, dX, *args)


def run_py(pool, u, dX, t, tf, count, data, *args):

    tStart = time()
    u = data[count].grid

    while t < tf:

        t0 = time()
        dt = timestep(u, count, t, tf, dX, *args)

        mats = array([u for i in range(m)])
        if RGFM:
            add_ghost_cells(mats, dt, *args)

        print_stats(count, t, dt)

        for i in range(m):
            mat = mats[i]
            matBC = BC(mat)
            MP = args[0][i]
            stepper(pool, mat, matBC, dt, dX, MP)

        u = make_u(mats)
        data.append(Data(u, t))

        if RGFM:
            # reinitialize level sets
            pass

        t += dt
        count += 1
        print('Iteration Time:', time() - t0, '\n')

    print('TOTAL RUNTIME:', time() - tStart)


if __name__ == "__main__":

    u, MPs, tf, dX = IC()
    data = [Data(u, 0)]
    m = len(MPs)

    assert(u.ndim == NDIM + 1)
    assert(u.shape[-1] == NV)
    pool = Parallel(n_jobs=NCORE)

    if CPP_LVL == 2:
        run_cpp(u, dX, 0, tf, 0, data, MPs)
    else:
        run_py(pool, u, dX, 0, tf, 0, data, MPs)

    # save_all(data)
