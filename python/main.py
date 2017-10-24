from time import time

from joblib import Parallel
from numpy import array, zeros

import auxiliary.boundaries

import tests_1d.diffusion
import tests_1d.multi
import tests_1d.validation
import tests_1d.toro
import tests_2d.validation
from gpr.plot import *

from auxiliary.classes import Data
from auxiliary.iterator import timestep
from solvers.solvers import ader_stepper, split_stepper
from auxiliary.save import print_stats, save_all, make_u
from multi.gfm import add_ghost_cells, interface_inds
from options import nx, ny, nz, dx, dy, dz, ndim, N1, tf
from options import NCORE, RGFM, SPLIT, USE_CPP, STRANG, HALF_STEP, PERRON_FROB


### CHECK ARGUMENTS ###
IC = tests_1d.validation.heat_conduction_IC
BC = auxiliary.boundaries.standard_BC


u, PARs, interfaceLocs = IC()
data = [Data(u, interfaceLocs, 0)]


if USE_CPP:
    import GPRpy
    from auxiliary.classes import CParameters
    extDims = GPRpy.solvers.extended_dimensions(nx, ny, nz)
    ub = zeros(extDims * 17);
    wh = zeros(extDims * int(pow(N1,ndim)) * 17);
    qh = zeros(extDims * int(pow(N1,ndim+1)) * 17);
    cPARs = [CParameters(PAR) for PAR in PARs]


def run(t, tf, count, data):

    tStart = time()

    u = data[count].grid
    interfaceLocs = data[count].int

    m = len(interfaceLocs)
    interfaceInds = interface_inds(interfaceLocs, nx)
    interfaceVels = zeros(m)

    pool = Parallel(n_jobs=NCORE)

    while t < tf:

        t0 = time()

        fluids = array([u for i in range(m+1)])
        dt = timestep(fluids, count, t, tf, PARs)

        if RGFM:
            add_ghost_cells(fluids, interfaceInds, interfaceVels, PARs, dt)

        print_stats(count, t, dt, interfaceLocs)

        for i in range(m+1):
            fluid = fluids[i]
            PAR = PARs[i]

            if USE_CPP:
                tmp = fluid[:,:,:,:17].ravel()
                MP = cPARs[i]
                if SPLIT:
                    GPRpy.solvers.split_stepper(tmp, ub, wh, ndim, nx, ny, nz,
                                                dt, dx, dy, dz, False,
                                                bool(STRANG), bool(HALF_STEP),
                                                bool(PERRON_FROB), MP)
                else:
                    GPRpy.solvers.ader_stepper(tmp, ub, wh, qh, ndim, nx, ny, nz,
                                               dt, dx, dy, dz, False,
                                               bool(PERRON_FROB), MP)

                fluid[:,:,:,:17] = tmp.reshape([nx,ny,nz,17])

            else:
                if SPLIT:
                    split_stepper(pool, fluid, BC, dt, PAR)
                else:
                    ader_stepper(pool, fluid, BC, dt, PAR)

        if RGFM:
            interfaceLocs += interfaceVels * dt
            interfaceInds = interface_inds(interfaceLocs, nx)

        u = make_u(fluids, interfaceInds)

        t += dt
        count += 1
        data.append(Data(u, interfaceLocs, t))
        print('Total Time:', time()-t0, '\n')

    print('TOTAL RUNTIME:', time()-tStart)

if __name__ == "__main__":
    run(0, tf, 0, data)
    save_all(data)
