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

from auxiliary.classes import save_arrays
from auxiliary.iterator import timestep
from solvers.solvers import aderweno_stepper, split_weno_stepper
from auxiliary.save import print_stats, record_data, save_all, make_u
from multi.gfm import add_ghost_cells, interface_inds
from options import NCORE, nx, RGFM, SPLIT, tf


### CHECK ARGUMENTS ###
IC = tests_1d.validation.heat_conduction_IC
BC = auxiliary.boundaries.standard_BC


u, PARs, intLocs = IC()
saveArrays = save_arrays(u, intLocs)

def run(t, tf, count, saveArrays):

    tStart = time()

    u = saveArrays.data[count]
    interfaceLocs = saveArrays.interfaces[count]

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

            if SPLIT:
                split_weno_stepper(pool, fluid, BC, dt, PAR)
            else:
                aderweno_stepper(pool, fluid, BC, dt, PAR)

        if RGFM:
            interfaceLocs += interfaceVels * dt
            interfaceInds = interface_inds(interfaceLocs, nx)

        u = make_u(fluids, interfaceInds)

        t += dt
        count += 1
        record_data(u, t, interfaceLocs, saveArrays)
        print('Total Time:', time()-t0, '\n')

    print('TOTAL RUNTIME:', time()-tStart)

if __name__ == "__main__":
    run(0, tf, 0, saveArrays)
    save_all(saveArrays)
