from time import time

from joblib import Parallel
from numpy import array, zeros

from auxiliary.boundaries import standard_BC, periodic_BC
from tests_1d.diffusion import barrier_IC, barrier_BC
from tests_1d.multi import sod_shock_IC, water_gas_IC, water_water_IC, helium_bubble_IC
from tests_1d.multi import helium_heat_transmission_IC
from tests_1d.validation import first_stokes_problem_IC, heat_conduction_IC
from tests_1d.validation import viscous_shock_IC, semenov_IC
from tests_1d.toro import toro_test1_IC
from tests_2d.validation import convected_isentropic_vortex_IC, circular_explosion_IC
from tests_2d.validation import laminar_boundary_layer_IC, hagen_poiseuille_duct_IC
from tests_2d.validation import lid_driven_cavity_IC, lid_driven_cavity_BC
from tests_2d.validation import double_shear_layer_IC, taylor_green_vortex_IC
from gpr.plot import *

from auxiliary.classes import save_arrays
from auxiliary.iterator import timestep
from solvers.solvers import aderweno_stepper, split_weno_stepper
from auxiliary.save import print_stats, record_data, save_all, make_u
from multi.gfm import add_ghost_cells, interface_inds
from options import ncore, nx, RGFM, SOLVER, tf



### CHECK ARGUMENTS ###
IC = heat_conduction_IC
BC = standard_BC


u, PARs, intLocs = IC()
saveArrays = save_arrays(u, intLocs)

def run(t, tf, count, saveArrays):

    tStart = time()

    u = saveArrays.data[count]
    interfaceLocs = saveArrays.interfaces[count]

    m = len(interfaceLocs)
    interfaceInds = interface_inds(interfaceLocs, nx)
    interfaceVels = zeros(m)

    pool = Parallel(n_jobs=ncore)

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

            if SOLVER == 'ADER-WENO':
                aderweno_stepper(pool, fluid, BC, dt, PAR)
            elif SOLVER == 'SPLIT-WENO':
                split_weno_stepper(pool, fluid, BC, dt, PAR)

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
