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
from auxiliary.save import print_stats, record_data, save_all
from multi.gfm import add_ghost_cells, interface_inds, update_interface_locs
from options import ncore, nx, NT, GFM, SOLVER, tf



### CHECK ARGUMENTS ###
IC = first_stokes_problem_IC
BC = standard_BC


u, PARs, intLocs = IC()
saveArrays = save_arrays(u, intLocs)

def run(t, tf, count, saveArrays):

    tStart = time()

    interfaceLocs = saveArrays.interfaces[count]
    m = len(interfaceLocs)
    interfaceInds = interface_inds(interfaceLocs, nx)
    fluids = array([saveArrays.data[count] for i in range(m+1)])
    dg = zeros([nx, NT, 18])

    pool = Parallel(n_jobs=ncore)
    while t < tf:

        t0 = time()

        dt = timestep(fluids, count, t, tf, PARs)

        if GFM:
            add_ghost_cells(fluids, interfaceInds, PARs, dt)

        print_stats(count, t, dt, interfaceLocs)

        for i in range(m+1):
            fluid = fluids[i]
            PAR = PARs[i]

            if SOLVER == 'ADER-WENO':
                qh = aderweno_stepper(pool, fluid, BC, dt, PAR)
            elif SOLVER == 'SPLIT-WENO':
                split_weno_stepper(pool, fluid, BC, dt, PAR)

            if GFM:
                dg[interfaceInds[i]:interfaceInds[i+1]] \
                  = qh[interfaceInds[i]+1:interfaceInds[i+1]+1, 0, 0]

        if GFM:
            interfaceLocs = update_interface_locs(dg, interfaceLocs, dt)
            interfaceInds = interface_inds(interfaceLocs, nx)

        t += dt
        count += 1
        record_data(fluids, interfaceInds, t, interfaceLocs, saveArrays)
        print('Total Time:', time()-t0, '\n')

    print('TOTAL RUNTIME:', time()-tStart)

if __name__ == "__main__":
    run(0, tf, 0, saveArrays)
    save_all(saveArrays)
