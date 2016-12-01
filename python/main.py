from time import time

from joblib import Parallel
from numpy import array, zeros

from auxiliary.bc import standard_BC, periodic_BC
from tests.oned.cookoff import CKR_BC, fixed_wall_temp_BC
from tests.oned.cookoff import chapman_jouguet_IC, CKR_IC
from tests.oned.diffusion import barrier_IC, barrier_BC
from tests.oned.multi import sod_shock_IC, water_gas_IC, water_water_IC, helium_bubble_IC
from tests.oned.multi import helium_heat_transmission_IC
from tests.oned.validation import first_stokes_problem_IC, heat_conduction_IC
from tests.oned.validation import viscous_shock_IC, semenov_IC
from tests.oned.toro import toro_test1_IC
from tests.twod.validation import convected_isentropic_vortex_IC, circular_explosion_IC
from tests.twod.validation import laminar_boundary_layer_IC, hagen_poiseuille_duct_IC
from tests.twod.validation import lid_driven_cavity_IC, double_shear_layer_IC
from tests.twod.validation import taylor_green_vortex_IC
from gpr.plot import *

import options
from auxiliary.adjust import thermal_conversion
from auxiliary.classes import save_arrays
from auxiliary.iterator import timestep, check_ignition_started, continue_condition
from auxiliary.solvers import cookoff_stepper, aderweno_stepper
from auxiliary.solvers import split_weno_stepper, split_dg_stepper
from auxiliary.save import print_stats, record_data, save_all
from multi.gfm import add_ghost_cells, interface_indices, update_interface_locations
from options import ncore, convertTemp, nx, NT, GFM, solver, altThermSolve


IC = taylor_green_vortex_IC
BC = periodic_BC               # CHECK ARGUMENTS


SYS, SFix, TFix = options.SYS, options.SFix, options.TFix
u, PARs, intLocs = IC()
saveArrays = save_arrays(u, intLocs)

def run(t, count):

    tStart = time()

    global saveArrays, SYS, SFix, TFix

    interfaceLocations = saveArrays.interfaces[count]
    m = len(interfaceLocations)
    inds = interface_indices(interfaceLocations, nx)
    fluids = array([saveArrays.data[count] for i in range(m+1)])
    dg = zeros([nx, NT, 18])

    pool = Parallel(n_jobs=ncore)
    while continue_condition(t, fluids):

        t0 = time()

        dt = timestep(fluids, count, t, PARs, SYS)
        add_ghost_cells(fluids, inds, PARs, dt, SYS, SFix, TFix)
        fluidsBC = array([BC(fluid) for fluid in fluids])

        print_stats(count, t, dt, interfaceLocations, SYS)

        for i in range(m+1):
            fluid = fluids[i]
            fluidBC = fluidsBC[i]
            PAR = PARs[i]

            if altThermSolve and not SYS.mechanical:
                cookoff_stepper(fluid, fluidBC, dt, PAR)
            elif solver == 'ADER-WENO':
                qh = aderweno_stepper(pool, fluid, fluidBC, dt, PAR, SYS)
            elif solver == 'SPLIT-WENO':
                split_weno_stepper(pool, fluid, dt, PAR, SYS)
            elif solver == 'SPLIT-DG':
                split_dg_stepper(pool, fluid, dt, PAR, SYS)
            else:
                print('SOLVER NOT RECOGNISED')
                return

            if GFM:
                dg[inds[i]:inds[i+1]] = qh[inds[i]+1:inds[i+1]+1, 0, 0]

        if GFM:
            interfaceLocations = update_interface_locations(dg, interfaceLocations, dt)
            inds = interface_indices(interfaceLocations, nx)

        if convertTemp and not SYS.mechanical:
            thermal_conversion(fluids, PARs)

        if not SYS.mechanical:
            SYS.mechanical, SYS.viscous = check_ignition_started(fluids)

#        if count > 5:
#            TFix = 0

        t += dt
        count += 1
        record_data(fluids, inds, t, interfaceLocations, saveArrays)
        print('Total Time:', time()-t0, '\n')

    print('TOTAL RUNTIME:', time()-tStart)

if __name__ == "__main__":
    run(0, 0)
    save_all(saveArrays)
