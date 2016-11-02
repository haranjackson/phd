from time import time

from joblib import Parallel
from numpy import array, zeros

from auxiliary.bc import standard_BC, periodic_BC
from tests.cookoff import CKR_BC, fixed_wall_temp_BC
from tests.cookoff import chapman_jouguet_IC, CKR_IC
from tests.diffusion import barrier_IC, barrier_BC
from tests.validation import first_stokes_problem_IC, heat_conduction_IC
from tests.validation import viscous_shock_IC, semenov_IC
from tests.multi import sod_shock_IC, water_gas_IC, water_water_IC, helium_bubble_IC
from tests.multi import helium_heat_transmission_IC
from tests.toro import toro_test1_IC
from gpr.plot import *

import options
from auxiliary.adjust import thermal_conversion
from auxiliary.classes import save_arrays
from auxiliary.iterator import timestep, check_ignition_started, continue_condition
from auxiliary.solvers import cookoff_stepper, aderweno_stepper, slic_stepper, new_stepper
from auxiliary.solvers import weno_stepper
from auxiliary.save import print_stats, record_data, save_all
from multi.gfm import add_ghost_cells, interface_indices, update_interface_locations
from options import ncore, convertTemp, nx, NT, GFM, solver, altThermSolve, useDG


IC = first_stokes_problem_IC
BC = standard_BC               # CHECK ARGUMENTS


SYS, SFix, TFix = options.SYS, options.SFix, options.TFix
u, PARs, intLocs = IC()
saveArrays = save_arrays(u, intLocs)

def run(t, count):

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
            elif solver == 'SLIC':
                slic_stepper(fluid, dt, PAR, SYS)
            elif solver == 'AW':
                if useDG:
                    qh = aderweno_stepper(pool, fluid, fluidBC, dt, PAR, SYS)
                else:
                    qh = weno_stepper(pool, fluid, fluidBC, dt, PAR, SYS)
            elif solver == 'NEW':
                qh = new_stepper(fluid, fluidBC, dt, PAR, SYS)

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

if __name__ == "__main__":
    run(0, 0)
    save_all(saveArrays)
