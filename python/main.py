from time import time

from joblib import Parallel
from numpy import array, expand_dims, zeros

from auxiliary.bc import standard_BC, periodic_BC
from tests.cookoff import CKR_BC, fixed_wall_temp_BC
from tests.cookoff import chapman_jouguet_IC, CKR_IC, detonation_IC
from tests.validation import first_stokes_problem_IC, heat_conduction_IC
from tests.validation import viscous_shock_IC, semenov_IC
from tests.multi import sod_shock_IC, water_gas_IC, water_water_IC, multimaterial2_IC
from tests.multi import helium_heat_transmission_IC
from gpr.plot import *

from auxiliary.adjust import renormalise_density, thermal_conversion
from auxiliary.iterator import timestep, stepper, check_ignition_started, continue_condition
from auxiliary.save import record_data, save_all
from multi.gfm import add_ghost_cells, interface_indices, update_interface_locations
from options import ncore, renormaliseRho, convertTemp, nx, NT, GFM
from options import mechanical, viscous, thermal, reactive


initial_condition   = CKR_IC
boundary_conditions = fixed_wall_temp_BC        # CHECK ARGUMENTS


mechanical, viscous = mechanical, viscous
u, materialParameters, intLocs = initial_condition()
dataArray = expand_dims(u.copy(), axis=0)
timeArray = array([0])
interArray = expand_dims(array(intLocs), axis=0)


def run(t, count):

    global dataArray, timeArray, interArray, mechanical, viscous

    interfaceLocations = interArray[count]
    m = len(interfaceLocations)
    inds = interface_indices(interfaceLocations, nx)
    fluids = array([dataArray[count] for i in range(m+1)])
    dg = zeros([nx, NT, 18])

    pool = Parallel(n_jobs=ncore)
    while continue_condition(t, fluids):

        t0 = time()

        dt = timestep(fluids, materialParameters, count, t, mechanical, viscous, thermal, reactive)
        add_ghost_cells(fluids, inds, materialParameters, dt, viscous, thermal, reactive)
        fluidsBC = array([boundary_conditions(fluids[i], t+1e-14, materialParameters[i],
                                              viscous, thermal, reactive)
                          for i in range(m+1)])

        print(count+1)
        print('t  =', t)
        print('dt =', dt)
        print('Interfaces =', interfaceLocations)

        for i in range(m+1):
            fluid = fluids[i]
            fluidBC = fluidsBC[i]
            params = materialParameters[i]
            qh = stepper(fluid, fluidBC, params, dt, pool, mechanical, viscous, thermal, reactive)
            if GFM:
                l = inds[i]
                r = inds[i+1]
                dg[l:r] = qh[l+1:r+1, 0, 0]

        interfaceLocations = update_interface_locations(dg, interfaceLocations, dt)
        inds = interface_indices(interfaceLocations, nx)

        if renormaliseRho:
            for i in range(m+1):
                renormalise_density(fluids[i])

        if convertTemp and not mechanical:
            for i in range(m+1):
                thermal_conversion(fluids[i], materialParameters[i])

        for i in range(m+1):
            l = inds[i]
            r = inds[i+1]
            u[l:r] = fluids[i][l:r]

        t += dt
        count += 1
        dataArray, timeArray, interArray = record_data(u, t, interfaceLocations, dataArray,
                                                       timeArray, interArray)

        print('M,V,T,R =', mechanical, viscous, thermal, reactive)
        if not mechanical:
            started = check_ignition_started(fluids)
            mechanical = started
            viscous = started

        print('Total Time:', time()-t0, '\n')

if __name__ == "__main__":
    run(0, 0)
    save_all(dataArray, timeArray, interArray)
