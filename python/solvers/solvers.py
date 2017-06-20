from time import time

from solvers.fv.fv import fv_launcher
from solvers.dg.dg import dg_launcher
from solvers.weno.weno import weno_launcher
from auxiliary.boundaries import standard_BC
from gpr.thermo import thermal_stepper
from solvers.split.homogeneous import weno_midstepper
from solvers.split.ode import ode_launcher
from options import wenoHalfStep, StrangSplit


def cookoff_stepper(fluid, fluidBC, dt, PAR):
    t0 = time()
    fluid[:] = thermal_stepper(fluidBC, dt, PAR)

    print('OS:', time()-t0)

def aderweno_stepper(pool, fluid, fluidBC, dt, PAR, SYS):
    t0 = time()

    wh = weno_launcher(fluidBC)
    t1 = time()

    qh = dg_launcher(pool, wh, dt, PAR, SYS)
    t2 = time()

    fluid += fv_launcher(pool, qh, dt, PAR, SYS)
    t3 = time()

    print('WENO:', t1-t0)
    print('DG:  ', t2-t1)
    print('FV:  ', t3-t2)

    return qh

def split_weno_stepper(pool, fluid, dt, PAR, SYS):

    Δt = dt/2 if StrangSplit else dt
    t0 = time()

    ode_launcher(fluid, Δt, PAR, SYS)
    t1 = time()

    fluidBC = standard_BC(fluid)
    wh = weno_launcher(fluidBC)
    if wenoHalfStep:
        weno_midstepper(wh, dt, PAR, SYS)
    t2 = time()

    fluid += fv_launcher(pool, wh, dt, PAR, SYS, 1)
    t3 = time()

    print('ODE: ', t1-t0)
    print('WENO:', t2-t1)
    print('FV:  ', t3-t2)

    if StrangSplit:
        ode_launcher(fluid, Δt, PAR, SYS)
        t4 = time()
        print('ODE: ', t4-t3)

def split_dg_stepper(pool, fluid, dt, PAR, SYS):
    t0 = time()

    ode_launcher(fluid, dt/2, PAR, SYS)
    t1 = time()

    fluidBC = standard_BC(fluid)
    wh = weno_launcher(fluidBC)
    t2 = time()

    qh = dg_launcher(pool, wh, dt, PAR, SYS, 1)
    t3 = time()

    fluid += fv_launcher(pool, qh, dt, PAR, SYS, 1)
    t4 = time()

    ode_launcher(fluid, dt/2, PAR, SYS)
    t5 = time()

    print('ODE1:', t1-t0)
    print('WENO:', t2-t1)
    print('DG:  ', t3-t2)
    print('FV:  ', t4-t3)
    print('ODE2:', t5-t4)
