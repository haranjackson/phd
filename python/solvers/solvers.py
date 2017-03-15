from time import time

from solvers.fv.fv import fv_launcher
from solvers.dg.dg import dg_launcher
from solvers.weno.weno import weno, weno_primitive
from auxiliary.boundaries import standard_BC
from gpr.thermo import thermal_stepper
from solvers.split.homogeneous import weno_midstepper
from solvers.split.ode import ode_stepper_fast, ode_stepper_full
from options import reconstructPrim, fullODE, wenoHalfStep, StrangSplit


def cookoff_stepper(fluid, fluidBC, dt, PAR):
    t0 = time()
    fluid[:] = thermal_stepper(fluidBC, dt, PAR)
    print('OS:', time()-t0)

def aderweno_stepper(pool, fluid, fluidBC, dt, PAR, SYS):
    t0 = time()

    if reconstructPrim:
        wh = weno_primitive(fluidBC, PAR, SYS)
    else:
        wh = weno(fluidBC)
    t1 = time()
    print('WENO:', t1-t0)

    qh = dg_launcher(pool, wh, dt, PAR, SYS)
    t2 = time()
    print('DG:  ', t2-t1)

    fluid += fv_launcher(pool, qh, dt, PAR, SYS)
    print('FV:  ', time()-t2)

    return qh

def split_weno_stepper(pool, fluid, dt, PAR, SYS):

    if StrangSplit:
        Δt = dt/2
    else:
        Δt = dt

    t1 = time()
    fluidBC = standard_BC(fluid)
    wh = weno(fluidBC)
    if wenoHalfStep:
        weno_midstepper(wh, Δt, PAR, SYS)
    t2 = time()
    print('WENO:',t2-t1)

    fluid += fv_launcher(pool, wh, Δt, PAR, SYS, 1)
    t3 = time()
    print('FV:  ',t3-t2)

    if fullODE:
        ode_stepper_full(fluid, dt, PAR, SYS)
    else:
        ode_stepper_fast(fluid, dt, PAR, SYS)
    t4 = time()
    print('ODE: ',t4-t3)

    if StrangSplit:
        fluidBC = standard_BC(fluid)
        wh = weno(fluidBC)
        if wenoHalfStep:
            weno_midstepper(wh, Δt, PAR, SYS)
        t5 = time()
        print('WENO:',t5-t4)

        fluid += fv_launcher(pool, wh, Δt, PAR, SYS, 1)
        print('FV:  ',time()-t5)

def split_dg_stepper(pool, fluid, dt, PAR, SYS):
    t1 = time()

    if fullODE:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_fast(fluid, dt/2, PAR, SYS)
    t2 = time()

    fluidBC = standard_BC(fluid)
    wh = weno(fluidBC)
    t3 = time()

    qh = dg_launcher(pool, wh, dt, PAR, SYS, 1)
    t4 = time()

    fluid += fv_launcher(pool, qh, dt, PAR, SYS, 1)
    t5 = time()

    if fullODE:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_fast(fluid, dt/2, PAR, SYS)
    t6 = time()

    print('ODE1:',t2-t1, '\nWENO:',t3-t2, '\nDG:  ',t4-t3, '\nFV:  ',t5-t4, '\nODE2:',t6-t5)
