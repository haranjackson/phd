from time import time

from ader.fv import fv_launcher
from ader.dg import dg_launcher
from ader.weno import weno, weno_primitive
from auxiliary.bc import standard_BC
from gpr.thermo import thermal_stepper
from split.homogeneous import weno_midstepper
from split.ode import ode_stepper_fast, ode_stepper_full
from options import reconstructPrim, fullODE, wenoHalfStep


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

    t1 = time()
    if fullODE:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_fast(fluid, dt/2, PAR, SYS)
    t2 = time()

    fluidBC = standard_BC(fluid)
    wh = weno(fluidBC)
    if wenoHalfStep:
        weno_midstepper(wh, dt, PAR, SYS)
    t3 = time()

    fluid += fv_launcher(pool, wh, dt, PAR, SYS, 1)
    t4 = time()

    if fullODE:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_fast(fluid, dt/2, PAR, SYS)
    t5 = time()

    print('ODE1:',t2-t1, '\nWENO:',t3-t2, '\nFV:  ',t4-t3, '\nODE2:',t5-t4)

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
