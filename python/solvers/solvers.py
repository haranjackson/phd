from time import time

from solvers.fv.fv import fv_launcher
from solvers.dg.dg import dg_launcher
from solvers.weno.weno import weno_launcher
from solvers.split.homogeneous import weno_midstepper
from solvers.split.ode import ode_launcher
from options import HALF_STEP, STRANG


def ader_stepper(pool, fluid, BC, dt, PAR):
    t0 = time()

    wh = weno_launcher(BC(fluid))
    t1 = time()

    qh = dg_launcher(pool, wh, dt, PAR)
    t2 = time()

    fluid += fv_launcher(pool, qh, dt, PAR)
    t3 = time()

    print('WENO:', t1-t0)
    print('DG:  ', t2-t1)
    print('FV:  ', t3-t2)

def split_stepper(pool, fluid, BC, dt, PAR):

    Δt = dt/2 if STRANG else dt
    t0 = time()

    ode_launcher(fluid, Δt, PAR)
    t1 = time()

    wh = weno_launcher(BC(fluid))
    if HALF_STEP:
        weno_midstepper(wh, dt, PAR)
    t2 = time()

    fluid += fv_launcher(pool, wh, dt, PAR, 1)
    t3 = time()

    print('ODE: ', t1-t0)
    print('WENO:', t2-t1)
    print('FV:  ', t3-t2)

    if STRANG:
        ode_launcher(fluid, Δt, PAR)
        t4 = time()
        print('ODE: ', t4-t3)
