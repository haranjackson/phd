from time import time

from solvers.fv.fv import fv_launcher
from solvers.dg.dg import dg_launcher
from solvers.weno.weno import weno_launcher
from solvers.split.split import ode_launcher, weno_midstepper
from options import HALF_STEP, STRANG


def ader_stepper(pool, mat, matBC, dt, dX, MP):
    t0 = time()

    wh = weno_launcher(matBC)
    t1 = time()

    qh = dg_launcher(pool, wh, dt, dX, MP)
    t2 = time()

    mat += fv_launcher(pool, qh, dt, dX, MP)
    t3 = time()

    print('WENO:', t1 - t0)
    print('DG:  ', t2 - t1)
    print('FV:  ', t3 - t2)


def split_stepper(pool, mat, matBC, dt, dX, MP):

    Δt = dt / 2 if STRANG else dt
    t0 = time()

    ode_launcher(matBC, Δt, MP)
    t1 = time()

    wh = weno_launcher(matBC)
    if HALF_STEP:
        weno_midstepper(wh, dt, dX, MP)
    t2 = time()

    mat += fv_launcher(pool, wh, dt, dX, MP, 1)
    t3 = time()

    print('ODE: ', t1 - t0)
    print('WENO:', t2 - t1)
    print('FV:  ', t3 - t2)

    if STRANG:
        ode_launcher(mat, Δt, MP)
        t4 = time()
        print('ODE: ', t4 - t3)
