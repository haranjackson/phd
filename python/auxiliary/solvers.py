from time import time

from numpy import repeat, newaxis

from ader.fv import fv_terms
from ader.fv_space_only import fv_terms_space_only
from ader.dg import predictor
from ader.parallel import para_predictor, para_fv_terms, para_fv_terms_space_only
from ader.weno import weno, weno_primitive

from auxiliary.bc import standard_BC

from experimental.new_solver import new_predictor

from gpr.variables.vectors import primitive, primitive_vector
from gpr.thermo import thermal_stepper

from split.ode import ode_stepper, ode_stepper_full
from split.slic import slic_stepper

from options import NT, N1, reconstructPrim, paraDG, paraFV, linODE


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

    if paraDG:
        qh = para_predictor(pool, wh, dt, PAR, SYS)
    else:
        qh = predictor(wh, dt, PAR, SYS)
    t2 = time()

    if paraFV:
        fluid += para_fv_terms(pool, qh, dt, PAR, SYS)
    else:
        fluid += fv_terms(qh, dt, PAR, SYS)
    t3 = time()

    print('WENO:', t1-t0, '\nDG:  ', t2-t1, '\nFV:  ', t3-t2)
    return qh

def weno_stepper(pool, fluid, fluidBC, dt, PAR, SYS):

    t0 = time()

    wh = weno(fluidBC)
    t1 = time()

    nx,ny,nz = wh.shape[:3]
    qh = repeat(wh[:,:,:,newaxis], N1, 3).reshape([nx, ny, nz, NT, 18])
    t2 = time()

    if paraFV:
        fluid += para_fv_terms_space_only(pool, wh, dt, PAR, SYS)
    else:
        fluid += fv_terms_space_only(wh, dt, PAR, SYS)
    t2 = time()

    print('WENO:', t1-t0, '\nFV:  ', t2-t1)
    return qh

def split_slic_stepper(fluid, dt, PAR, SYS):
    if linODE:
        ode_stepper(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    fluidn = standard_BC(standard_BC(fluid))
    slic_stepper(fluid, fluidn, dt, PAR, SYS)
    if linODE:
        ode_stepper(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    return None

def split_weno_stepper(fluid, dt, PAR, SYS):
    ode_stepper_full(fluid, dt/2, PAR, SYS)
    fluidBC = standard_BC(fluid)
    wh = weno(fluidBC)
    nx,ny,nz = wh.shape[:3]
    qh = repeat(wh[:,:,:,newaxis], N1, 3).reshape([nx, ny, nz, NT, 18])
    fluid += fv_terms(qh, dt, PAR, SYS, 1)
    ode_stepper_full(fluid, dt/2, PAR, SYS)

def split_dg_stepper(fluid, dt, PAR, SYS):
    t1 = time()
    ode_stepper(fluid, dt/2, PAR, SYS)
    t2 = time()
    fluidBC = standard_BC(fluid)
    wh = weno(fluidBC)
    t3 = time()
    qh = predictor(wh, dt, PAR, SYS, 1)
    t4 = time()
    fluid += fv_terms(qh, dt, PAR, SYS, 1)
    t5 = time()
    ode_stepper_full(fluid, dt/2, PAR, SYS)
    t6 = time()
    print('ODE1:', t2-t1, '\nWENO:', t3-t2, '\nDG:  ', t4-t3, '\nFV:  ', t5-t4, '\nODE2:', t6-t5)

def new_stepper(fluid, fluidBC, dt, PAR, SYS):

    t0 = time()

    nx = len(fluidBC)
    for i in range(nx):
        P = primitive(fluidBC[i,0,0], PAR, SYS)
        fluidBC[i,0,0] = primitive_vector(P)
    wh = weno(fluidBC)
    t1 = time()

    qh = new_predictor(wh, dt, PAR, SYS)
    t2 = time()

    fluid += fv_terms(qh, dt, PAR, SYS)
    t3 = time()

    print('WENO:', t1-t0, '\nDG:  ', t2-t1, '\nFV:  ', t3-t2)

    return qh
