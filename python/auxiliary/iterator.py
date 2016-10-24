from time import time

from numpy import repeat, newaxis

from ader.fv import fv_terms
from ader.fv_space_only import fv_terms_space_only
from ader.dg import predictor
from ader.parallel import para_predictor, para_fv_terms, para_fv_terms_space_only
from ader.weno import reconstruct

from auxiliary.bc import standard_BC
from auxiliary.adjust import limit_noise

from experimental.new_solver import new_predictor

from gpr.eig import max_abs_eigs
from gpr.functions import primitive, primitive_vector
from gpr.thermo import thermal_stepper

from slic.ode import ode_stepper
from slic.homogeneous import flux_stepper

from options import NT, CFL, dx, tf, N1, altThermSolve
from options import fullBurn, burnProp, useDG, paraDG, paraFV


def continue_condition(t, fluids):
    if fullBurn:
        propRemaining = remaining_reactant(fluids)
        print('Unburnt Cells:', int(100*propRemaining), '%')
        return propRemaining > burnProp
    else:
        return t < tf

def timestep(fluids, materialParameters, count, t, subsystems):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    m = len(fluids)
    MAX = 0
    for i in range(m):
        u = fluids[i]
        params = materialParameters[i]
        n = len(u)
        for j in range(n):
            Q = u[j, 0, 0]
            MAX = max(MAX, max_abs_eigs(Q, 0, params, subsystems))

    dt = CFL * dx / MAX
    if count <= 5:
        dt *= 0.2
    if t + dt > tf:
        return tf - t
    else:
        return dt

def check_ignition_started(fluids):
    m = len(fluids)
    for i in range(m):
        ρ = fluids[i,:,0,0,0]
        ρλ = fluids[i,:,0,0,17]
        if (ρλ/ρ < 0.975).any():
            print('/// IGNITION STARTED ///')
            return 1
    return 0

def remaining_reactant(fluids):
    return sum(fluids[0,:,0,0,17]/fluids[0,:,0,0,0] > 6e-6) / len(fluids[0,:,0,0])

def stepper(fluid, fluidBC, params, dt, pool, subsystems):

    wenoTime = 0; dgTime = 0; fvTime = 0

    if altThermSolve and not subsystems.mechanical:
        t0 = time()
        fluid = thermal_stepper(fluidBC, params, dt)
        qh = None
        t1 = time()

    else:
        t0 = time()
        wh = reconstruct(fluidBC)
        t1 = time()

        if useDG:
            if paraDG:
                qh = para_predictor(pool, wh, params, dt, subsystems)
            else:
                qh = predictor(wh, params, dt, subsystems)
            t2 = time()

            if paraFV:
                fluid += limit_noise(para_fv_terms(pool, qh, params, dt, subsystems))
            else:
                fluid += limit_noise(fv_terms(qh, params, dt, subsystems))
            t3 = time()

        else:
            nx = wh.shape[0]; ny = wh.shape[1]; nz = wh.shape[2]
            qh = repeat(wh[:,:,:,newaxis], N1, 3).reshape([nx, ny, nz, NT, 18])
            t2 = time()

            if paraFV:
                fluid += limit_noise(para_fv_terms_space_only(pool, wh, params, dt, subsystems))
            else:
                fluid += limit_noise(fv_terms_space_only(wh, params, dt, subsystems))
            t3 = time()

        wenoTime += t1-t0; dgTime += t2-t1; fvTime += t3-t2;

    if altThermSolve and not subsystems.mechanical:
        print('OS:', t1-t0)
    else:
        print('WENO:', wenoTime, '\nDG:  ', dgTime, '\nFV:  ', fvTime)

    return qh

def slic_stepper(fluid, params, dt, subsystems):
    ode_stepper(fluid, params, subsystems, dt/2)
    fluidn = standard_BC(standard_BC(fluid))
    flux_stepper(fluid, fluidn, params, subsystems, dt)
    ode_stepper(fluid, params, subsystems, dt/2)
    return None

def new_stepper(fluid, fluidBC, params, dt, pool, subsystems):

    wenoTime = 0; dgTime = 0; fvTime = 0
    t0 = time()

    nx = len(fluidBC)
    for i in range(nx):
        P = primitive(fluidBC[i,0,0], params, subsystems)
        fluidBC[i,0,0] = primitive_vector(P)
    wh = reconstruct(fluidBC)
    t1 = time()

    qh = new_predictor(wh, params, dt, subsystems)
    t2 = time()

    fluid += limit_noise(fv_terms(qh, params, dt, subsystems))
    t3 = time()

    wenoTime += t1-t0; dgTime += t2-t1; fvTime += t3-t2;

    print('WENO:', wenoTime, '\nDG:  ', dgTime, '\nFV:  ', fvTime)

    return qh
