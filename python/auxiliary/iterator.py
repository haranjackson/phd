from time import time

from numpy import repeat, newaxis

from ader.fv import fv_terms
from ader.fv_space_only import fv_terms_space_only
from ader.dg import predictor
from ader.parallel import para_predictor, para_fv_terms, para_fv_terms_space_only
from ader.weno import weno, weno_primitive

from auxiliary.bc import standard_BC
from auxiliary.adjust import limit_noise

from experimental.new_solver import new_predictor

from gpr.eig import max_abs_eigs
from gpr.variables.vectors import primitive, primitive_vector, Cvec_to_Pvec
from gpr.thermo import thermal_stepper

from slic.ode import ode_stepper
from slic.homogeneous import flux_stepper

from options import NT, N1, CFL, dx, tf
from options import useDG, reconstructPrim, altThermSolve, fullBurn, burnProp, paraDG, paraFV


def continue_condition(t, fluids):
    if fullBurn:
        propRemaining = remaining_reactant(fluids)
        print('Unburnt Cells:', int(100*propRemaining), '%')
        return propRemaining > burnProp
    else:
        return t < tf

def timestep(fluids, count, t, PARs, SYS):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    m = len(fluids)
    MAX = 0
    for i in range(m):
        u = fluids[i]
        PAR = PARs[i]
        n = len(u)
        for j in range(n):
            P = Cvec_to_Pvec(u[j,0,0], PAR, SYS)
            MAX = max(MAX, max_abs_eigs(P, 0, PAR, SYS))

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

def stepper(pool, fluid, fluidBC, dt, PAR, SYS):

    wenoTime = 0; dgTime = 0; fvTime = 0

    if altThermSolve and not SYS.mechanical:
        t0 = time()
        fluid = thermal_stepper(fluidBC, dt, PAR)
        qh = None
        t1 = time()

    else:
        t0 = time()
        if reconstructPrim:
            wh = weno_primitive(fluidBC, PAR, SYS)
        else:
            wh = weno(fluidBC)
        t1 = time()

        if useDG:
            if paraDG:
                qh = para_predictor(pool, wh, dt, PAR, SYS)
            else:
                qh = predictor(wh, dt, PAR, SYS)
            t2 = time()

            if paraFV:
                fluid += limit_noise(para_fv_terms(pool, qh, dt, PAR, SYS))
            else:
                fluid += limit_noise(fv_terms(qh, dt, PAR, SYS))
            t3 = time()

        else:
            nx = wh.shape[0]; ny = wh.shape[1]; nz = wh.shape[2]
            qh = repeat(wh[:,:,:,newaxis], N1, 3).reshape([nx, ny, nz, NT, 18])
            t2 = time()

            if paraFV:
                fluid += limit_noise(para_fv_terms_space_only(pool, wh, dt, PAR, SYS))
            else:
                fluid += limit_noise(fv_terms_space_only(wh, dt, PAR, SYS))
            t3 = time()

        wenoTime += t1-t0; dgTime += t2-t1; fvTime += t3-t2;

    if altThermSolve and not SYS.mechanical:
        print('OS:', t1-t0)
    else:
        print('WENO:', wenoTime, '\nDG:  ', dgTime, '\nFV:  ', fvTime)

    return qh

def slic_stepper(fluid, dt, PAR, SYS):
    ode_stepper(fluid, dt/2, PAR, SYS)
    fluidn = standard_BC(standard_BC(fluid))
    flux_stepper(fluid, fluidn, dt, PAR, SYS)
    ode_stepper(fluid, dt/2, PAR, SYS)
    return None

def new_stepper(fluid, fluidBC, dt, PAR, SYS):

    wenoTime = 0; dgTime = 0; fvTime = 0
    t0 = time()

    nx = len(fluidBC)
    for i in range(nx):
        P = primitive(fluidBC[i,0,0], PAR, SYS)
        fluidBC[i,0,0] = primitive_vector(P)
    wh = weno(fluidBC)
    t1 = time()

    qh = new_predictor(wh, dt, PAR, SYS)
    t2 = time()

    fluid += limit_noise(fv_terms(qh, dt, PAR, SYS))
    t3 = time()

    wenoTime += t1-t0; dgTime += t2-t1; fvTime += t3-t2;

    print('WENO:', wenoTime, '\nDG:  ', dgTime, '\nFV:  ', fvTime)

    return qh
