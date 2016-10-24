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

from options import nx, NT, CFL, dx, tf, N1, reducedDomain, altThermSolve
from options import fullBurn, burnProp, useDG, minParaDGLen, minParaFVLen


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

def changing_cells(u):
    """ Returns the ranges of the cells which may change at the next time step
    """
    if reducedDomain:
        def add(arr, start, stop, d):
            if stop-start > 2*d:
                if start==0:
                    arr.append([start, stop-d])
                else:
                    arr.append([start+d, stop-d])
            return arr

        inds = []       # ranges of cells that will remain the same at the next time step
        n = len(u)
        u0 = u[0,0,0]
        start = 0
        for i in range(n):
            if not (u[i] == u0).all():
                inds = add(inds, start, i, N1)
                start = i
                u0 = u[i]
        inds = add(inds, start, n+N1, N1)

        ret = []       # ranges of cells that may change at the next time step
        start = 0
        for i in range(len(inds)):
            stop = inds[i][0]
            ret = add(ret, start, stop, 0)
            start = inds[i][1]
        ret = add(ret, start, n, 0)
        for i in range(len(ret)):           # Add end cells, which will be removed in finite volume
            ret[i][0] = max(0, ret[i][0]-1)
            ret[i][1] = min(n, ret[i][1]+1)

        return ret

    else:
        return [[0, nx+2]]

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
    changeRanges = changing_cells(fluidBC)
    print(changeRanges)

    for changeRange in changeRanges:
        l = changeRange[0]; r = changeRange[1]

        if altThermSolve and not subsystems.mechanical:
            t0 = time()
            fluid[l:r-2] = thermal_stepper(fluidBC[l:r], params, dt)
            qh = None
            t1 = time()

        else:
            t0 = time()
            wh = reconstruct(fluidBC[l:r])
            t1 = time()

            if useDG:
                if r-l >= minParaDGLen:
                    qh = para_predictor(pool, wh, params, dt, subsystems)
                else:
                    qh = predictor(wh, params, dt, subsystems)
                t2 = time()

                if r-l >= minParaFVLen:
                    fluid[l:r-2] += limit_noise(para_fv_terms(pool, qh, params, dt,
                                                                             subsystems))
                else:
                    fluid[l:r-2] += limit_noise(fv_terms(qh, params, dt, subsystems))
                t3 = time()

            else:
                nx = wh.shape[0]; ny = wh.shape[1]; nz = wh.shape[2]
                qh = repeat(wh[:,:,:,newaxis], N1, 3).reshape([nx, ny, nz, NT, 18])
                t2 = time()

                if r-l >= minParaFVLen:
                    fluid[l:r-2] += limit_noise(para_fv_terms_space_only(pool, wh, params, dt,
                                                                             subsystems))
                else:
                    fluid[l:r-2] += limit_noise(fv_terms_space_only(wh, params, dt, subsystems))
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
