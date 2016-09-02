from time import time

from ader.fv import finite_volume_terms
from ader.dg import predictor
from ader.parallel import parallel_predictor, parallel_finite_volume_terms
from ader.weno import reconstruct

from auxiliary.adjust import limit_noise

from gpr.eig import max_abs_eigs
from gpr.thermo import thermal_stepper

from options import minParaDGLen, minParaFVLen, CFL, dx, tf, N1, reducedDomain, nx, altThermSolve
from options import fullBurn, burnProp


def continue_condition(t, fluids):
    if fullBurn:
        propRemaining = remaining_reactant(fluids)
        print('Unburnt Cells:', int(100*propRemaining), '%')
        return propRemaining > burnProp
    else:
        return t < tf

def timestep(fluids, materialParameters, count, t, mechanical, viscous, thermal, reactive):
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
            MAX = max(MAX, max_abs_eigs(Q, 0, params, mechanical, viscous, thermal, reactive))

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
        r = fluids[i,:,0,0,0]
        rc = fluids[i,:,0,0,17]
        if (rc/r < 0.975).any():
            print('/// IGNITION STARTED ///')
            return 1
    return 0

def remaining_reactant(fluids):
    return sum(fluids[0,:,0,0,17]/fluids[0,:,0,0,0] > 6e-6) / len(fluids[0,:,0,0])

def stepper(fluid, fluidBC, params, dt, pool, mechanical, viscous, thermal, reactive):

    wenoTime = 0; dgTime = 0; fvTime = 0
    changeRanges = changing_cells(fluidBC)
    print(changeRanges)

    for changeRange in changeRanges:
        l = changeRange[0]; r = changeRange[1]

        if altThermSolve and not mechanical:
            t0 = time()
            fluid[l:r-2] = thermal_stepper(fluidBC[l:r], params, dt)
            qh = None
            t1 = time()

        else:
            t0 = time()
            wh = reconstruct(fluidBC[l:r])
            t1 = time()

            if r-l >= minParaDGLen:
                qh = parallel_predictor(pool, wh, params, dt,
                                        mechanical, viscous, thermal, reactive)
            else:
                qh = predictor(wh, params, dt, mechanical, viscous, thermal, reactive)
            t2 = time()

            if r-l >= minParaFVLen:
                fluid[l:r-2] += limit_noise(parallel_finite_volume_terms(pool, qh, params, dt,
                                                                         mechanical, viscous,
                                                                         thermal, reactive))
            else:
                fluid[l:r-2] += limit_noise(finite_volume_terms(qh, params, dt, mechanical, viscous,
                                                                thermal, reactive))
            t3 = time()
            wenoTime += t1-t0; dgTime += t2-t1; fvTime += t3-t2;

    if altThermSolve and not mechanical:
        print('OS:', t1-t0)
    else:
        print('WENO:', wenoTime, '\nDG:  ', dgTime, '\nFV:  ', fvTime)

    return qh
