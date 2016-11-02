from gpr.eig import max_abs_eigs
from gpr.variables.vectors import Cvec_to_Pvec

from options import CFL, dx, tf, fullBurn, burnProp


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
