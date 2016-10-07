from gpr.functions import primitive
from gpr.variables import total_energy
from options import ndim
from ader.weno import extend


def standard_BC(u, reflect=0):
    ret = extend(u, 1, 0)
    if reflect:
        ret[0, :, :, 2:5] *= -1
        ret[0, :, :, 14:17] *= -1
        ret[-1, :, :, 2:5] *= -1
        ret[-1, :, :, 14:17] *= -1
    if ndim > 1:
        ret = extend(u, 1, 1)
        if reflect:
            ret[:, 0, :, 2:5] *= -1
            ret[:, 0, :, 14:17] *= -1
            ret[:, -1, :, 2:5] *= -1
            ret[:, -1, :, 14:17] *= -1
    if ndim > 2:
        ret = extend(u, 1, 2)
        if reflect:
            ret[:, :, 0, 2:5] *= -1
            ret[:, :, 0, 14:17] *= -1
            ret[:, :, -1, 2:5] *= -1
            ret[:, :, -1, 14:17] *= -1
    return ret

def periodic_BC(u):
    ret = extend(u, 1, 0)
    ret[0] = ret[-2]
    ret[-1] = ret[1]
    if ndim > 1:
        ret = extend(ret, 1, 1)
        ret[:,0] = ret[:,-2]
        ret[:,-1] = ret[:,1]
    if ndim > 2:
        ret = extend(ret, 1, 2)
        ret[:,:,0] = ret[:,:,-2]
        ret[:,:,-1] = ret[:,:,1]
    return ret

def energy_fix(Q1, Q2):
    """ Calculates the energy in Q1, given that it must have the same pressure as Q2
    """
    P1 = primitive(Q1)
    P2 = primitive(Q2)
    return P1.ρ * total_energy(P1.ρ, P2.p, P1.v, P1.A, P1.J)

def temperature_fix_density(p, T, params):
    """ Calculates the density in Q, given that the cell must be at temperature T
    """
    return (p + params.pINF) / ((params.γ - 1) * T * params.cv)

def temperature_fix_pressure(ρ, T, params):
    """ Calculates the pressure in Q, given that the cell must be at temperature T
    """
    return ρ *(params.γ-1) * T * params.cv - params.pINF
