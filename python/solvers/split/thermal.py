from numpy import exp, eye, sqrt

from gpr.misc.functions import L2_1D
from gpr.variables.eos import E_2A, E_2J, E_3, dEdJ
from gpr.variables.sources import theta2inv


def E_to_T(ρ, E, A, J, MP):
    """ Returns the temperature of an ideal gas, given the energy
        (minus the kinetic energy or any chemical energy)
    """
    E1 = E - E_2A(ρ, A, MP) - E_2J(J, MP)
    return E1 / MP.cv

def f_J(ρ, E, A, J, MP):
    T = E_to_T(ρ, E, A, J, MP)
    return - dEdJ(J, MP) * theta2inv(ρ, T, MP)

def jac_J(ρ, E, A, J, MP):
    T = E_to_T(ρ, E, A, J, MP)
    return -(T * MP.ρ0) / (MP.T0 * ρ * MP.τ1) * eye(3)

def solver_thermal_analytic_ideal(ρ, E, A, J, v, dt, MP):
    """ Solves the thermal impulse ODE analytically in 3D for the ideal gas EOS
    """
    c1 = E - E_2A(ρ, A, MP) - E_3(v)
    c2 = MP.cα2 / 2
    k = 2 * MP.ρ0 / (MP.τ2 * MP.T0 * ρ * MP.cv)
    c1 *= k
    c2 *= k

    # To avoid NaNs if dt>>1
    ea = exp(-c1*dt/2)
    den = 1 - c2/c1 * (1-ea**2) * L2_1D(J)
    ret = J / sqrt(den)
    return ea * ret
