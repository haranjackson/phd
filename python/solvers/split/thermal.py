from numpy import exp, eye, sqrt

from system.gpr.misc.functions import L2_1D
from system.gpr.variables.eos import E_2A, E_3, dEdJ, E_to_T
from system.gpr.variables.sources import theta2inv


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
    c2 = MP.α2 / 2
    k = 2 * MP.ρ0 / (MP.τ2 * MP.T0 * ρ * MP.cv)
    c1 *= k
    c2 *= k

    # To avoid NaNs if dt>>1
    ea = exp(-c1*dt/2)
    den = 1 - c2/c1 * (1-ea**2) * L2_1D(J)
    ret = J / sqrt(den)
    return ea * ret
