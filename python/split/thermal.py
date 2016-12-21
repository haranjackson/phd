from numpy import exp, eye, log, sign, sqrt

from auxiliary.funcs import L2_1D
from gpr.variables.eos import E_2A, E_3, energy_to_temperature
from gpr.variables.vectors import primitive


def jac_J(J, ρ, E, A, PAR):
    T = energy_to_temperature(E, A, J, PAR)
    return -(T * PAR.ρ0) / (PAR.T0 * ρ * PAR.τ1) * eye(3)

def solver_thermal_analytic_constp(ρ, Q, dt, PAR, SYS):
    """ Returns the analytic solution to the thermal impulse ODE, assuming pressure is constant
        over the timescale of the ODE.
        NB This may not be a good assumption.
    """
    P0 = primitive(Q, PAR, SYS)
    return ρ * exp(-(P0.T * PAR.ρ0 * dt)/(PAR.T0 * ρ * PAR.τ2)) * P0.J

def solver_thermal_analytic_ideal(ρ, E, A, J, v, dt, PAR):
    """ Solves the thermal impulse ODE analytically in 3D for the ideal gas EOS
    """
    c1 = E - E_2A(A, PAR.cs2) - E_3(v)
    c2 = PAR.α2 / 2
    k = 2 * PAR.ρ0 / (PAR.τ2 * PAR.T0 * ρ * PAR.cv)
    c1 *= k
    c2 *= k

    ea = exp(c1*dt)
    den = ea - c2/c1*(ea-1)*L2_1D(J)
    return J * sqrt(1 / den)
