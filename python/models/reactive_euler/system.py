from numpy import zeros


modelParams = {'γ': 1.4,
               'cv': 2.5,
               'Qc': 1,
               'Ti': 0.25,
               'K0': 250}


def sqnorm(x):
    """ Returns the squared norm of a 3-vector
        NOTE: numpy.linalg.norm and numpy.dot are not currently supported by
        the tangent library, so this function is used instead
    """
    return x[0]**2 + x[1]**2 + x[2]**2


def pressure(ρ, E, v, λ, γ, Qc):
    e = E - sqnorm(v) / 2 - Qc * (1 - λ)
    return (γ - 1) * ρ * e


def temperature(ρ, E, v, λ, Qc, cv):
    e = E - sqnorm(v) / 2 - Qc * (1 - λ)
    return e / cv


def flux(Q, d, model_params):

    ret = zeros(6)

    γ = model_params['γ']
    Qc = model_params['Qc']

    ρ = Q[0]
    E = Q[1] / ρ

    # (30/Mar/18) The tangent library currently has a bug affecting array
    # slicing, so this is used, rather than Q[2:5] / ρ. A ticket has been opened.
    v = [Q[2] / ρ, Q[3] / ρ, Q[4] / ρ]

    λ = Q[5] / ρ
    p = pressure(ρ, E, v, λ, γ, Qc)

    ret[0] = ρ * v[d]
    ret[1] = ρ * v[d] * E + p * v[d]
    ret[2] = ρ * v[d] * v[0]
    ret[3] = ρ * v[d] * v[1]
    ret[4] = ρ * v[d] * v[2]
    ret[2 + d] += p
    ret[5] = ρ * v[d] * λ

    return ret


def reaction_rate(ρ, E, v, λ, cv, Qc, Ti, K0):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    T = temperature(ρ, E, v, λ, Qc, cv)
    return K0 if T > Ti else 0


def source(Q, model_params):

    ret = zeros(6)

    cv = model_params['cv']
    Qc = model_params['Qc']
    Ti = model_params['Ti']
    K0 = model_params['K0']

    ρ = Q[0]
    λ = Q[5] / ρ

    ret[5] = -ρ * λ * reaction_rate(ρ, E, v, λ, cv, Qc, Ti, K0)

    return ret
