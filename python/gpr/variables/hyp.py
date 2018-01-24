from numpy import dot, exp, eye, zeros
from numpy.linalg import det, inv

from gpr.misc.functions import I_1, I_2, I_3, gram

from options import nV


I = eye(3)


def U_1(I3, HYP):
    """ Returns the first component of the thermal energy density,
        given the third invariant of G
    """
    K0 = HYP.K0
    α = HYP.α
    return K0 / (2 * α**2) * (I3**(α / 2) - 1)**2


def U_2(I3, S, HYP):
    """ Returns the second component of the thermal energy density,
        given the third invariant of G and entropy
    """
    cv = HYP.cv
    T0 = HYP.T0
    γ = HYP.γ
    return cv * T0 * I3**(γ / 2) * (exp(S / cv) - 1)


def W(I1, I2, I3, HYP):
    """ Returns the internal energy due to shear deformations,
        given the invariants of G
    """
    B0 = HYP.B0
    β = HYP.β
    return B0 / 2 * I3**(β / 2) * (I1**2 / 3 - I2)


def GdU_1dG(I3, HYP):
    """ Returns G * dU1/dG
    """
    K0 = HYP.K0
    α = HYP.α
    return K0 / (2 * α) * (I3**α - I3**(α / 2)) * I


def GdU_2dG(I3, S, HYP):
    """ Returns G * dU2/dG
    """
    cv = HYP.cv
    T0 = HYP.T0
    γ = HYP.γ
    return cv * T0 * γ / 2 * (exp(S / cv) - 1) * I3**(γ / 2) * I


def GdWdG(G, I1, I2, I3, HYP):
    """ Returns G * dW/dG
    """
    B0 = HYP.B0
    β = HYP.β
    const = B0 / 2 * I3**(β / 2)
    return const * ((β / 2) * (I1**2 / 3 - I2) * I - I1 / 3 * G + dot(G, G))


def total_energy_hyp(A, S, v, HYP):
    """ Returns the total energy, given the distortion tensor and entropy
    """
    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)
    return U_1(I3, HYP) + U_2(I3, S, HYP) + W(I1, I2, I3, HYP) + dot(v, v) / 2


def pressure_hyp(ρ, A, S, HYP):
    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)

    K0 = HYP.K0
    α = HYP.α
    cv = HYP.cv
    T0 = HYP.T0
    γ = HYP.γ
    B0 = HYP.B0
    β = HYP.β
    const = B0 / 2 * I3**(β / 2)

    ret = K0 / (2 * α) * (I3**α - I3**(α / 2))
    ret += cv * T0 * γ / 2 * (exp(S / cv) - 1) * I3**(γ / 2)
    ret += const * ((β / 2) * (I1**2 / 3 - I2))
    return 2 * ρ * ret


def Sigma_hyp(ρ, A, S, HYP):
    """ Returns the total stress tensor
    """
    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)
    GdedG = GdU_1dG(I3, HYP) + GdU_2dG(I3, S, HYP) + GdWdG(G, I1, I2, I3, HYP)
    return -2 * ρ * GdedG


def temperature_hyp(S, A, HYP):
    G = gram(A)
    I3 = I_3(G)
    T0 = HYP.T0
    γ = HYP.γ
    cv = HYP.cv
    return T0 * I3**(γ / 2) * exp(S / cv)


def entropy_hyp(E, A, v, HYP):
    cv = HYP.cv
    T0 = HYP.T0
    γ = HYP.γ

    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)

    U2 = E - (U_1(I3, HYP) + W(I1, I2, I3, HYP) + dot(v, v) / 2)
    return cv * log(1 + U2 / (cv * T0 * I3**(γ / 2)))


def Cvec_hyp(F, S, v, HYP):
    """ Returns the vector of conserved variables, given the hyperelastic
        variables
    """
    Q = zeros(nV)

    ρ = HYP.ρ0 / det(F)
    A = inv(F)
    E = total_energy_hyp(A, S, v, HYP)

    Q[0] = ρ
    Q[1] = ρ * E
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()

    return Q
