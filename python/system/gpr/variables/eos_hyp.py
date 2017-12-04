from numpy import dot, exp, trace
from numpy.linalg import det


def I_1(G):
    """ Returns the first invariant of G
    """
    return trace(G)

def I_2(G):
    """ Returns the second invariant of G
    """
    return 1/2 * (trace(G)**2 - trace(dot(G,G)))

def I_3(G):
    """ Returns the third invariant of G
    """
    return det(G)

def U_1(G, HYP):
    """ Returns the first component of the thermal energy density, given G
    """
    K0 = HYP.K0
    α = HYP.α
    return K0/(2*α**2) * (I_3(G)**(α/2) - 1)**2

def U_2(G, S, HYP):
    """ Returns the second component of the thermal energy density, given G and entropy
    """
    cv = HYP.cv
    T0 = HYP.T0
    γ = HYP.γ
    return cv * T0 * I_3(G)**(γ/2) * (exp(S/cv) - 1)

def W(G, HYP):
    """ Returns the internal energy due to shear deformations, given the distortion tensor
    """
    B0 = HYP.B0
    β = HYP.β
    return B0/2 * I_3(G)**(β/2) * (I_1(G)**2 / 3 - I_2(G))

def total_energy_hyp(A, S, v, HYP):
    """ Returns the total energy, given the distortion tensor and entropy
    """
    G = dot(A.T, A)
    return U_1(G, HYP) + U_2(G, S, HYP) + W(G, HYP) + dot(v,v) / 2

def temperature_hyp(S, A, HYP):
    G = dot(A.T, A)
    I3 = I_3(G)
    T0 = HYP.T0
    γ = HYP.γ
    cv = HYP.cv
    return T0 * I3**(γ/2) * exp(S / cv)
