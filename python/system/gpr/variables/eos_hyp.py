from numpy import dot, trace


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

def U_1(G, PAR):
    """ Returns the first component of the thermal energy density, given G
    """
    K0 = PAR.K0
    α = PAR.α
    return K0/(2*α**2) * (I_3(G)**(α/2) - 1)**2

def U_2(G, S, PAR):
    """ Returns the second component of the thermal energy density, given G and entropy
    """
    cv = PAR.cv
    T0 = PAR.T0
    γ = PAR.γ
    return cv * T0 * I_3(G)**(γ/2) * (exp(S/cv) - 1)

def W(G, PAR):
    """ Returns the internal energy due to shear deformations, given the distortion tensor
    """
    B0 = PAR.B0
    β = PAR.β
    return B0/2 * I_3(G)**(β/2) * (I_1(G)**2 / 3 - I_2(G))

def total_energy_hyp(A, S, v, PAR):
    """ Returns the total energy, given the distortion tensor and entropy
    """
    G = dot(A.T, A)
    return U_1(G, PAR) + U_2(G, S, PAR) + W(G, PAR) + dot(v,v) / 2

def density_hyp(A, PAR):
    """ Returns the density, given the distortion tensor
    """
    ρ0 = PAR.ρ0
    return ρ0 * det(A)
