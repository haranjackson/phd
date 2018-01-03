from numpy import dot, eye

from system.gpr.misc.functions import AdevG, gram
from system.gpr.variables.mg import e_ref, p_ref, Γ_MG
from system.gpr.variables.eos import E_2A, E_2J, E_3, E_R, dEdA
from options import VISCOUS, THERMAL, REACTIVE


def pressure(ρ, E, v, A, J, MP, λ=None):
    """ Returns the pressure under the Mie-Gruneisen EOS
    """
    E1 = E - E_3(v)

    if VISCOUS:
        E1 -= E_2A(ρ, A, MP)

    if THERMAL:
        E1 -= E_2J(J, MP)

    if REACTIVE:
        E1 -= E_R(λ, MP)

    Γ = Γ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    er = e_ref(ρ, MP)

    return (E1 - er) * ρ * Γ + pr

def temperature(ρ, p, MP):
    """ Returns the temperature under the Mie-Gruneisen EOS
    """
    cv = MP.cv
    Γ = Γ_MG(ρ, MP)
    pr = p_ref(ρ, MP)
    return (p - pr) / (ρ * Γ * cv)

def heat_flux(T, J, MP):
    """ Returns the heat flux vector
    """
    α2 = MP.α2
    return α2 * T * J

def sigma(ρ, A, MP):
    """ Returns the symmetric viscous shear stress tensor
    """
    return -ρ * dot(A.T, dEdA(ρ, A, MP))

def dsigmadA(ρ, A, MP):
    """ Returns T_ijmn = dσ_ij / dA_mn, holding ρ constant.
        NOTE: Only valid for EOS with E_2A = cs2/4 * (ρ/ρ0)**(β/2) * (devG)**2
    """
    cs2 = MP.cs2
    β = MP.β

    G = gram(A)
    AdevGT = AdevG(A,G).T
    GA = dot(G[:,:,None], A[:,None])
    ret = GA.swapaxes(0,3) + GA.swapaxes(1,3) - 2/3 * GA

    for i in range(3):
        ret[i, :, :, i] += AdevGT
        ret[:, i, :, i] += AdevGT

    return -ρ * cs2 * (ρ/ρ0)**β * ret

def Sigma(p, ρ, A, MP):
    """ Returns the total symmetric stress tensor
    """
    return p * eye(3) - sigma(ρ, A, MP)
