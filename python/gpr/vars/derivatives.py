from gpr.misc.functions import AdevG, dev, gram, L2_2D
from gpr.vars import mg
from gpr.vars.shear import c_s2, C_0, dC_0dρ


def dEdρ(ρ, p, A, MP):
    """ Returns the partial derivative of E by ρ (holding p,A constant)
    """
    G = gram(A)
    dC0dρ = dC_0dρ(ρ, MP)
    ret = mg.dedρ(ρ, p, MP) + dC0dρ / 4 * L2_2D(dev(G))
    return ret


def dEdp(ρ, MP):
    """ Returns the partial derivative of E by p (holding ρ constant)
    """
    return mg.dedp(ρ, MP)


def dEdA(ρ, A, MP):
    """ Returns the partial derivative of E by A (holding ρ,s constant)
    """
    C0 = C_0(ρ, MP)
    G = gram(A)
    return C0 * AdevG(A, G)


def dEdA_s(ρ, A, MP):
    """ Returns the partial derivative of E by A (holding ρ,s constant)
    """
    cs2 = c_s2(ρ, MP)
    G = gram(A)
    return cs2 * AdevG(A, G)


def dEdJ(J, MP):
    """ Returns the partial derivative of E by J
    """
    cα2 = MP.cα2
    return cα2 * J


def dTdρ(ρ, p, MP):
    """ Returns the partial derivative of T by ρ
    """
    return mg.dTdρ(ρ, p, MP)


def dTdp(ρ, MP):
    """ Returns the partial derivative of T by p
    """
    return mg.dTdp(ρ, MP)
