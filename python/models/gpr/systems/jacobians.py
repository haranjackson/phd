from numpy import dot, eye, outer, tensordot, zeros

from models.gpr.misc.functions import L2_1D

from options import NV


def dQdP(P):
    """ Returns the Jacobian of the conserved variables with respect to the
        primitive variables
    """
    ret = eye(NV)

    MP = P.MP

    ρ = P.ρ
    v = P.v
    E = P.E

    Eρ = P.dEdρ()
    Ep = P.dEdp()

    ret[1, 0] = E + ρ * Eρ
    ret[1, 1] = ρ * Ep
    ret[1, 2:5] = ρ * v
    ret[2:5, 0] = v
    for i in range(2, 5):
        ret[i, i] = ρ

    if MP.VISCOUS:
        ψ_ = P.dEdA()
        ret[1, 5:14] = ρ * ψ_.ravel()

    if MP.THERMAL:
        J = P.J
        H = P.H()
        ret[1, 14:17] = ρ * H
        ret[14:17, 0] = J
        for i in range(14, 17):
            ret[i, i] = ρ

    if MP.REACTIVE:
        Qc = MP.Qc
        ret[1, 17] = Qc * ρ
        ret[17, 0] = P.λ
        ret[17, 17] *= ρ

    return ret


def dPdQ(P):
    """ Returns the Jacobian of the primitive variables with respect to the
        conserved variables
    """
    ret = eye(NV)

    MP = P.MP

    ρ = P.ρ
    v = P.v
    E = P.E
    J = P.J

    Eρ = P.dEdρ()
    Ep = P.dEdp()
    Γ = 1 / (ρ * Ep)

    tmp = L2_1D(v) - (E + ρ * Eρ)
    if MP.THERMAL:
        cα2 = MP.cα2
        tmp += cα2 * L2_1D(J)
    Υ = Γ * tmp

    ret[1, 0] = Υ
    ret[1, 1] = Γ
    ret[1, 2:5] = -Γ * v
    ret[2:5, 0] = -v / ρ

    for i in range(2, 5):
        ret[i, i] = 1 / ρ

    if MP.VISCOUS:
        ψ_ = P.dEdA()
        ret[1, 5:14] = -Γ * ρ * ψ_.ravel()

    if MP.THERMAL:
        H = P.H()
        ret[1, 14:17] = -Γ * H
        ret[14:17, 0] = -J / ρ
        for i in range(14, 17):
            ret[i, i] = 1 / ρ

    if MP.REACTIVE:
        λ = P.λ
        Qc = MP.Qc
        ret[17, 0] = -λ / ρ
        ret[17, 17] /= ρ
        ret[1, 0] += Γ * Qc * λ
        ret[1, 17] -= Γ * Qc

    return ret


def dFdP(P, d):
    """ Returns the Jacobian of the flux vector with respect to the
        primitive variables
        NOTE: Primitive variables are assumed to be in standard ordering
    """
    MP = P.MP

    ρ = P.ρ
    p = P.p()
    A = P.A
    v = P.v
    E = P.E

    Eρ = P.dEdρ()
    Ep = P.dEdp()

    ρvd = ρ * v[d]

    vv = outer(v, v)
    Ψ = ρ * vv
    Φ = vv
    Δ = (E + ρ * Eρ) * v
    Π = (ρ * Ep + 1) * v

    if MP.VISCOUS:

        σ = P.σ()
        ψ_ = P.dEdA()
        σρ = P.dσdρ()
        σA = P.dσdA()

        Ψ -= σ
        Φ -= σρ
        Ω = ρ * outer(v, ψ_).reshape([3, 3, 3]) - tensordot(v, σA, axes=(0, 0))
        Δ -= dot(σρ, v)

    if MP.THERMAL:

        Tρ = P.dTdρ()
        Tp = P.dTdp()

        H = P.H()
        Δ += Tρ * H
        Π += Tp * H

    ret = zeros([NV, NV])
    ret[0, 0] = v[d]
    ret[0, 2 + d] = ρ
    ret[1, 0] = Δ[d]
    ret[1, 1] = Π[d]
    ret[1, 2:5] = Ψ[d]
    ret[1, 2 + d] += ρ * E + p
    ret[2:5, 0] = Φ[d]
    for i in range(2, 5):
        ret[i, i] = ρvd
    ret[2:5, 2 + d] += ρ * v
    ret[2 + d, 1] = 1

    if MP.VISCOUS:
        ret[1, 5:14] = Ω[d].ravel()
        ret[2:5, 5:14] = -σA[d].reshape([3, 9])
        ret[5 + d, 2:5] = A[0]
        ret[8 + d, 2:5] = A[1]
        ret[11 + d, 2:5] = A[2]
        ret[5 + d, 5:8] = v
        ret[8 + d, 8:11] = v
        ret[11 + d, 11:14] = v

    if MP.THERMAL:
        T = P.T()
        J = P.J
        cα2 = MP.cα2
        ret[1, 14:17] = ρvd * H
        ret[1, 14 + d] += cα2 * T
        ret[14:17, 0] = v[d] * J
        ret[14 + d, 0] += Tρ
        ret[14 + d, 1] = Tp
        ret[14:17, 2 + d] = ρ * J
        for i in range(14, 17):
            ret[i, i] = ρvd

    if MP.REACTIVE:
        λ = P.λ
        Qc = MP.Qc
        ret[17, 0] = v[d] * λ
        ret[17, 2 + d] = ρ * λ
        ret[17, 17] = ρvd
        ret[1, 17] += Qc * ρvd

    return ret


def dSdQ(Q, MP):
    """ WARNING: incomplete, and does not work for plastic solids
    """
    P = Cvec_to_Pclass(Q, MP)
    A = P.A

    ret = zeros([NV, NV])

    if MP.VISCOUS:
        τ1 = MP.τ1
        G = gram(A)
        Grev = gram_rev(A)
        A_devG = AdevG(A, G)
        AinvT = inv(A).T
        AA = einsum('ij,mn', A, A)
        L2A = L2_2D(A)

        ret = 5 / 3 * einsum('ij,mn', A_devG, AinvT) - \
            2 / 3 * AA + AA.swapaxes(1, 3)

        for i in range(3):
            for j in range(3):
                ret[i, j, i, j] -= L2A / 3

        for k in range(3):
            ret[k, :, k, :] += G
            ret[:, k, :, k] += Grev

        ret *= -3 / τ1 * det3(A)**(5 / 3)
        ret[5:14, 5:14] = ret.reshape([9, 9])

    if MP.THERMAL:
        ret[14:17, 14:17] = -MP.α2 * P.θ2_1() * eye(3)

    return ret
