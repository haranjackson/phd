from numpy import array, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve

from gpr.misc.functions import reorder
from gpr.systems.eigenvalues import thermo_acoustic_tensor, Xi1, Xi2
from gpr.systems.jacobians import dQdP, dPdQ
from gpr.variables.wavespeeds import c_0, c_h
from gpr.variables import mg

from options import nV


def eig_prim(P, d, left=1, right=1):
    """ Returns eigenvalues and set of left and right eigenvectors of the
        matrix returned by system_prim
        NOTE: Might need work for d != 0
    """
    L = zeros([nV, nV])
    R = zeros([nV, nV])

    ρ = P.ρ
    p = P.p()
    A = P.A
    T = P.T()

    vd = P.v[d]
    σ = P.σ()
    dσdρ = P.dσdρ()
    dσdA = P.dσdA()
    dTdρ = P.dTdρ()
    dTdp = P.dTdp()

    Π1 = dσdA[d, :, :, 0]
    Π2 = dσdA[d, :, :, 1]
    Π3 = dσdA[d, :, :, 2]

    O = thermo_acoustic_tensor(P, d)
    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    w, vl, vr = eig(O, left=1)
    sw = sqrt(w.real)
    D = diag(sw)
    Q = vl.T
    Q_1 = vr
    I = dot(Q, Q_1)
    Q = solve(I, Q, overwrite_a=1, check_finite=0)
    DQ = dot(D, Q)

    if right:
        temp = solve(DQ.T, Ξ2.T, overwrite_a=1, overwrite_b=1, check_finite=0).T
        temp2 = Q_1
        R[:5, :4] = temp
        R[:5, 4:8] = temp
        R[11:15, :4] = temp2
        R[11:15, 4:8] = -temp2

        b = - σ[d]
        b[d] += p
        Π1A = dot(Π1, A)
        c = 2 / (solve(Π1A, b, overwrite_a=1, check_finite=0)[0] - 1)
        R[0, 8] = c * ρ
        R[1, 8] = c * p
        R[2:5, 8] = c * solve(Π1, b, overwrite_b=1, check_finite=0)

        R[2:5, 9:12] = -2 * solve(Π1, Π2)
        R[2:5, 12:15] = -2 * solve(Π1, Π3)
        R[5:11, 9:15] = 2 * eye(6)
        R[15:17, 15:17] = 2 * eye(2)

    if left:
        temp = solve(D, dot(Q, Ξ1), overwrite_b=1, check_finite=0)
        temp2 = -solve(D, dot(Q[:, :3], Π2), overwrite_b=1, check_finite=0) / ρ
        temp3 = -solve(D, dot(Q[:, :3], Π3), overwrite_b=1, check_finite=0) / ρ
        temp4 = Q
        L[:4, :5] = temp
        L[4:8, :5] = temp
        L[:4, 5:8] = temp2
        L[4:8, 5:8] = temp2
        L[:4, 8:11] = temp3
        L[4:8, 8:11] = temp3
        L[:4, 11:15] = temp4
        L[4:8, 11:15] = -temp4

        temp = solve(A.T, array([1, 0, 0]), overwrite_b=1, check_finite=0)
        L[8, 0] = -1 / ρ
        L[8, 2:5] = temp
        L[8, 5:8] = dot(temp, solve(Π1, Π2, check_finite=0))
        L[8, 8:11] = dot(temp, solve(Π1, Π3, check_finite=0))

        L[9:15, 5:11] = eye(6)
        L[15:17, 15:17] = eye(2)

    l = array([vd + s for s in sw] + [vd - s for s in sw] + [vd] * 9).real
    return l, reorder(L.T).T, reorder(0.5 * R)


def eig_cons(P):
    """ Returns the eigenvalues and left and right eigenvectors of the
        conserved system
    """
    Λ, L, R = eig_prim(P)
    DPDQ = dPdQ(P)
    DQDP = dQdP(P)
    return Λ, dot(L, DPDQ), dot(DQDP, R)


def eig_cons2(P, d, CONS=1):

    R = zeros([nV, nV])

    ρ = P.ρ
    E = P.E
    v = P.v
    J = P.J
    vd = v[d]
    Γ = mg.Γ_MG(ρ, MP)

    dE_dρ = P.dEdρ()
    dE_dA = P.dEdA()
    dσ_dρ = P.dσdρ()
    dσ_dA = P.dσdA()
    dT_dρ = P.dTdρ()
    dT_dp = P.dTdp()
    H = P.H()

    Π1 = dσ_dA[d, :, :, 0]
    Π2 = dσ_dA[d, :, :, 1]
    Π3 = dσ_dA[d, :, :, 2]

    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    O = dot(Ξ1, Ξ2)
    w, vl, vr = eig(O, left=1)
    sw = sqrt(w.real)
    D = diag(sw)
    Q = vl.T
    Q_1 = vr
    DQ = dot(D, Q)

    tmp = solve(DQ.T, Ξ2.T, overwrite_a=1, overwrite_b=1, check_finite=0).T
    R[:5, :4] = tmp
    R[:5, 4:8] = tmp
    R[11:15, :4] = Q_1
    R[11:15, 4:8] = -Q_1

    b = dT_dp * dσ_dρ[d] + array([dT_dρ, 0, 0])
    R[0, 8] = -dT_dp
    R[1, 8] = dT_dρ
    R[2:5, 8] = solve(Π1, b, check_finite=0)

    R[2:5, 9:12] = -solve(Π1, Π2)
    R[2:5, 12:15] = -solve(Π1, Π3)
    R[5:11, 9:15] = eye(6)
    R[15, 15] = 1
    R[16, 16] = 1

    if CONS:
        tmp = zeros(nV)
        tmp[0] = E + ρ * dE_dρ
        tmp[1] = 1 / Γ
        tmp[2:11] = ρ * dE_dA.ravel(order='F')
        tmp[11:14] = ρ * v
        tmp[14:17] = ρ * H
        R[1] = dot(tmp, R)
        for i in range(3):
            R[11+i] = v[i] * R[0] + ρ * R[11+i]
        for i in range(3):
            R[14+i] = J[i] * R[0] + ρ * R[14+i]

    l = array([vd + s for s in sw] + [vd - s for s in sw] + [vd] * 9).real
    return l, reorder(R)
