from numpy import array, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve

from gpr.misc.functions import reorder
from gpr.systems.eigenvalues import thermo_acoustic_tensor, Xi1, Xi2
from gpr.systems.jacobians import dQdP, dPdQ
from gpr.variables.wavespeeds import c_0, c_h

from options import nV, VISCOUS, THERMAL


def eig_prim(P, left=1, right=1):
    """ Returns eigenvalues and set of left and right eigenvectors of the
        matrix returned by system_prim
    """
    L = zeros([nV, nV])
    R = zeros([nV, nV])

    ρ = P.ρ
    p = P.p()
    A = P.A
    T = P.T()

    vd = P.v[0]
    σ = P.σ()
    dσdρ = P.dσdρ()
    dσdA = P.dσdA()
    dTdρ = P.dTdρ()
    dTdp = P.dTdp()

    Π1 = dσdA[d, :, :, 0]
    Π2 = dσdA[d, :, :, 1]
    Π3 = dσdA[d, :, :, 2]

    O = thermo_acoustic_tensor(P, 0)
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

        b = array([p, 0, 0]) - σ[d]
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
