from numpy import array, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve, norm

from gpr.misc.functions import reorder
from gpr.opts import THERMAL
from gpr.sys.eigenvalues import Xi1, Xi2
from gpr.vars import mg
from gpr.vars.wavespeeds import c_0


def get_indexes():

    n1 = 3 + int(THERMAL)
    n2 = 6 + 2 * int(THERMAL)
    n3 = 8 + int(THERMAL)
    n4 = 11 + int(THERMAL)
    n5 = 14 + int(THERMAL)

    return n1, n2, n3, n4, n5


def decompose_Ξ(Ξ1, Ξ2):

    Ξ = dot(Ξ1, Ξ2)
    w, vl, vr = eig(Ξ, left=1)

    sw = sqrt(w.real)
    D = diag(sw)
    D_1 = diag(1 / sw)

    Q = vl.T
    Q_1 = vr
    I = dot(Q, Q_1)
    Q = solve(I, Q)

    return Q, Q_1, D, D_1


def convert_to_conservative(P, R, L, MP):
    """ Effectively multiplies matrices by relevant jacobians
        TODO: Fix for non-thermal systems
    """
    Eρ = P.dEdρ()
    EA = P.dEdA().ravel(order='F')
    H = P.H()

    Γ = mg.Γ_MG(P.ρ, MP)

    if R is not None:

        b = zeros(17)
        b[0] = P.E + P.ρ * Eρ
        b[1] = 1 / Γ
        b[2:11] = P.ρ * EA
        b[11:14] = P.ρ * P.v
        b[14:17] = P.ρ * H

        R[1] = dot(b, R)
        for i in range(3):
            R[11 + i] = P.v[i] * R[0] + P.ρ * R[11 + i]
        for i in range(3):
            R[14 + i] = P.J[i] * R[0] + P.ρ * R[14 + i]

    if L is not None:

        tmp = norm(P.v)**2 - (P.E + P.ρ * Eρ)
        if THERMAL:
            cα2 = MP.cα2
            tmp += cα2 * norm(P.J)**2
        Υ = Γ * tmp

        L[:, 0] += Υ * L[:, 1] - 1 / P.ρ * \
            (dot(L[:, 11:14], P.v) + dot(L[:, 14:17], P.J))
        L[:, 1] *= Γ
        for i in range(9):
            L[:, 2 + i] -= P.ρ * EA[i] * L[:, 1]
        for i in range(3):
            L[:, 11 + i] = 1 / P.ρ * L[:, 11 + i] - P.v[i] * L[:, 1]
        for i in range(3):
            L[:, 14 + i] = 1 / P.ρ * L[:, 14 + i] - H[i] * L[:, 1]


def eigen(P, d, CONS, MP, right=True, left=True, typical_order=True):

    ρ = P.ρ
    A = P.A
    v = P.v
    vd = v[d]

    σA = P.dσdA()
    σρ = P.dσdρ()

    Π1 = σA[d, :, :, 0]
    Π2 = σA[d, :, :, 1]
    Π3 = σA[d, :, :, 2]

    Ξ1 = Xi1(P, d, MP)
    Ξ2 = Xi2(P, d, MP)
    Q, Q_1, D, D_1 = decompose_Ξ(Ξ1, Ξ2)

    n1, n2, n3, n4, n5 = get_indexes()

    e0 = array([1, 0, 0])

    if right:

        R = zeros([17, 17])

        tmp1 = 0.5 * dot(Ξ2, dot(Q_1, D_1**2))
        tmp2 = 0.5 * dot(Q_1, D_1)

        R[:5, :n1] = tmp1
        R[:5, n1:n2] = tmp1
        R[11:n5, :n1] = tmp2
        R[11:n5, n1:n2] = -tmp2

        if THERMAL:
            Tρ = P.dTdρ()
            Tp = P.dTdp()
            b = Tp * σρ[d] + Tρ * e0
            c = 1 / (solve(dot(Π1, A), b)[d] + Tp / ρ)

            R[0, 8] = -c * Tp
            R[1, 8] = c * Tρ
            R[2:5, 8] = c * solve(Π1, b)
            R[15, 15] = 1
            R[16, 16] = 1

        else:
            R[0, 6] = 1
            R[1, 7] = 1
            R[2:5, 6] = -solve(Π1, σρ[0])
            R[2:5, 7] = solve(Π1, e0)

        R[2:5, n3:n4] = -solve(Π1, Π2)
        R[2:5, n4:n5] = -solve(Π1, Π3)
        for i in range(6):
            R[5 + i, n3 + i] = 1

        if typical_order:
            R = reorder(R)

        if not THERMAL:
            R = R[:14, :14]

    else:
        R = None

    if left:

        L = zeros([17, 17])

        tmp1 = dot(Q, Ξ1)
        tmp2 = -dot(Q[:, :3], Π2) / ρ
        tmp3 = -dot(Q[:, :3], Π3) / ρ
        tmp4 = dot(D, Q)

        L[:n1, :5] = tmp1
        L[n1:n2, :5] = tmp1
        L[:n1, 5:8] = tmp2
        L[n1:n2, 5:8] = tmp2
        L[:n1, 8:11] = tmp3
        L[n1:n2, 8:11] = tmp3
        L[:n1, 11:n5] = tmp4
        L[n1:n2, 11:n5] = -tmp4

        if THERMAL:
            tmp = solve(A.T, e0)
            L[8, 0] = -1 / ρ
            L[8, 2:5] = tmp
            L[8, 5:8] = dot(tmp, solve(Π1, Π2))
            L[8, 8:11] = dot(tmp, solve(Π1, Π3))
            L[15, 15] = 1
            L[16, 16] = 1

        else:
            σ = P.σ()
            p = P.p()
            c0 = c_0(ρ, p, A, MP)

            B = zeros([2, 3])
            B[0, 0] = ρ
            B[1] = σ[0] - ρ * σρ[0]
            B[1, 0] += ρ * c0**2

            C = zeros([3, 2])
            C[:, 0] = -solve(Π1, σρ[0])
            C[:, 1] = solve(Π1, e0)

            BA_1 = solve(A.T, B.T).T
            Z = eye(2) - dot(BA_1, C)
            X = zeros([2, 14])
            X[:, :2] = eye(2)
            X[:, 2:5] = -BA_1
            X[:, 5:8] = -dot(BA_1, solve(Π1, Π2))
            X[:, 8:11] = -dot(BA_1, solve(Π1, Π3))
            L[6:8, :14] = solve(Z, X)

        for i in range(6):
            L[n3 + i, 5 + i] = 1

        if typical_order:
            L = reorder(L.T).T

        if not THERMAL:
            L = L[:14, :14]

    else:
        L = None

    if CONS:
        convert_to_conservative(P, R, L, MP)

    l = array([vd + λ for λ in diag(D)] +
              [vd - λ for λ in diag(D)] + [vd] * n3)

    return l, L, R


def test(Q, d, CONS, MP):

    from numpy import amax, sort
    from numpy.linalg import eigvals

    from gpr.misc.structures import State

    P = State(Q, MP)
    l, L, R = eigen(P, d, CONS, MP)

    n = 17 if THERMAL else 14

    if CONS:
        from gpr.sys.conserved import M_cons
        M = M_cons(Q, d, MP)[:n, :n]
    else:
        from gpr.sys.primitive import M_prim
        M = M_prim(Q, d, MP)[:n, :n]

    print("Λ:", amax(abs(sort(eigvals(M)) - sort(l))))

    for i in range(n):
        print(i)
        print('L:', amax(abs(dot(L[i], M) - (l[i] * L[i]))[L[i] != 0]))
        print('R:', amax(
            abs(dot(M, R[:, i]) - (l[i] * R[:, i]))[R[:, i] != 0]))
