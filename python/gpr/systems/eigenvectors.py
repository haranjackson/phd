from numpy import array, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve

from gpr.misc.functions import reorder
from gpr.systems.eigenvalues import Xi1, Xi2
from gpr.systems.jacobians import dQdP, dPdQ
from gpr.variables.wavespeeds import c_0, c_h
from gpr.variables import mg

from options import nV


def eigen(P, d, CONS=1, RIGHT=1, LEFT=1):

    R = zeros([nV, nV])
    L = zeros([nV, nV])

    ρ = P.ρ
    A = P.A
    v = P.v
    vd = v[d]

    σA = P.dσdA()

    Π1 = σA[d, :, :, 0]
    Π2 = σA[d, :, :, 1]
    Π3 = σA[d, :, :, 2]

    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    O = dot(Ξ1, Ξ2)
    w, vl, vr = eig(O, left=1)
    sw = sqrt(w.real)
    D = diag(sw)
    Q = vl.T
    Q_1 = vr
    I = dot(Q, Q_1)
    Q = solve(I, Q)
    DQ = dot(D, Q)

    if RIGHT:

        σρ = P.dσdρ()
        Tρ = P.dTdρ()
        Tp = P.dTdp()

        tmp = solve(DQ.T, Ξ2.T).T
        R[:5, :4] = tmp
        R[:5, 4:8] = tmp
        R[11:15, :4] = Q_1
        R[11:15, 4:8] = -Q_1

        b = Tp * σρ[d] + array([Tρ, 0, 0])
        c = 1 / (solve(dot(Π1, A), b)[d] + Tp / ρ)
        R[0, 8] = -c * Tp
        R[1, 8] = c * Tρ
        R[2:5, 8] = c * solve(Π1, b)

        R[2:5, 9:12] = -solve(Π1, Π2)
        R[2:5, 12:15] = -solve(Π1, Π3)
        R[5:11, 9:15] = eye(6)
        R[15, 15] = 1
        R[16, 16] = 1

    if LEFT:

        tmp = solve(D, dot(Q, Ξ1))
        tmp2 = -solve(D, dot(Q[:, :3], Π2)) / ρ
        tmp3 = -solve(D, dot(Q[:, :3], Π3)) / ρ
        L[:4, :5] = tmp
        L[4:8, :5] = tmp
        L[:4, 5:8] = tmp2
        L[4:8, 5:8] = tmp2
        L[:4, 8:11] = tmp3
        L[4:8, 8:11] = tmp3
        L[:4, 11:15] = Q
        L[4:8, 11:15] = -Q

        tmp = solve(A.T, array([1, 0, 0]))
        L[8, 0] = -1 / ρ
        L[8, 2:5] = tmp
        L[8, 5:8] = dot(tmp, solve(Π1, Π2))
        L[8, 8:11] = dot(tmp, solve(Π1, Π3))

        L[9:15, 5:11] = eye(6)
        L[15:17, 15:17] = eye(2)

    if CONS:
        E = P.E
        J = P.J

        Eρ = P.dEdρ()
        EA = P.dEdA().ravel(order='F')
        H = P.H()

        Γ = mg.Γ_MG(ρ, MP)

        if RIGHT:

            b = zeros(nV)
            b[0] = E + ρ * Eρ
            b[1] = 1 / Γ
            b[2:11] = ρ * EA
            b[11:14] = ρ * v
            b[14:17] = ρ * H

            R[1] = dot(b, R)
            for i in range(3):
                R[11 + i] = v[i] * R[0] + ρ * R[11 + i]
            for i in range(3):
                R[14 + i] = J[i] * R[0] + ρ * R[14 + i]

        if LEFT:

            tmp = norm(v)**2 - (E + ρ * Eρ)
            if MP.THERMAL:
                cα2 = MP.cα2
                tmp += cα2 * norm(J)**2
            Υ = Γ * tmp

            L[:, 0] += Υ * L[:, 1] - 1 / ρ * \
                (dot(L[:, 11:14], v) + dot(L[:, 14:17], J))
            L[:, 1] *= Γ
            for i in range(9):
                L[:, 2 + i] -= ρ * EA[i] * L[:, 1]
            for i in range(3):
                L[:, 11 + i] = 1 / ρ * L[:, 11 + i] - v[i] * L[:, 1]
            for i in range(3):
                L[:, 14 + i] = 1 / ρ * L[:, 14 + i] - H[i] * L[:, 1]

    l = array([vd + s for s in sw] + [vd - s for s in sw] + [vd] * 9).real
    return l, reorder(L.T).T, reorder(R)


def test(Q, d, CONS):

    from gpr.misc.structures import Cvec_to_Pclass
    P = Cvec_to_Pclass(Q, MP)
    l, L, R = eigen(P, d, CONS=CONS, LEFT=1)

    if CONS:
        from gpr.systems.conserved import system_cons
        M = system_cons(Q, d, MP)
    else:
        from gpr.systems.primitive import system_prim
        M = system_prim(Q, d, MP)

    print("Λ:", amax(abs(sort(eigvals(M)) - sort(l))))

    for i in range(17):
        print(i)
        print('L:', amax(abs(dot(L[i], M) - (l[i] * L[i]))[L[i] != 0]))
        print('R:', amax(abs(dot(M, R[:, i]) - (l[i] * R[:, i]))[R[:, i] != 0]))
