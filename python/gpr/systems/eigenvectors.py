from numpy import array, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve, norm

from gpr.misc.functions import reorder
from gpr.systems.eigenvalues import Xi1, Xi2
from gpr.systems.jacobians import dQdP, dPdQ
from gpr.variables.wavespeeds import c_0, c_h
from gpr.variables import mg


def eigen(P, d, CONS, RIGHT=1, LEFT=1):

    R = zeros([17, 17])
    L = zeros([17, 17])

    MP = P.MP

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
    if not MP.THERMAL:
        sw[-1] = 1
    D = diag(sw)
    D_1 = diag(1/sw)

    Q = vl.T
    Q_1 = vr
    I = dot(Q, Q_1)
    Q = solve(I, Q)

    if RIGHT:

        σρ = P.dσdρ()
        Tρ = P.dTdρ()
        Tp = P.dTdp()

        tmp1 = 0.5 * dot(Ξ2, dot(Q_1, D_1**2))
        tmp2 = 0.5 * dot(Q_1, D_1)
        R[:5, :4] = tmp1
        R[:5, 4:8] = tmp1
        R[11:15, :4] = tmp2
        R[11:15, 4:8] = -tmp2

        b = Tp * σρ[d] + array([Tρ, 0, 0])
        c = 1 / (solve(dot(Π1, A), b)[d] + Tp / ρ)
        R[0, 8] = -c * Tp
        R[1, 8] = c * Tρ
        R[2:5, 8] = c * solve(Π1, b)

        R[2:5, 9:12] = -solve(Π1, Π2)
        R[2:5, 12:15] = -solve(Π1, Π3)
        for i in range(6):
            R[5 + i, 9 + i] = 1
        R[15, 15] = 1
        R[16, 16] = 1

    if LEFT:

        tmp1 = dot(Q, Ξ1)
        tmp2 = -dot(Q[:, :3], Π2) / ρ
        tmp3 = -dot(Q[:, :3], Π3) / ρ
        tmp4 = dot(D, Q)
        L[:4, :5] = tmp1
        L[4:8, :5] = tmp1
        L[:4, 5:8] = tmp2
        L[4:8, 5:8] = tmp2
        L[:4, 8:11] = tmp3
        L[4:8, 8:11] = tmp3
        L[:4, 11:15] = tmp4
        L[4:8, 11:15] = -tmp4

        tmp = solve(A.T, array([1, 0, 0]))
        L[8, 0] = -1 / ρ
        L[8, 2:5] = tmp
        L[8, 5:8] = dot(tmp, solve(Π1, Π2))
        L[8, 8:11] = dot(tmp, solve(Π1, Π3))

        for i in range(6):
            L[9 + i, 5 + i] = 1
        L[15, 15] = 1
        L[16, 16] = 1

    if CONS:
        # DOESNT WORK IF NOT THERMAL

        E = P.E
        J = P.J

        Eρ = P.dEdρ()
        EA = P.dEdA().ravel(order='F')
        H = P.H()

        Γ = mg.Γ_MG(ρ, MP)

        if RIGHT:

            b = zeros(17)
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
    l, L, R = eigen(P, d, CONS=CONS)

    if CONS:
        from gpr.systems.conserved import system_cons
        M = system_cons(Q, d, MP)[:17, :17]
    else:
        from gpr.systems.primitive import system_prim
        M = system_prim(Q, d, MP)[:17, :17]

    print("Λ:", amax(abs(sort(eigvals(M)) - sort(l))))

    for i in range(17):
        print(i)
        print('L:', amax(abs(dot(L[i], M) - (l[i] * L[i]))[L[i] != 0]))
        print('R:', amax(abs(dot(M, R[:, i]) - (l[i] * R[:, i]))[R[:, i] != 0]))
