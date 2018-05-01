""" Attempt at replacing eigen() with scipy's eig()
    Doesn't appear to work (e.g. with water-gas or pbm-copper)
    I think errors develop over time in eig()
"""
from numpy import amax, array, column_stack, concatenate, diag, dot, eye, \
    sqrt, zeros
from scipy.linalg import eig, inv, solve

from gpr.misc.structures import State
from gpr.opts import THERMAL
from gpr.sys.analytical import ode_solver_cons
from gpr.sys.eigenvalues import Xi1, Xi2
from gpr.sys.primitive import M_prim
from gpr.vars.eos import total_energy
from gpr.vars.wavespeeds import c_0
from gpr.opts import NV


STICK = True
RELAXATION = True
STAR_TOL = 1e-6


n1 = 3 + int(THERMAL)
n2 = 6 + 2 * int(THERMAL)
n3 = 8 + int(THERMAL)
n4 = 11 + int(THERMAL)
n5 = 14 + int(THERMAL)


def Pvec(Q, MP):
    """ Vector of primitive variables
        NOTE: Uses atypical ordering
    """
    P = State(Q, MP)
    ret = Q.copy()

    ret[1] = P.p()
    ret[2:5] /= P.ρ

    if THERMAL:
        ret[14:17] /= P.ρ

    return ret


def Pvec_to_Cvec(P, MP):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()

    ρ = P[0]
    p = P[1]
    v = P[2:5]
    A = P[5:14].reshape([3, 3])
    J = P[14:17]
    λ = 0

    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, MP)
    Q[2:5] *= ρ
    if THERMAL:
        Q[14:17] *= ρ

    return Q


def check_star_convergence(QL_, QR_, MPL, MPR):

    PL_ = State(QL_, MPL)
    PR_ = State(QR_, MPR)

    cond1 = amax(abs(PL_.Σ()[0] - PR_.Σ()[0])) < STAR_TOL

    if THERMAL:
        cond2 = abs(PL_.T() - PR_.T()) < STAR_TOL
    else:
        cond2 = True

    return cond1 and cond2


def Y_matrix(P, MP, sgn):

    ρ = P.ρ
    A = P.A
    σρ = P.dσdρ()
    σA = P.dσdA()

    e0 = array([1, 0, 0])
    Π1 = σA[0, :, :, 0]

    tmp = zeros([5, 5])
    tmp[:3, 0] = -σρ[0]
    tmp[0, 1] = 1
    tmp[:3, 2:5] = -Π1

    if THERMAL:

        tmp[3, 0] = P.dTdρ()
        tmp[3, 1] = P.dTdp()
        tmp[4, 0] = -1 / ρ
        tmp[4, 2:5] = inv(A)[0]

    else:

        p = P.p()
        σ = P.σ()
        c0 = c_0(ρ, p, A, MP)

        B = zeros([2, 3])
        B[0, 0] = ρ
        B[1] = σ[0] - ρ * σρ[0]
        B[1, 0] += ρ * c0**2

        rhs = column_stack((-σρ[0], e0))
        C = solve(Π1, rhs)

        BA_1 = dot(B, inv(A))
        Z = eye(2) - dot(BA_1, C)
        W = concatenate([eye(2), -BA_1], axis=1)
        tmp[3:5] = solve(Z, W)

    b = zeros([5, n1])
    b[:n1, :n1] = eye(n1)
    X = solve(tmp, b)

    Ξ1 = Xi1(P, 0, MP)
    Ξ2 = Xi2(P, 0, MP)

    Ξ = dot(Ξ1, Ξ2)
    w, vr = eig(Ξ)

    D_1 = diag(1 / sqrt(w.real))
    Q_1 = vr
    Q = inv(Q_1)

    Y0 = dot(Q_1, dot(D_1, Q))

    return sgn * dot(Y0, dot(Ξ1, X))


def riemann_constraints(Q, sgn, MP):

    M = M_prim(Q, 0, MP)
    λ, L = eig(M, left=True, right=False)

    λ = λ.real
    L = L.T

    Lhat = L[λ.argsort()]
    Lhat[:n1] *= 0

    P = State(Q, MP)

    σρ = P.dσdρ()
    σA = P.dσdA()

    Lhat[:3, 0] = -σρ[0]
    Lhat[0, 1] = 1
    for i in range(3):
        Lhat[:3, 5 + 3 * i : 8 + 3 * i] = -σA[0, :, i]

    if THERMAL:
        Lhat[3, 0] = P.dTdρ()
        Lhat[3, 1] = P.dTdp()

    Lhat[-n1:, 2:5] *= -sgn
    if THERMAL:
        Lhat[-n1:, 14] *= -sgn

    return Lhat


def star_stepper(QL, QR, MPL, MPR):

    PL = State(QL, MPL)
    PR = State(QR, MPR)

    LhatL = riemann_constraints(QL, -1, MPL)
    LhatR = riemann_constraints(QR, 1, MPR)

    cL = zeros(NV)
    cR = zeros(NV)

    if STICK:

        YL = Y_matrix(PL, MPL, -1)
        YR = Y_matrix(PR, MPR, 1)

        if THERMAL:
            xL = concatenate([PL.Σ()[0], [PL.T()]])
            xR = concatenate([PR.Σ()[0], [PR.T()]])
            yL = concatenate([PL.v, PL.J[:1]])
            yR = concatenate([PR.v, PR.J[:1]])
        else:
            xL = PL.Σ()[0]
            xR = PR.Σ()[0]
            yL = PL.v
            yR = PR.v

        x_ = solve(YL - YR, yR - yL + dot(YL, xL) - dot(YR, xR))

    else:  # slip conditions - only implemented for non-thermal

        if THERMAL:
            YL = Y_matrix(PL, MPL, -1)[[0,3]]
            YR = Y_matrix(PR, MPR, 1)[[0,3]]

            xL = array([PL.Σ()[0], PL.T()])
            xR = array([PR.Σ()[0], PR.T()])
            yL = array([PL.v[0], PL.J[0]])
            yR = array([PR.v[0], PR.J[0]])

            M = YL[:, [0, -1]] - YR[:, [0, -1]]
            x_ = solve(M, yR - yL + dot(YL, xL) - dot(YR, xR))
            x_ = array([x_[0], 0, 0, x_[1]])

        else:
            YL = Y_matrix(PL, MPL, -1)[0]
            YR = Y_matrix(PR, MPR, 1)[0]

            xL = PL.Σ()[0]
            xR = PR.Σ()[0]
            yL = PL.v[0]
            yR = PR.v[0]

            x_ = (yR - yL + dot(YL, xL) - dot(YR, xR)) / (YL - YR)[0]
            x_ = array([x_, 0, 0])

    cL[:n1] = x_ - xL
    cR[:n1] = x_ - xR

    PLvec = Pvec(QL, MPL)
    PRvec = Pvec(QR, MPR)
    PL_vec = solve(LhatL, cL) + PLvec
    PR_vec = solve(LhatR, cR) + PRvec
    QL_ = Pvec_to_Cvec(PL_vec, MPL)
    QR_ = Pvec_to_Cvec(PR_vec, MPR)

    return QL_, QR_


def star_states(QL, QR, MPL, MPR, dt):

    QL_ = QL.copy()
    QR_ = QR.copy()

    while not check_star_convergence(QL_, QR_, MPL, MPR):

        if RELAXATION:
            ode_solver_cons(QL_, dt / 2, MPL)
            ode_solver_cons(QR_, dt / 2, MPR)

        QL_, QR_ = star_stepper(QL_, QR_, MPL, MPR)

    return QL_, QR_
