from numpy import amax, array, concatenate, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve

from gpr.misc.functions import reorder
from gpr.misc.structures import State
from gpr.systems.eigenvalues import Xi1, Xi2
from gpr.systems.eigenvectors import eigen
from gpr.variables.eos import total_energy


def Pvec(P, THERMAL):

    if THERMAL:
        NV = 17
    else:
        NV = 14

    ret = zeros(NV)
    ret[0] = P.ρ
    ret[1] = P.p()
    ret[2:5] = P.v
    ret[5:14] = P.A.ravel()
    if THERMAL:
        ret[14:17] = P.J
    return ret


def Pvec_to_Cvec(P, MP):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    A = P[5:14].reshape([3, 3])

    λ = 0

    Q[1] = ρ * total_energy(ρ, P[1], P[2:5], A, P[14:17], λ, MP, VISCOUS=True,
                            THERMAL=True, REACTIVE=False)
    Q[2:5] *= ρ
    Q[14:] *= ρ
    return Q


def check_star_convergence(QL_, QR_, MPL, MPR, STAR_TOL=1e-6):

    PL_ = State(QL_, MPL)
    PR_ = State(QR_, MPR)

    cond1 = amax(abs(PL_.Σ()[0] - PR_.Σ()[0])) < STAR_TOL
    cond2 = abs(PL_.T() - PR_.T()) < STAR_TOL

    return cond1 and cond2


def riemann_constraints(P, sgn, MP):
    """ K=R: sgn = -1
        K=L: sgn = 1
        NOTE: Uses atypical ordering

        Extra constraints are:
        dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
        dT = dT/dρ * dρ + dT/dp * dp
        v*L = v*R
        J*L = J*R
    """
    _, Lhat, Rhat = eigen(P, 0, False, MP)
    Lhat = reorder(Lhat.T, order='atypical').T
    Rhat = reorder(Rhat, order='atypical')

    σA = P.dσdA()
    σρ = P.dσdρ()
    Tρ = P.dTdρ()
    Tp = P.dTdp()

    Lhat[:3, 0] = -σρ[0]
    Lhat[:3, 1] = array([1, 0, 0])
    for i in range(3):
        Lhat[:3, 2 + 3 * i:5 + 3 * i] = -σA[0, :, :, i]
    Lhat[:3, 11:] = 0

    if MP.THERMAL:
        Lhat[3, 0] = Tρ
        Lhat[3, 1] = Tp
        Lhat[3, 2:] = 0

    Lhat[4:8, 11:15] *= -sgn

    Ξ1 = Xi1(P, 0, MP)
    Ξ2 = Xi2(P, 0, MP)
    O = dot(Ξ1, Ξ2)
    w, vl, vr = eig(O, left=1)

    D_1 = diag(1 / sqrt(w.real))
    Q = vl.T
    Q_1 = vr
    I = dot(Q, Q_1)
    Q = solve(I, Q)

    tmp = zeros([5, 5])
    tmp[:4] = Lhat[:4, :5]
    tmp[4] = Lhat[8, :5]
    b = zeros([5, 4])
    b[:4, :4] = eye(4)
    X = solve(tmp, b)
    Rhat[:5, :4] = X

    Y0 = dot(Q_1, dot(D_1, Q))
    Y = -sgn * dot(Y0, dot(Ξ1, X))
    Rhat[11:15, :4] = Y
    Rhat[:, 4:8] = 0
    Rhat[11:15, 4:8] = sgn * Q_1

    return Lhat, Rhat


def star_stepper(QL, QR, MPL, MPR):

    PL = State(QL, MPL)
    PR = State(QR, MPR)
    LL, RL = riemann_constraints(PL, 1, MPL)
    LR, RR = riemann_constraints(PR, -1, MPR)
    YL = RL[11:15, :4]
    YR = RR[11:15, :4]

    xL = concatenate([PL.Σ()[0], [PL.T()]])
    xR = concatenate([PR.Σ()[0], [PR.T()]])

    Ξ1L = Xi1(PL, 0, MPL)
    Ξ2L = Xi2(PL, 0, MPL)
    OL = dot(Ξ1L, Ξ2L)
    Ξ1R = Xi1(PR, 0, MPR)
    Ξ2R = Xi2(PR, 0, MPR)
    OR = dot(Ξ1R, Ξ2R)

    _, QL_1 = eig(OL)
    _, QR_1 = eig(OR)

    yL = concatenate([PL.v, [PL.J[0]]])
    yR = concatenate([PR.v, [PR.J[0]]])
    x_ = solve(YL - YR, yR - yL + dot(YL, xL) - dot(YR, xR))

    cL = zeros(17)
    cR = zeros(17)
    cL[:4] = x_ - xL
    cR[:4] = x_ - xR

    PLvec = reorder(Pvec(PL, MPL.THERMAL), order='atypical')
    PRvec = reorder(Pvec(PR, MPR.THERMAL), order='atypical')
    PL_vec = dot(RL, cL) + PLvec
    PR_vec = dot(RR, cR) + PRvec
    QL_ = Pvec_to_Cvec(reorder(PL_vec), MPL)
    QR_ = Pvec_to_Cvec(reorder(PR_vec), MPR)
    return QL_, QR_


def star_states(QL_, QR_, MPL, MPR):

    while not check_star_convergence(QL_, QR_, MPL, MPR):
        QL_, QR_ = star_stepper(QL_, QR_, MPL, MPR)

    return QL_, QR_
