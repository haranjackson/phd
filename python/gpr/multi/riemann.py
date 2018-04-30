from numpy import amax, array, concatenate, dot, eye, zeros
from scipy.linalg import solve

from gpr.misc.functions import reorder
from gpr.misc.structures import State
from gpr.opts import THERMAL
from gpr.sys.analytical import ode_solver_cons
from gpr.sys.eigenvalues import Xi1, Xi2
from gpr.sys.eigenvectors import eigen, decompose_Ξ, get_indexes
from gpr.vars.eos import total_energy


RELAXATION = True
STAR_TOL = 1e-6


def Pvec(P):
    """ Vector of primitive variables
        NOTE: Uses atypical ordering
    """
    if THERMAL:
        ret = zeros(17)
        ret[14:17] = P.J
    else:
        ret = zeros(14)

    ret[0] = P.ρ
    ret[1] = P.p()
    ret[2:11] = P.A.ravel(order='F')
    ret[11:14] = P.v

    return ret


def Pvec_to_Cvec(P, MP):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    A = P[5:14].reshape([3, 3])

    λ = 0

    Q[1] = ρ * total_energy(ρ, P[1], P[2:5], A, P[14:17], λ, MP)
    Q[2:5] *= ρ
    Q[14:] *= ρ

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
    _, Lhat, Rhat = eigen(P, 0, False, MP, typical_order=False)

    σA = P.dσdA()
    σρ = P.dσdρ()
    Tρ = P.dTdρ()
    Tp = P.dTdp()

    Lhat[:3, 0] = -σρ[0]
    Lhat[:3, 1] = array([1, 0, 0])
    for i in range(3):
        Lhat[:3, 2 + 3 * i:5 + 3 * i] = -σA[0, :, :, i]
    Lhat[:3, 11:] = 0

    n1, n2, n3, n4, n5 = get_indexes()

    if THERMAL:
        Lhat[3, 0] = Tρ
        Lhat[3, 1] = Tp
        Lhat[3, 2:] = 0
        tmp = Lhat[array([0, 1, 2, 3, 8]), :5]
    else:
        tmp = Lhat[array([0, 1, 2, 6, 7]), :5]

    Lhat[n1:n2, 11:n5] *= -sgn

    Ξ1 = Xi1(P, 0, MP)
    Ξ2 = Xi2(P, 0, MP)
    Q, Q_1, _, D_1 = decompose_Ξ(Ξ1, Ξ2)

    b = zeros([5, n1])
    b[:n1, :n1] = eye(n1)
    X = solve(tmp, b)
    Rhat[:5, :n1] = X

    Y0 = dot(Q_1, dot(D_1, Q))
    Rhat[11:n5, :n1] = -sgn * dot(Y0, dot(Ξ1, X))
    Rhat[:11, n1:n2] = 0
    Rhat[11:n5, n1:n2] = sgn * dot(Q_1, D_1)

    return Lhat, Rhat


def star_stepper(QL, QR, MPL, MPR, STICK=True):

    n = 17 if THERMAL else 14
    n1, n2, n3, n4, n5 = get_indexes()

    PL = State(QL, MPL)
    PR = State(QR, MPR)

    _, RL = riemann_constraints(PL, 1, MPL)
    _, RR = riemann_constraints(PR, -1, MPR)

    cL = zeros(n)
    cR = zeros(n)

    if STICK:

        YL = RL[11:n5, :n1]
        YR = RR[11:n5, :n1]

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
        cL[:n1] = x_ - xL
        cR[:n1] = x_ - xR

    else:  # slip conditions - only implemented for non-thermal

        if THERMAL:
            YL = RL[[11, 14], :n1]
            YR = RR[[11, 14], :n1]

            xL = array([PL.Σ()[0], PL.T()])
            xR = array([PR.Σ()[0], PR.T()])
            yL = array([PL.v[0], PL.J[0]])
            yR = array([PR.v[0], PR.J[0]])

            M = YL[:, [0, -1]] - YR[:, [0, -1]]
            x_ = solve(M, yR - yL + dot(YL, xL) - dot(YR, xR))
            x_ = array([x_[0], 0, 0, x_[1]])

        else:
            YL = RL[11, :n1]
            YR = RR[11, :n1]

            xL = PL.Σ()[0]
            xR = PR.Σ()[0]
            yL = PL.v[0]
            yR = PR.v[0]

            x_ = (yR - yL + dot(YL, xL) - dot(YR, xR)) / (YL - YR)[0]
            x_ = array([x_, 0, 0])

        cL[:n1] = x_ - xL
        cR[:n1] = x_ - xR

    PLvec = Pvec(PL)
    PRvec = Pvec(PR)
    PL_vec = dot(RL, cL) + PLvec
    PR_vec = dot(RR, cR) + PRvec
    QL_ = Pvec_to_Cvec(reorder(PL_vec), MPL)
    QR_ = Pvec_to_Cvec(reorder(PR_vec), MPR)

    return QL_, QR_


def star_states(QL_, QR_, MPL, MPR, dt):

    while not check_star_convergence(QL_, QR_, MPL, MPR):

        if RELAXATION:
            ode_solver_cons(QL_, dt / 2, MPL)
            ode_solver_cons(QR_, dt / 2, MPR)

        QL_, QR_ = star_stepper(QL_, QR_, MPL, MPR)

    return QL_, QR_
