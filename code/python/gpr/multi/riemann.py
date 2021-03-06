from numpy import amax, array, column_stack, concatenate, dot, eye, sqrt, zeros
from scipy.linalg import inv, solve

from gpr.misc.functions import reorder
from gpr.misc.structures import State
from gpr.multi.conditions import slip_bcs, stick_bcs
from gpr.multi.rotations import rotation_matrix, rotate_tensors
from gpr.multi.vectors import Pvec, Pvec_to_Cvec
from gpr.sys.analytical import ode_solver_cons
from gpr.sys.eigenvalues import Xi1, Xi2
from gpr.sys.eigenvectors import eigen, decompose_Ξ, get_indexes
from gpr.vars.wavespeeds import c_0


RELAXATION = False
STAR_TOL = 1e-6


n1, n2, n3, n4, n5, n6 = get_indexes()


def q_dims(MP):
    """ Returns characteristic dimensions of heat flux
    """
    α2 = MP.cα2
    ρ0 = MP.ρ0
    T0 = MP.T0
    cv = MP.cv
    return α2 / ρ0 * sqrt(T0**3 / cv)


def check_star_convergence(QL_, QR_, MPL, MPR):

    PL_ = State(QL_, MPL)

    if MPR.EOS > -1:  # not a vacuum
        PR_ = State(QR_, MPR)

        ρ0 = min(MPL.ρ0, MPR.ρ0)
        b02 = min(MPL.b02, MPR.b02)

        cond = amax(abs(PL_.Σ()[0] - PR_.Σ()[0])) / (b02 * ρ0) < STAR_TOL
        cond &= abs(PL_.v[0] - PR_.v[0]) / sqrt(b02) < STAR_TOL
    else:
        ρ0 = MPL.ρ0
        b02 = MPL.b02

        cond = amax(abs(PL_.Σ()[0])) / (b02 * ρ0) < STAR_TOL

    if THERMAL:
        if MPR.EOS > -1:
            q0 = min(q_dims(MPL), q_dims(MPR))
            T0 = min(MPL.T0, MPR.T0)

            cond &= abs(PL_.q()[0] - PR_.q()[0]) / q0 < STAR_TOL
            cond &= abs(PL_.T() - PR_.T()) / T0 < STAR_TOL
        else:
            q0 = q_dims(MPL)

            cond &= abs(PL_.q()[0]) / q0 < STAR_TOL

    return cond


def left_riemann_constraints(P, Lhat, sgn):

    σρ = P.dσdρ()
    σA = P.dσdA()

    Lhat[:3, 0] = -σρ[0]
    Lhat[:3, 1] = array([1, 0, 0])
    for i in range(3):
        Lhat[:3, 2 + 3 * i:5 + 3 * i] = -σA[0, :, :, i]
    Lhat[:3, 11:] = 0

    if THERMAL:
        Lhat[3, 0] = P.dTdρ()
        Lhat[3, 1] = P.dTdp()
        Lhat[3, 2:] = 0

    Lhat[n1:n2, 11:n5] *= -sgn

    return Lhat


def riemann_constraints(P, sgn, MP, left=False):
    """ K=R: sgn = -1
        K=L: sgn = 1
        NOTE: Uses atypical ordering

        Extra constraints are:
        dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
        dT = dT/dρ * dρ + dT/dp * dp
        v*L = v*R
        J*L = J*R
    """
    _, Lhat, Rhat = eigen(P, 0, False, MP, values=False, right=True, left=left,
                          typical_order=False)

    if left:
        Lhat = left_riemann_constraints(P, Lhat, sgn)

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

    Rhat[:5, :n1] = X

    Ξ1 = Xi1(P, 0, MP)
    Ξ2 = Xi2(P, 0, MP)
    Q, Q_1, _, D_1 = decompose_Ξ(Ξ1, Ξ2)
    Y0 = dot(Q_1, dot(D_1, Q))

    Rhat[11:n5, :n1] = -sgn * dot(Y0, dot(Ξ1, X))
    Rhat[:11, n1:n2] = 0
    Rhat[11:n5, n1:n2] = sgn * dot(Q_1, D_1)

    return Lhat, Rhat


def star_stepper(QL_, QR_, MPL, MPR, interfaceType):
    """ Iterates to the next approximation of the star states.
        NOTE: the material on the right may be a vacuum.
    """
    PL = State(QL_, MPL)
    _, RL = riemann_constraints(PL, 1, MPL)

    if THERMAL:
        xL = concatenate([PL.Σ()[0], [PL.T()]])
    else:
        xL = PL.Σ()[0]

    cL = zeros(n6)

    if MPR.EOS > -1:  # not a vacuum

        PR = State(QR_, MPR)
        _, RR = riemann_constraints(PR, -1, MPR)

        if THERMAL:
            xR = concatenate([PR.Σ()[0], [PR.T()]])
        else:
            xR = PR.Σ()[0]

        cR = zeros(n6)

        if interfaceType == 'stick':
            x_ = stick_bcs(RL, RR, PL, PR, xL, xR)

        elif interfaceType == 'slip':
            x_ = slip_bcs(RL, RR, PL, PR, xL, xR)

        cL[:n1] = x_ - xL
        cR[:n1] = x_ - xR

        PRvec = Pvec(PR)
        PR_vec = dot(RR, cR) + PRvec
        QR_ = Pvec_to_Cvec(reorder(PR_vec), MPR)

    else:
        cL[:n1] = - xL
        if THERMAL:
            YL = RL[14, :4]
            cL[3] = (dot(YL[:3], PL.Σ()[0]) - PL.J[0]) / YL[3]
        QR_ = zeros(n6)

    PLvec = Pvec(PL)
    PL_vec = dot(RL, cL) + PLvec
    QL_ = Pvec_to_Cvec(reorder(PL_vec), MPL)

    return QL_, QR_


def star_states(QL, QR, MPL, MPR, dt, n, interfaceType='slip'):

    QL_ = QL[:n6].copy()
    QR_ = QR[:n6].copy()

    R = rotation_matrix(n)
    rotate_tensors(QL_, R)
    rotate_tensors(QR_, R)

    while not check_star_convergence(QL_, QR_, MPL, MPR):

        if RELAXATION:
            ode_solver_cons(QL_, dt / 2, MPL)
            if MPR.EOS > -1:
                ode_solver_cons(QR_, dt / 2, MPR)

        QL_, QR_ = star_stepper(QL_, QR_, MPL, MPR, interfaceType)

    rotate_tensors(QL_, R.T)
    rotate_tensors(QR_, R.T)

    retL = QL.copy()
    retR = QR.copy()
    retL[:n6] = QL_
    retR[:n6] = QR_

    return retL, retR
