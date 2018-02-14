from numpy import amax, concatenate, diag, dot, eye, sqrt, zeros
from scipy.linalg import eig, solve
from scipy.optimize import newton_krylov

from gpr.misc.functions import reorder
from gpr.misc.structures import Cvec_to_Pclass, Cvec_to_Pvec, Pvec, Pvec_to_Cvec
from gpr.systems.eigenvalues import Xi1, Xi2
from gpr.systems.eigenvectors import eigen
from gpr.systems.primitive import source_prim

from options import nV, STAR_TOL, STIFF_RGFM


def check_star_convergence(QL_, QR_, MPL, MPR):

    PL_ = Cvec_to_Pclass(QL_, MPL)
    PR_ = Cvec_to_Pclass(QR_, MPR)

    cond1 = amax(abs(PL_.Σ()[0] - PR_.Σ()[0])) < STAR_TOL
    cond2 = abs(PL_.T() - PR_.T()) < STAR_TOL

    return cond1 and cond2


def riemann_constraints1(P, d, sgn, MP):
    """ K=R: sgn = -1
        K=L: sgn = 1
        NOTE: Uses atypical ordering

        Extra constraints are:
        dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
        dT = dT/dρ * dρ + dT/dp * dp
        v*L = v*R
        J*L = J*R
    """
    _, Lhat, Rhat = eigen(P, d, 0)
    Lhat = reorder(Lhat.T, order='atypical').T
    Rhat = reorder(Rhat, order='atypical')

    σA = P.dσdA()
    σρ = P.dσdρ()
    Tρ = P.dTdρ()
    Tp = P.dTdp()

    Π1 = σA[d, :, :, 0]
    Π2 = σA[d, :, :, 1]
    Π3 = σA[d, :, :, 2]

    Lhat[:4] = 0
    Lhat[:3, 0] = -σρ[d]
    Lhat[d, 1] = 1
    Lhat[:3, 2:5] = -Π1
    Lhat[:3, 5:8] = -Π2
    Lhat[:3, 8:11] = -Π3
    Lhat[3, 0] = Tρ
    Lhat[3, 1] = Tp
    Lhat[4:8, 11:15] *= -sgn

    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    O = dot(Ξ1, Ξ2)
    w, vl, vr = eig(O, left=1)
    D = diag(sqrt(w.real))
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

    Y0 = dot(Q_1, solve(D, Q))
    Y = -sgn * dot(Y0, dot(Ξ1, X))
    Rhat[11:15, :4] = Y
    Rhat[:, 4:8] = 0
    Rhat[11:15, 4:8] = sgn * Q_1

    return Lhat, Rhat


def star_stepper(QL, QR, dt, MPL, MPR, SL=zeros(nV), SR=zeros(nV)):

    d = 0

    PL = Cvec_to_Pclass(QL, MPL)
    PR = Cvec_to_Pclass(QR, MPR)
    LL, RL = riemann_constraints1(PL, 0, 1, MPL)
    LR, RR = riemann_constraints1(PR, 0, -1, MPR)
    YL = RL[11:15, :4]
    YR = RR[11:15, :4]

    xL = concatenate([PL.Σ()[d], [PL.T()]])
    xR = concatenate([PR.Σ()[d], [PR.T()]])

    Ξ1L = Xi1(PL, d)
    Ξ2L = Xi2(PL, d)
    OL = dot(Ξ1L, Ξ2L)
    Ξ1R = Xi1(PR, d)
    Ξ2R = Xi2(PR, d)
    OR = dot(Ξ1R, Ξ2R)

    _, QL_1 = eig(OL)
    _, QR_1 = eig(OR)
    cL = dot(LL, reorder(SL, order='atypical'))
    cR = dot(LR, reorder(SR, order='atypical'))
    XL = dot(QL_1, cL[4:8])
    XR = dot(QR_1, cR[4:8])

    yL = concatenate([PL.v, [PL.J[d]]])
    yR = concatenate([PR.v, [PR.J[d]]])
    x_ = solve(YL - YR, yR - yL - dt * (XL + XR) + dot(YL, xL) - dot(YR, xR))

    cL[:4] = x_ - xL
    cR[:4] = x_ - xR

    PLvec = reorder(Pvec(PL), order='atypical')
    PRvec = reorder(Pvec(PR), order='atypical')
    PL_vec = dot(RL, cL) + PLvec
    PR_vec = dot(RR, cR) + PRvec
    QL_ = Pvec_to_Cvec(reorder(PL_vec), MPL)
    QR_ = Pvec_to_Cvec(reorder(PR_vec), MPR)
    return QL_, QR_


def star_states_iterative(QL, QR, dt, MPL, MPR):

    #SL = source_prim(QL, MPL)
    #SR = source_prim(QR, MPR)
    QL_, QR_ = star_stepper(QL, QR, dt, MPL, MPR)#, SL, SR)

    while not check_star_convergence(QL_, QR_, MPL, MPR):
        #SL_ = source_prim(QL_, MPL)
        #SR_ = source_prim(QR_, MPR)
        QL_, QR_ = star_stepper(QL_, QR_, dt, MPL, MPR)#, SL_, SR_)

    return QL_, QR_


def riemann_constraints2(P, side, MP):
    """ Extra constraints are:
        dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
        dq = dq/dρ * dρ + dq/dp * dp + dq/dJ * dJ
        v*L = v*R
        T*L = T*R
    """
    _, Lhat, _ = eigen(P, 0, 0)

    ρ = P.ρ
    p = P.p()
    T = P.T()

    q0 = P.q()[0]
    σ0 = P.σ()[0]
    dσdA0 = P.dσdA()[0]

    pINF = MP.pINF
    cα2 = MP.cα2

    if side == 'L':
        Lhat[4:8] = Lhat[:4]

    Lhat[:4] = 0
    Lhat[:3, 0] = -σ0 / ρ
    Lhat[0, 1] = 1
    Lhat[:3, 5:14] = -dσdA0.reshape([3, 9])
    Lhat[3, 0] = -q0 / ρ
    Lhat[3, 1] = q0 / (p + pINF)
    Lhat[3, 14] = cα2 * T

    return Lhat


def star_stepper_obj(x, QL, QR, dt, MPL, MPR):

    X = x.reshape([2, 21])
    ret = zeros([2, 21])

    QL_ = X[0, :17]
    QR_ = X[1, :17]

    PL = Cvec_to_Pclass(QL, MPL)
    PR = Cvec_to_Pclass(QR, MPR)
    PL_ = Cvec_to_Pclass(QL_, MPL)
    PR_ = Cvec_to_Pclass(QR_, MPR)

    pL = Cvec_to_Pvec(QL, MPL)
    pR = Cvec_to_Pvec(QR, MPR)
    pL_ = Cvec_to_Pvec(QL_, MPL)
    pR_ = Cvec_to_Pvec(QR_, MPR)

    ML = riemann_constraints2(PL, 'L', MPL)[:17, :17]
    MR = riemann_constraints2(PR, 'R', MPR)[:17, :17]

    xL_ = X[0, 17:]
    xR_ = X[1, 17:]

    xL = zeros(4)
    xR = zeros(4)

    xL[:3] = PL.Σ()[0]
    xR[:3] = PR.Σ()[0]
    xL[3] = PL.q()[0]
    xR[3] = PR.q()[0]

    ret[0, :4] = dot(ML[:4], pL_ - pL) - (xL_ - xL)
    ret[1, :4] = dot(MR[:4], pR_ - pR) - (xR_ - xR)

    SL = source_prim(QL, MPL)[:17]
    SR = source_prim(QR, MPR)[:17]
    SL_ = source_prim(QL_, MPL)[:17]
    SR_ = source_prim(QR_, MPR)[:17]

    ret[0, 4:17] = dot(ML[4:], (pL_ - pL) - dt / 2 * (SL + SL_))
    ret[1, 4:17] = dot(MR[4:], (pR_ - pR) - dt / 2 * (SR + SR_))

    bR = zeros(4)
    bL = zeros(4)
    bL[:3] = PL_.v
    bR[:3] = PR_.v
    bL[3] = PL_.T()
    bR[3] = PR_.T()

    ret[0, 17:] = xL_ - xR_
    ret[1, 17:] = bL - bR

    return ret.ravel()


def star_states_stiff(QL, QR, dt, MPL, MPR):

    PL = Cvec_to_Pclass(QL, MPL)
    PR = Cvec_to_Pclass(QR, MPR)

    x0 = zeros(42)
    x0[:21] = concatenate([QL, PL.Σ()[0], [PL.q()[0]]])
    x0[21:] = concatenate([QR, PR.Σ()[0], [PR.q()[0]]])

    def obj(x): return star_stepper_obj(x, QL, QR, dt, MPL, MPR)

    ret = newton_krylov(obj, x0).reshape([2, 21])

    return ret[0, :17], ret[1, :17]


def star_states(QL, QR, dt, MPL, MPR):

    if STIFF_RGFM:
        return star_states_stiff(QL, QR, dt, MPL, MPR)
    else:
        return star_states_iterative(QL, QR, dt, MPL, MPR)
