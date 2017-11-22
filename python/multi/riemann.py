from scipy.optimize import newton_krylov
from numpy import amax, concatenate, diag, dot, eye, outer, sqrt, zeros
from scipy.linalg import eig, solve

from system.eigenvalues import thermo_acoustic_tensor
from system.gpr.systems.eig import eig_prim, Xi1mat
from system.gpr.misc.functions import reorder
from system.gpr.misc.structures import Cvec_to_Pclass, Cvec_to_Pvec, Pvec, Pvec_to_Cvec
from system.gpr.systems.primitive import source_prim

from options import nV, STAR_TOL, STIFF_RGFM


e0 = zeros(3); e0[0]=1


def check_star_convergence(QL_, QR_, PARL, PARR):

    PL_ = Cvec_to_Pclass(QL_, PARL)
    PR_ = Cvec_to_Pclass(QR_, PARR)

    cond1 = amax(abs(PL_.Σ()[0]-PR_.Σ()[0])) < STAR_TOL
    cond2 = abs(PL_.T-PR_.T) < STAR_TOL

    return cond1 and cond2

def riemann_constraints1(P, sgn, PAR):
    """ K=R: sgn = -1
        K=L: sgn = 1
        NOTE: Uses atypical ordering

        Extra constraints are:
        dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
        dT = dT/dρ * dρ + dT/dp * dp
        v*L = v*R
        J*L = J*R
    """
    _, Lhat, Rhat = eig_prim(P)
    Lhat = reorder(Lhat.T, order='atypical').T
    Rhat = reorder(Rhat, order='atypical')

    ρ = P.ρ
    p = P.p
    A = P.A
    T = P.T

    σ0 = P.σ[0]
    dσdA0 = P.dσdA()[0]

    pINF = PAR.pINF

    Π1 = dσdA0[:,:,0]
    Π2 = dσdA0[:,:,1]
    Π3 = dσdA0[:,:,2]

    Lhat[:4] = 0
    Lhat[:3, 0] = -σ0 / ρ
    Lhat[0, 1] = 1
    Lhat[:3, 2:5] = -Π1
    Lhat[:3, 5:8] = -Π2
    Lhat[:3, 8:11] = -Π3
    Lhat[3, 0] = -T / ρ
    Lhat[3, 1] = T / (p+pINF)
    Lhat[4:8, 11:15] *= -sgn

    Z0 = eye(3,4)
    Z0[0,3] = -(p+pINF)/T
    Z1 = outer((p+pINF)*e0 - σ0, e0) - dot(Π1, A)
    Z2 = solve(Z1,Z0)
    X = dot(A,Z2)
    a = Z2[0]

    Ξ1 = Xi1mat(ρ, p, T, pINF, σ0, Π1)
    O = thermo_acoustic_tensor(P, 0)
    w, vl, vr = eig(O, left=1)
    D = diag(sqrt(w.real))
    Q = vl.T
    Q_1 = vr
    I = dot(Q,Q_1)
    Q = solve(I, Q, overwrite_a=1, check_finite=0)

    Rhat[0,:4] = ρ * a
    Rhat[1,:4] = (p+pINF) * a
    Rhat[1,3] += (p+pINF) / T
    Rhat[2:5, :4] = X

    Y0 = -sgn * dot(Q_1, solve(D, Q))
    Y = dot(Y0, dot(Ξ1, Rhat[:5,:4]))
    Rhat[11:15, :4] = Y
    Rhat[:,4:8] = 0
    Rhat[11:15, 4:8] = sgn * Q_1

    return Lhat, Rhat

def star_stepper(QL, QR, dt, PARL, PARR, SL=zeros(nV), SR=zeros(nV)):

    PL = Cvec_to_Pclass(QL, PARL)
    PR = Cvec_to_Pclass(QR, PARR)
    LL, RL = riemann_constraints1(PL, 1, PARL)
    LR, RR = riemann_constraints1(PR, -1, PARR)
    YL = RL[11:15, :4]
    YR = RR[11:15, :4]

    xL = concatenate([PL.Σ()[0], [PL.T]])
    xR = concatenate([PR.Σ()[0], [PR.T]])

    OL = thermo_acoustic_tensor(PL, 0)
    OR = thermo_acoustic_tensor(PR, 0)
    _, QL_1 = eig(OL)
    _, QR_1 = eig(OR)
    cL = dot(LL, reorder(SL,order='atypical'))
    cR = dot(LR, reorder(SR,order='atypical'))
    XL = dot(QL_1, cL[4:8])
    XR = dot(QR_1, cR[4:8])

    yL = concatenate([PL.v, [PL.J[0]]])
    yR = concatenate([PR.v, [PR.J[0]]])
    x_ = solve(YL-YR, yR-yL - dt*(XL+XR) + dot(YL,xL) - dot(YR,xR))

    cL[:4] = x_ - xL
    cR[:4] = x_ - xR

    PLvec = reorder(Pvec(PL), order='atypical')
    PRvec = reorder(Pvec(PR), order='atypical')
    PL_vec = dot(RL, cL) + PLvec
    PR_vec = dot(RR, cR) + PRvec
    QL_ = Pvec_to_Cvec(reorder(PL_vec), PARL)
    QR_ = Pvec_to_Cvec(reorder(PR_vec), PARR)
    return QL_, QR_

def star_states_iterative(QL, QR, dt, PARL, PARR):
    SL = source_prim(QL, PARL)
    SR = source_prim(QR, PARR)
    QL_, QR_ = star_stepper(QL, QR, dt, PARL, PARR, SL, SR)
    while not check_star_convergence(QL_, QR_, PARL, PARR):
        SL_ = source_prim(QL_, PARL)
        SR_ = source_prim(QR_, PARR)
        QL_, QR_ = star_stepper(QL_, QR_, dt, PARL, PARR, SL_, SR_)
    return QL_, QR_


def riemann_constraints2(P, side, PAR):
    """ Extra constraints are:
        dΣ = dΣ/dρ * dρ + dΣ/dp * dp + dΣ/dA * dA
        dq = dq/dρ * dρ + dq/dp * dp + dq/dJ * dJ
        v*L = v*R
        T*L = T*R
    """
    _, Lhat, _ = eig_prim(P)

    ρ = P.ρ
    p = P.p
    T = P.T

    q0 = P.q[0]
    σ0 = P.σ[0]
    dσdA0 = P.dσdA()[0]

    pINF = PAR.pINF
    α2 = PAR.α2

    if side=='L':
        Lhat[4:8] = Lhat[:4]

    Lhat[:4] = 0
    Lhat[:3, 0] = -σ0 / ρ
    Lhat[0, 1] = 1
    Lhat[:3, 5:14] = -dσdA0.reshape([3,9])
    Lhat[3, 0] = -q0 / ρ
    Lhat[3, 1] = q0 / (p+pINF)
    Lhat[3, 14] = α2 * T

    return Lhat

def star_stepper_obj(x, QL, QR, dt, PARL, PARR):

    X = x.reshape([2,21])
    ret = zeros([2,21])

    QL_ = X[0,:17]
    QR_ = X[1,:17]

    PL = Cvec_to_Pclass(QL, PARL)
    PR = Cvec_to_Pclass(QR, PARR)
    PL_ = Cvec_to_Pclass(QL_, PARL)
    PR_ = Cvec_to_Pclass(QR_, PARR)

    pL = Cvec_to_Pvec(QL, PARL)
    pR = Cvec_to_Pvec(QR, PARR)
    pL_ = Cvec_to_Pvec(QL_, PARL)
    pR_ = Cvec_to_Pvec(QR_, PARR)

    ML = riemann_constraints2(PL, 'L', PARL)[:17,:17]
    MR = riemann_constraints2(PR, 'R', PARR)[:17,:17]

    xL_ = X[0,17:]
    xR_ = X[1,17:]

    xL = zeros(4)
    xR = zeros(4)

    xL[:3] = PL.Σ()[0]
    xR[:3] = PR.Σ()[0]
    xL[3] = PL.q[0]
    xR[3] = PR.q[0]

    ret[0,:4] = dot(ML[:4], pL_-pL) - (xL_-xL)
    ret[1,:4] = dot(MR[:4], pR_-pR) - (xR_-xR)

    SL = source_prim(QL, PARL)[:17]
    SR = source_prim(QR, PARR)[:17]
    SL_ = source_prim(QL_, PARL)[:17]
    SR_ = source_prim(QR_, PARR)[:17]

    ret[0,4:17] = dot(ML[4:], (pL_ - pL) - dt/2 * (SL + SL_))
    ret[1,4:17] = dot(MR[4:], (pR_ - pR) - dt/2 * (SR + SR_))

    bR = zeros(4)
    bL = zeros(4)
    bL[:3] = PL_.v
    bR[:3] = PR_.v
    bL[3] = PL_.T
    bR[3] = PR_.T

    ret[0,17:] = xL_ - xR_
    ret[1,17:] = bL - bR

    return ret.ravel()

def star_states_stiff(QL, QR, dt, PARL, PARR):

    PL = Cvec_to_Pclass(QL, PARL)
    PR = Cvec_to_Pclass(QR, PARR)

    x0 = zeros(42)
    x0[:21] = concatenate([QL, PL.Σ()[0], [PL.q[0]]])
    x0[21:] = concatenate([QR, PR.Σ()[0], [PR.q[0]]])

    obj = lambda x: star_stepper_obj(x, QL, QR, dt, PARL, PARR)

    ret = newton_krylov(obj, x0).reshape([2,21])

    return ret[0,:17], ret[1,:17]


def star_states(QL, QR, dt, PARL, PARR):

    if STIFF_RGFM:
        return star_states_stiff(QL, QR, dt, PARL, PARR)
    else:
        return star_states_iterative(QL, QR, dt, PARL, PARR)
