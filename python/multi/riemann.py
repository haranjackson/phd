from numpy import amax, concatenate, diag, dot, eye, outer, sqrt, zeros
from scipy.linalg import eig, solve

from system.eigenvalues import thermo_acoustic_tensor
from system.gpr.systems.eig import eig_prim, Xi1mat
from system.gpr.misc.structures import Cvec_to_Pclass, Pvec_reordered, Pvec_reordered_to_Cvec
from system.gpr.systems.primitive import source_prim, reordered

from options import nV, STAR_TOL


e0 = zeros(3); e0[0]=1


def check_star_convergence(QL_, QR_, PARL, PARR):

    PL_ = Cvec_to_Pclass(QL_, PARL)
    PR_ = Cvec_to_Pclass(QR_, PARR)

    cond1 = amax(abs(PL_.Σ()[0]-PR_.Σ()[0])) < STAR_TOL
    cond2 = abs(PL_.T-PR_.T) < STAR_TOL

    return cond1 and cond2

def riemann_constraints(P, sgn, PAR):
    """ K=R: sgn = -1
        K=L: sgn = 1
    """
    _, Lhat, Rhat = eig_prim(P)

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
    LL, RL = riemann_constraints(PL, 1, PARL)
    LR, RR = riemann_constraints(PR, -1, PARR)
    ΘL = RL[11:15, :4]
    ΘR = RR[11:15, :4]

    xL = concatenate([PL.Σ()[0], [PL.T]])
    xR = concatenate([PR.Σ()[0], [PR.T]])

    OL = thermo_acoustic_tensor(PL, 0)
    OR = thermo_acoustic_tensor(PR, 0)
    _, QL_1 = eig(OL)
    _, QR_1 = eig(OR)
    cL = dt * dot(LL, SL)
    cR = dt * dot(LR, SR)
    XL = dot(QL_1, cL[4:8])
    XR = dot(QR_1, cR[4:8])

    yL = concatenate([PL.v, [PL.J[0]]])
    yR = concatenate([PR.v, [PR.J[0]]])
    x_ = solve(ΘL-ΘR, yR-yL + dt*(XL+XR) + dot(ΘL,xL) - dot(ΘR,xR))

    cL[:4] = x_ - xL
    cR[:4] = x_ - xR

    PLvec = Pvec_reordered(PL)
    PRvec = Pvec_reordered(PR)
    PL_vec = dot(RL, cL) + PLvec
    PR_vec = dot(RR, cR) + PRvec
    QL_ = Pvec_reordered_to_Cvec(PL_vec, PARL)
    QR_ = Pvec_reordered_to_Cvec(PR_vec, PARR)
    return QL_, QR_

def star_states(QL, QR, dt, PARL, PARR):
    SL = reordered(source_prim(QL, PARL))
    SR = reordered(source_prim(QR, PARR))
    QL_, QR_ = star_stepper(QL, QR, dt, PARL, PARR, SL, SR)
    while not check_star_convergence(QL_, QR_, PARL, PARR):
        SL_ = reordered(source_prim(QL_, PARL))
        SR_ = reordered(source_prim(QR_, PARR))
        QL_, QR_ = star_stepper(QL_, QR_, dt, PARL, PARR, SL_, SR_)
    return QL_, QR_


###### EXPERIMENTAL ######


def conds(P, sgn, PAR):
    """ K=R: sgn = -1
        K=L: sgn = 1
    """
    _, Lhat, Rhat = eig_prim(P)

    ρ = P.ρ
    p = P.p
    T = P.T

    q0 = P.q[0]
    σ0 = P.σ[0]
    dσdA0 = P.dσdA()[0]

    pINF = PAR.pINF
    α2 = PAR.α2

    Π1 = dσdA0[:,:,0]
    Π2 = dσdA0[:,:,1]
    Π3 = dσdA0[:,:,2]

    Lhat[:4] = 0
    Lhat[:3, 0] = -σ0 / ρ
    Lhat[0, 1] = 1
    Lhat[:3, 2:5] = -Π1
    Lhat[:3, 5:8] = -Π2
    Lhat[:3, 8:11] = -Π3
    Lhat[3, 0] = -q0 / ρ
    Lhat[3, 1] = q0 / (p+pINF)
    Lhat[3, 14] = α2 * T
    Lhat[4:8, 11:15] *= -sgn

    return Lhat

def star_stepper2(QL, QR, PARL, PARR, d):
    PL = Cvec_to_Pclass(QL, PARL)
    PR = Cvec_to_Pclass(QR, PARR)
    LL = conds(PL, 1, PARL)
    LR = conds(PR, -1, PARR)

    ΣL = PL.Σ()
    ΣR = PR.Σ()
    Σ_ = (ΣL + ΣR) / 2
    qL = PL.q()
    qR = PR.q()
    q_ = (qL + qR) / 2

    bL = zeros(nV)
    bL[:3] = (Σ_ - ΣL)[d]
    bL[3] = (q_ - qL)[d]
    bR = zeros(nV)
    bR[:3] = (Σ_ - ΣR)[d]
    bR[3] = (q_ - qR)[d]

    PLvec = Pvec_reordered(PL)
    PRvec = Pvec_reordered(PR)

    PL_vec = solve(LL, bL) + PLvec
    PR_vec = solve(LR, bL) + PRvec
    QL_ = Pvec_reordered_to_Cvec(PL_vec, PARL)
    QR_ = Pvec_reordered_to_Cvec(PR_vec, PARR)
    return QL_, QR_

def star_states2(QL, QR, dt, PARL, PARR):
    QL_, QR_ = star_stepper2(QL, QR, PARL, PARR, 0)
    for i in range(10):
        QL_, QR_ = star_stepper2(QL_, QR_, PARL, PARR, 0)
    return QL_, QR_
