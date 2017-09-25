from numpy import amax, concatenate, diag, dot, eye, outer, sqrt, zeros
from scipy.linalg import eig, solve

from auxiliary.funcs import gram
from gpr.eig import primitive_eigs, Xi1mat, thermo_acoustic_tensor
from gpr.matrices.primitive import source_primitive_reordered
from gpr.variables.state import sigma, sigma_A, Sigma, temperature
from gpr.variables.vectors import Cvec_to_Pclass, Pvec_reordered, Pvec_reordered_to_Cvec


starTOL = 1e-8
e0 = zeros(3); e0[0]=1

def check_star_convergence(QL_, QR_, PARL, PARR):

    PL_ = Cvec_to_Pclass(QL_, PARL)
    PR_ = Cvec_to_Pclass(QR_, PARR)
    ΣL_ = Sigma(PL_.p, PL_.ρ, PL_.A, PARL.cs2)[0]
    ΣR_ = Sigma(PR_.p, PR_.ρ, PR_.A, PARR.cs2)[0]
    TL_ = temperature(PL_.ρ, PL_.p, PARL.γ, PARL.pINF, PARL.cv)
    TR_ = temperature(PR_.ρ, PR_.p, PARR.γ, PARR.pINF, PARR.cv)

    return amax(abs(ΣL_-ΣR_)) < starTOL and abs(TL_-TR_) < starTOL

def riemann_constraints(P, sgn, PAR):
    """ K=R: sgn = -1
        K=L: sgn = 1
    """
    _, Lhat, Rhat = primitive_eigs(P, PAR)
    ρ = P.ρ; p = P.p; A = P.A
    γ = PAR.γ; pINF = PAR.pINF; cs2 = PAR.cs2; cv = PAR.cv

    T = temperature(ρ, p, γ, pINF, cv)
    σ0 = sigma(ρ, A, cs2)[0]
    dσdA = sigma_A(ρ, A, cs2)[0]
    Π1 = dσdA[:,:,0]
    Π2 = dσdA[:,:,1]
    Π3 = dσdA[:,:,2]

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
    O = thermo_acoustic_tensor(ρ, gram(A), p, T, 0, PAR)
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

def star_stepper(QL, QR, dt, PARL, PARR, SL=zeros(18), SR=zeros(18)):

    PL = Cvec_to_Pclass(QL, PARL)
    PR = Cvec_to_Pclass(QR, PARR)
    LL, RL = riemann_constraints(PL, 1, PARL)
    LR, RR = riemann_constraints(PR, -1, PARR)
    ΘL = RL[11:15, :4]
    ΘR = RR[11:15, :4]

    ΣL = Sigma(PL.p, PL.ρ, PL.A, PARL.cs2)[0]
    ΣR = Sigma(PR.p, PR.ρ, PR.A, PARR.cs2)[0]
    TL = temperature(PL.ρ, PL.p, PARL.γ, PARL.pINF, PARL.cv)
    TR = temperature(PR.ρ, PR.p, PARR.γ, PARR.pINF, PARR.cv)
    xL = concatenate([ΣL, [TL]])
    xR = concatenate([ΣR, [TR]])

    OL = thermo_acoustic_tensor(PL.ρ, gram(PL.A), PL.p, PL.T, 0, PARL)
    OR = thermo_acoustic_tensor(PR.ρ, gram(PR.A), PR.p, PR.T, 0, PARR)
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
    SL = source_primitive_reordered(QL, PARL)
    SR = source_primitive_reordered(QR, PARR)
    QL_, QR_ = star_stepper(QL, QR, dt, PARL, PARR, SL, SR)
    while not check_star_convergence(QL_, QR_, PARL, PARR):
        SL_ = source_primitive_reordered(QL_, PARL)
        SR_ = source_primitive_reordered(QR_, PARR)
        QL_, QR_ = star_stepper(QL_, QR_, dt, PARL, PARR, SL_, SR_)
    return QL_, QR_
