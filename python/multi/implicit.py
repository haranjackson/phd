from numpy import dot, zeros
from scipy.optimize import newton_krylov, root, leastsq

from solvers.basis import end_values
from solvers.dg.dg import rhs
from solvers.dg.matrices import system_matrices
from system.gpr.misc.objects import material_parameters
from system.gpr.misc.structures import Cvec_to_Pclass, Cvec
from options import nV, N1, NT


W, U, _, _, _ = system_matrices()
ENDVALS = end_values()


def obj(x, WwL, WwR, dt, PARL, PARR):

    nX = NT*nV

    qL = x[0 : nX].reshape([NT,nV])
    qR = x[nX : 2*nX].reshape([NT,nV])
    ρvL = x[2*nX : 2*nX+3*N1].reshape([N1,3])
    ρvR = x[2*nX+3*N1 : 2*nX+6*N1].reshape([N1,3])

    ret = zeros(2*nX+6*N1)

    ret[0 : nX]    = (dot(U, qL) - rhs(qL, WwL, dt, PARL, 0)).ravel()
    ret[nX : 2*nX] = (dot(U, qR) - rhs(qR, WwR, dt, PARR, 0)).ravel()

    qL_ = dot(ENDVALS[:,1], qL.reshape([N1,N1,nV]))
    qR_ = dot(ENDVALS[:,0], qR.reshape([N1,N1,nV]))

    qL_[:,2:5] += ρvL
    qR_[:,2:5] += ρvR

    vL_ = zeros([N1,3])
    vR_ = zeros([N1,3])
    ΣL_ = zeros([N1,3])
    ΣR_ = zeros([N1,3])
    for i in range(N1):
        PL_ = Cvec_to_Pclass(qL_[i], PARL)
        PR_ = Cvec_to_Pclass(qR_[i], PARR)
        vL_[i] = PL_.v
        vR_[i] = PR_.v
        ΣL_[i] = PL_.Σ()[0]
        ΣR_[i] = PR_.Σ()[0]

    ret[2*nX : 2*nX+3*N1]      = (vL_-vR_).ravel()
    #ret[2*nX+3*N1 : 2*nX+6*N1] = (ΣL_-ΣR_).ravel()

    return ret

if __name__ == "__main__":

    PAR = material_parameters(γ=1.4, pINF=0, cv=1, ρ0=1, p0=1, cs=1, α=1,
                              μ=1e-2, Pr=0.75)
    ρL = 1
    pL = 1
    vL = zeros(3)
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)
    PARL = PAR

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)
    PARR = PAR

    QL = Cvec(ρL, pL, vL, AL, JL, 0, PARR)
    QR = Cvec(ρR, pR, vR, AR, JR, 0, PARL)
    PL = Cvec_to_Pclass(QL, PARL)
    PR = Cvec_to_Pclass(QR, PARL)

    wL = array([QL for i in range(N1)])
    wR = array([QR for i in range(N1)])
    WwL = dot(W, wL)
    WwR = dot(W, wR)
    qL0 = array([wL for i in range(N1)])
    qR0 = array([wR for i in range(N1)])

    nX = NT*nV
    x0 = zeros(2*nX+6*N1)
    x0[0 : nX]    = qL0.ravel()
    x0[nX : 2*nX] = qR0.ravel()

    dt = 0.01

    f = lambda x : obj(x, WwL, WwR, dt, PAR, PAR)
    ret = newton_krylov(f, x0)

    qL  = ret[0 : nX].reshape([N1,N1,nV])
    qR  = ret[nX : 2*nX].reshape([N1,N1,nV])
    ρvL = ret[2*nX : 2*nX+3*N1].reshape([N1,3])
    ρvR = ret[2*nX+3*N1 : 2*nX+6*N1].reshape([N1,3])
