from numpy import array, dot, zeros
from scipy.optimize import newton_krylov, leastsq, root, anderson

from solvers.basis import end_values, derivative_values, quad
from solvers.dg.dg import rhs
from solvers.dg.matrices import system_matrices
from system.gpr.misc.objects import material_parameters
from system.gpr.misc.structures import Cvec_to_Pclass, Cvec
from options import nV, N1, NT


W, U, _, _, _ = system_matrices()
ENDVALS = end_values()
DERVALS = derivative_values()
NODES, _, _ = quad()


def obj_eul(x, WwL, WwR, dt, MPL, MPR):

    nX = NT*nV

    qL = x[0 : nX].reshape([NT,nV])
    qR = x[nX : 2*nX].reshape([NT,nV])
    ρvL = x[2*nX : 2*nX+3*N1].reshape([N1,3])
    ρvR = x[2*nX+3*N1 : 2*nX+6*N1].reshape([N1,3])

    ret = zeros(2*nX+6*N1)

    ret[0 : nX]    = (dot(U, qL) - rhs(qL, WwL, dt, MPL, 0)).ravel()
    ret[nX : 2*nX] = (dot(U, qR) - rhs(qR, WwR, dt, MPR, 0)).ravel()

    qL_ = dot(ENDVALS[:,1], qL.reshape([N1,N1,nV]))
    qR_ = dot(ENDVALS[:,0], qR.reshape([N1,N1,nV]))

    qL_[:,2:5] += ρvL
    qR_[:,2:5] += ρvR

    vL_ = zeros([N1,3])
    vR_ = zeros([N1,3])
    ΣL_ = zeros([N1,3])
    ΣR_ = zeros([N1,3])
    for i in range(N1):
        PL_ = Cvec_to_Pclass(qL_[i], MPL)
        PR_ = Cvec_to_Pclass(qR_[i], MPR)
        vL_[i] = PL_.v
        vR_[i] = PR_.v
        ΣL_[i] = PL_.Σ()[0]
        ΣR_[i] = PR_.Σ()[0]

    ret[2*nX : 2*nX+3*N1]      = (vL_-vR_).ravel()
    ret[2*nX+3*N1 : 2*nX+6*N1] = (ΣL_-ΣR_).ravel()

    return ret

def dΧ(xh, dt):
    """ returns dΧ/dx and dΧ/dt at spatial node i and temporal node j
    """
    dxdΧ = dot(DERVALS, xh)
    dxdτ = dot(xh, DERVALS.T)
    dΧdx = 1 / dxdΧ
    dΧdt = -dxdτ / (dxdΧ * dt)
    return dΧdx, dΧdt

if __name__ == "__main__":

    MP = material_parameters(EOS='sg', ρ0=1, cv=1, γ=1.4, pINF=0, p0=1,
                              b0=1, α=1, μ=1e-2, Pr=0.75)
    """
    ρL = 1
    pL = 1
    vL = zeros(3)
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)
    MPL = MP

    ρR = 0.1
    pR = 0.1
    vR = zeros(3)
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)
    MPR = MP

    QL = Cvec(ρL, pL, vL, AL, JL, MPR)
    QR = Cvec(ρR, pR, vR, AR, JR, MPL)

    PL = Cvec_to_Pclass(QL, MPL)
    PR = Cvec_to_Pclass(QR, MPL)

    wL = array([QL for i in range(N1)])
    wR = array([QR for i in range(N1)])
    """

    wL = zeros([N1,nV])
    wR = zeros([N1,nV])
    v0 = zeros(3)
    J0 = zeros(3)
    for i in range(N1):
        ρL = 3-NODES[i]
        pL = ρL
        AL = ρL**(1/3) * eye(3)
        ρR = 2-NODES[i]
        pR = ρR
        AR = ρR**(1/3) * eye(3)
        wL[i] = Cvec(ρL, pL, v0, AL, J0, MP)
        wR[i] = Cvec(ρR, pR, v0, AR, J0, MP)

    WwL = dot(W, wL)
    WwR = dot(W, wR)
    qL0 = array([wL for i in range(N1)])
    qR0 = array([wR for i in range(N1)])

    nX = NT*nV
    x0 = zeros(2*nX+6*N1)
    x0[0 : nX]    = qL0.ravel()
    x0[nX : 2*nX] = qR0.ravel()

    dt = 0.01

    f = lambda x : obj(x, WwL, WwR, dt, MP, MP)
    #ret = newton_krylov(f, x0)
    #ret = root(f, x0)
    #ret = anderson(f, x0)
    ret = leastsq(f, x0)[0]

    qL  = ret[0 : nX].reshape([N1,N1,nV])
    qR  = ret[nX : 2*nX].reshape([N1,N1,nV])
    ρvL = ret[2*nX : 2*nX+3*N1].reshape([N1,3])
    ρvR = ret[2*nX+3*N1 : 2*nX+6*N1].reshape([N1,3])

    qL[abs(qL)<1e-12] = 0
    qR[abs(qR)<1e-12] = 0
