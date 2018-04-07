from numpy import array, dot, zeros
from scipy.optimize import newton_krylov, leastsq, root, anderson

from solvers.basis import end_values, derivative_values, quad
from solvers.dg.dg import rhs
from solvers.dg.matrices import system_matrices
from gpr.misc.objects import material_parameters
from gpr.misc.structures import State, Cvec


W, U, _, _, _ = system_matrices()
ENDVALS = end_values()
DERVALS = derivative_values()
NODES, _, _ = quad()


# The left and right end polynomials used in the implicit interfaces method
PSIL = lagrange(concatenate((NODES, [1])), [0] * N + [1])
PSIR = lagrange(concatenate(([0], NODES)), [1] + [0] * N)


def obj_eul(x, WwL, WwR, dt, MPL, MPR):

    nX = NT * NV

    qL = x[0: nX].reshape([NT, NV])
    qR = x[nX: 2 * nX].reshape([NT, NV])
    ρvL = x[2 * nX: 2 * nX + 3 * N].reshape([N, 3])
    ρvR = x[2 * nX + 3 * N: 2 * nX + 6 * N].reshape([N, 3])

    ret = zeros(2 * nX + 6 * N)

    ret[0: nX] = (dot(U, qL) - rhs(qL, WwL, dt, MPL, 0)).ravel()
    ret[nX: 2 * nX] = (dot(U, qR) - rhs(qR, WwR, dt, MPR, 0)).ravel()

    qL_ = dot(ENDVALS[:, 1], qL.reshape([N, N, NV]))
    qR_ = dot(ENDVALS[:, 0], qR.reshape([N, N, NV]))

    qL_[:, 2:5] += ρvL
    qR_[:, 2:5] += ρvR

    vL_ = zeros([N, 3])
    vR_ = zeros([N, 3])
    ΣL_ = zeros([N, 3])
    ΣR_ = zeros([N, 3])
    for i in range(N):
        PL_ = State(qL_[i], MPL)
        PR_ = State(qR_[i], MPR)
        vL_[i] = PL_.v
        vR_[i] = PR_.v
        ΣL_[i] = PL_.Σ()[0]
        ΣR_[i] = PR_.Σ()[0]

    ret[2 * nX: 2 * nX + 3 * N] = (vL_ - vR_).ravel()
    ret[2 * nX + 3 * N: 2 * nX + 6 * N] = (ΣL_ - ΣR_).ravel()

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
                             b0=1, cα=1, μ=1e-2, Pr=0.75)
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

    PL = State(QL, MPL)
    PR = State(QR, MPL)

    wL = array([QL for i in range(N)])
    wR = array([QR for i in range(N)])
    """

    wL = zeros([N, NV])
    wR = zeros([N, NV])
    v0 = zeros(3)
    J0 = zeros(3)
    for i in range(N):
        ρL = 3 - NODES[i]
        pL = ρL
        AL = ρL**(1 / 3) * eye(3)
        ρR = 2 - NODES[i]
        pR = ρR
        AR = ρR**(1 / 3) * eye(3)
        wL[i] = Cvec(ρL, pL, v0, AL, J0, MP)
        wR[i] = Cvec(ρR, pR, v0, AR, J0, MP)

    WwL = dot(W, wL)
    WwR = dot(W, wR)
    qL0 = array([wL for i in range(N)])
    qR0 = array([wR for i in range(N)])

    nX = NT * NV
    x0 = zeros(2 * nX + 6 * N)
    x0[0: nX] = qL0.ravel()
    x0[nX: 2 * nX] = qR0.ravel()

    dt = 0.01

    def f(x): return obj(x, WwL, WwR, dt, MP, MP)
    #ret = newton_krylov(f, x0)
    #ret = root(f, x0)
    #ret = anderson(f, x0)
    ret = leastsq(f, x0)[0]

    qL = ret[0: nX].reshape([N, N, NV])
    qR = ret[nX: 2 * nX].reshape([N, N, NV])
    ρvL = ret[2 * nX: 2 * nX + 3 * N].reshape([N, 3])
    ρvR = ret[2 * nX + 3 * N: 2 * nX + 6 * N].reshape([N, 3])

    qL[abs(qL) < 1e-12] = 0
    qR[abs(qR) < 1e-12] = 0
