from numpy import concatenate, dot, zeros

from models.gpr.systems.eigenvectors import eigen
from models.gpr.systems.primitive import source_prim
from models.gpr.misc.structures import State
from scipy.optimize import newton_krylov


def Cvec_to_Pvec(Q, MP):
    """ Returns the vector of primitive variables in standard ordering,
        given the vector of conserved variables.
    """
    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3, 3])
    J = Q[14:17] / ρ

    if MP.REACTIVE:
        λ = Q[17] / ρ
    else:
        λ = None

    p = pressure(ρ, E, v, A, J, MP, λ)

    ret = Q.copy()
    ret[1] = p
    ret[2:5] /= ρ
    ret[14:] /= ρ

    return ret


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

    PL = State(QL, MPL)
    PR = State(QR, MPR)
    PL_ = State(QL_, MPL)
    PR_ = State(QR_, MPR)

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

    PL = State(QL, MPL)
    PR = State(QR, MPR)

    x0 = zeros(42)
    x0[:21] = concatenate([QL, PL.Σ()[0], [PL.q()[0]]])
    x0[21:] = concatenate([QR, PR.Σ()[0], [PR.q()[0]]])

    def obj(x): return star_stepper_obj(x, QL, QR, dt, MPL, MPR)

    ret = newton_krylov(obj, x0).reshape([2, 21])

    return ret[0, :17], ret[1, :17]
