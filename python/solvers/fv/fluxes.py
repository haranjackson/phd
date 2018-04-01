from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS


def B_INT(obj, qL, qR, d):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(obj.NV)
    Δq = qR - qL
    for i in range(obj.N):
        q = qL + NODES[i] * Δq
        B = obj.nonconservative_matrix(q, d, obj.model_params)
        ret += WGHTS[i] * dot(B, Δq)
    return ret


def D_OSH(obj, qL, qR, d):
    """ Returns the Osher flux component, in the dth direction
    """
    ret = zeros(obj.NV, dtype=complex128)
    Δq = qR - qL

    for i in range(obj.N):
        q = qL + NODES[i] * Δq
        M = obj.system_matrix(q, d, obj.model_params)
        λ, R = eig(M, overwrite_a=1)
        b = solve(R, Δq)
        ret += WGHTS[i] * dot(R, abs(λ) * b)

    return ret.real


def D_ROE(obj, qL, qR, d):
    """ Returns the Roe flux component, in the dth direction
    """
    M = zeros([obj.NV, obj.NV])
    Δq = qR - qL

    for i in range(obj.N):
        q = qL + NODES[i] * Δq
        M += WGHTS[i] * obj.system_matrix(q, d, obj.model_params)

    λ, R = eig(M, overwrite_a=1)
    b = solve(R, Δq)
    return dot(R, abs(λ) * b).real


def D_RUS(obj, qL, qR, d):
    """ Returns the Rusanov flux component, in the dth direction
    """
    max1 = obj.max_eig(qL, d, obj.model_params)
    max2 = obj.max_eig(qR, d, obj.model_params)
    return max(max1, max2) * (qR - qL)
