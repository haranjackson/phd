from numpy import array, diag, dot, eye, outer, sqrt, zeros
from scipy.linalg import solve, eig
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork

from auxiliary.funcs import GdevG, gram
from gpr.matrices.jacobians import dQdP, dPdQ, jacobian_variables
from gpr.variables.state import sigma, sigma_A, temperature
from gpr.variables.vectors import primitive
from gpr.variables.wavespeeds import c_h


def eigvalsn(a, n):
    geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a,))
    lwork = _compute_lwork(geev_lwork, n, compute_vl=0, compute_vr=0)
    w, _, _, _, _ = geev(a, lwork=lwork, compute_vl=0, compute_vr=0, overwrite_a=1)
    return w


def thermo_acoustic_tensor(ρ, A, p, T, γ, pINF, cs2, α2, d, viscous, thermal):
    """ Returns the tensor T_dij corresponding to the (i,j) component of the thermo-acoustic tensor
        in the dth direction
    """
    G = gram(A)
    Gd = G[d]
    ret = zeros([4,4])

    if viscous:
        O = GdevG(G)
        O[:, d] *= 2
        O[d] *= 2
        O[d, d] *= 3/4
        O += Gd[d] * G + 1/3 * outer(Gd, Gd)
        O *= cs2
        ret[:3, :3] = O

    ret[d, d] += γ * p / ρ

    if thermal:
        ret[3, 0] = ((γ-1) * p - pINF) * T / (ρ * (p+pINF))
        temp = (γ-1) * α2 * T / ρ
        ret[0, 3] = temp
        ret[3, 3] = temp * T / (p+pINF)

    return ret

def max_abs_eigs(Q, d, params, subsystems):
    """ Returns the maximum of the absolute values of the eigenvalues of the GPR system
    """
    P = primitive(Q, params, subsystems)

    if not subsystems.mechanical:
        return c_h(P.ρ, P.T, params.α, params.cv)

    else:
        vd = P.v[d]
        O = thermo_acoustic_tensor(P.ρ, P.A, P.p, P.T, params.γ, params.pINF, params.cs2, params.α2,
                                   d, subsystems.viscous, subsystems.thermal)
        lam = sqrt(eigvalsn(O, 4).max())
        if vd > 0:
            return vd + lam
        else:
            return lam - vd

def max_abs_eigs_prim(P, d, γ, pINF, cv, cs2, α2, viscous, thermal):
    ρ = P[0]
    p = P[1]
    vd = P[2+d]
    A = P[5:14]
    T = temperature(ρ, p, γ, pINF, cv)

    O = thermo_acoustic_tensor(ρ, A, p, T, γ, pINF, cs2, α2, d, viscous, thermal)
    lam = sqrt(eigvalsn(O, 4).max())
    if vd > 0:
        return vd + lam
    else:
        return lam - vd

def Xi1mat(ρ, p, T, pINF, σd, dσdAd):
    ret = zeros([4, 5])
    ret[:3, 0] = -σd / ρ**2
    ret[0, 1] = 1 / ρ
    ret[:3, 2:] = -dσdAd / ρ
    ret[3, 0] = -T / ρ**2
    ret[3, 1] = T / (ρ * (p + pINF))
    return ret

def Xi2mat(ρ, p, A, T, γ, α2):
    ret = zeros([5,4])
    ret[2:, :3] = A
    ret[0, 0] = ρ
    ret[1, 0] = γ * p
    ret[1, 3] = (γ-1) * α2 * T
    return ret

def primitive_eigs(q, params, subsystems):
    """ Returns eigenvalues and set of left and right eigenvectors of the matrix returned by
        system_primitive_reordered
    """
    P = primitive(q, params, subsystems)
    ρ = P.ρ; p = P.p; A = P.A; T = P.T; vd = P.v[0]
    γ = params.γ; pINF = params.pINF; cs2 = params.cs2; α2 = params.α2

    L = zeros([18,18])
    R = zeros([18,18])
    σ0 = sigma(ρ, A, cs2)[0]
    dσdA = sigma_A(ρ, A, cs2)[0]
    Π1 = dσdA[:,:,0]
    Π2 = dσdA[:,:,1]
    Π3 = dσdA[:,:,2]

    O = thermo_acoustic_tensor(ρ, A, p, T, γ, pINF, cs2, α2, 0, subsystems.viscous,
                               subsystems.thermal)
    Ξ1 = Xi1mat(ρ, p, T, pINF, σ0, Π1)
    Ξ2 = Xi2mat(ρ, p, A, T, γ, α2)
    w, vl, vr = eig(O, left=1)
    sw = sqrt(w.real)
    D = diag(sw)
    Q = vl.T
    Q_1 = vr
    I = dot(Q,Q_1)
    Q = solve(I, Q, overwrite_a=1, check_finite=0)
    DQ = dot(D,Q)

    temp = solve(DQ.T, Ξ2.T, overwrite_a=1, overwrite_b=1, check_finite=0).T
    temp2 = Q_1
    R[:5, :4] = temp
    R[:5, 4:8] = temp
    R[11:15, :4] = temp2
    R[11:15, 4:8] = -temp2

    temp = solve(D, dot(Q, Ξ1), overwrite_b=1, check_finite=0)
    temp2 = -solve(D, dot(Q[:,:3], Π2), overwrite_b=1, check_finite=0) / ρ
    temp3 = -solve(D, dot(Q[:,:3], Π3), overwrite_b=1, check_finite=0) / ρ
    temp4 = Q
    L[:4, :5] = temp
    L[4:8, :5] = temp
    L[:4, 5:8] = temp2
    L[4:8, 5:8] = temp2
    L[:4, 8:11] = temp3
    L[4:8, 8:11] = temp3
    L[:4, 11:15] = temp4
    L[4:8, 11:15] = -temp4

    b = array([p+pINF, 0, 0]) - σ0
    Π1A = dot(Π1, A)
    c = 2 / (solve(Π1A, b, overwrite_a=1, check_finite=0)[0] - 1)
    R[0, 8] = c * ρ
    R[1, 8] = c * (p + pINF)
    R[2:5, 8] = c * solve(Π1, b, overwrite_b=1, check_finite=0)

    temp = solve(A.T, array([1, 0, 0]), overwrite_b=1, check_finite=0)
    L[8, 0] = -1 / ρ
    L[8, 2:5] = temp
    L[8, 5:8] = dot(temp, solve(Π1, Π2, check_finite=0))
    L[8, 8:11] = dot(temp, solve(Π1, Π3, check_finite=0))

    R[2:5, 9:12] = -2 * solve(Π1, Π2)
    R[2:5, 12:15] = -2 * solve(Π1, Π3)
    R[5:11, 9:15] = 2 * eye(6)
    R[15:18, 15:18] = 2 * eye(3)

    L[9:15, 5:11] = eye(6)
    L[15:18, 15:18] = eye(3)

    nonDegenList = [vd+sw[0], vd+sw[1], vd+sw[2], vd+sw[3], vd-sw[0], vd-sw[1], vd-sw[2], vd-sw[3]]
    return array(nonDegenList + [vd]*10).real, L, 0.5 * R

def conserved_eigs(q, params, subsystems):
    """ Returns the eigenvalues and left and right eigenvectors of the conserved system.
        NOTE: This doesn't currently appear to be implemented properly. It is taking the reordered
              eigenvectors of the primitive system and transforming them into conserved eigenvectors
              without attempting to put them in the standard ordering.
    """
    Λ, L, R = primitive_eigs(q, params, subsystems)
    P = primitive(q, params, subsystems)
    jacVars = jacobian_variables(P, params)
    DPDQ = dPdQ(P, params, jacVars, subsystems)
    DQDP = dQdP(P, params, jacVars, subsystems)
    return Λ, dot(L, DPDQ), dot(DQDP, R)
