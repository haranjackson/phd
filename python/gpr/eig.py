from numpy import array, diag, dot, eye, outer, sqrt, zeros
from scipy.linalg import solve, eig

from auxiliary.funcs import GdevG, gram
from gpr.functions import primitive
from gpr.variables import sigma, sigma_A, c_h


from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
def eigvalsn(a, n):
    geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a,))
    lwork = _compute_lwork(geev_lwork, n, compute_vl=0, compute_vr=0)
    w, _, _, _, _ = geev(a, lwork=lwork, compute_vl=0, compute_vr=0, overwrite_a=1)
    return w


def thermo_acoustic_tensor(r, A, p, T, y, pINF, cs2, alpha2, d, viscous, thermal):
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

    ret[d, d] += y*p / r

    if thermal:
        ret[3, 0] = ((y-1) * p - pINF) * T / (r * (p+pINF))
        temp = (y-1) * alpha2 * T / r
        ret[0, 3] = temp
        ret[3, 3] = temp * T / (p+pINF)

    return ret

def max_abs_eigs(Q, d, params, mechanical, viscous, thermal, reactive):
    P = primitive(Q, params, viscous, thermal, reactive)

    if not mechanical:
        return c_h(P.r, P.T, params.alpha, params.cv)

    else:
        v = P.v[d]
        O = thermo_acoustic_tensor(P.r, P.A, P.p, P.T, params.y, params.pINF,
                                   params.cs2, params.alpha2, d, viscous, thermal)
        lam = sqrt(eigvalsn(O, 4).max())
        if v > 0:
            return v + lam
        else:
            return lam - v

def Xi1mat(r, p, T, pINF, sigd, dsdAd):
    X = zeros([4, 5])
    X[:3, 0] = -sigd / r**2
    X[0, 1] = 1 / r
    X[:3, 2:] = -1/r * dsdAd
    X[3, 0] = -T / r**2
    X[3, 1] = T / (r * (p + pINF))
    return X

def Xi2mat(r, p, A, T, y, alpha2):
    Y = zeros([5,4])
    Y[2:, :3] = A
    Y[0, 0] = r
    Y[1, 0] = y * p
    Y[1, 3] = (y-1) * alpha2 * T
    return Y

def primitive_eigs(q, params, viscous, thermal, reactive):
    """ Returns eigenvalues and set of left and right eigenvectors of the matrix returned by
        jacobian_primitive_reordered
    """
    P = primitive(q, params, viscous, thermal, reactive)
    r = P.r; p = P.p; A = P.A; T = P.T; vd = P.v[0]
    y = params.y; pINF = params.pINF; cs2 = params.cs2; alpha2 = params.alpha2

    L = zeros([18,18])
    R = zeros([18,18])
    sig0 = sigma(r, A, cs2)[0]
    dsdA = sigma_A(r, A, cs2)[0]
    Pi1 = dsdA[:,:,0]
    Pi2 = dsdA[:,:,1]
    Pi3 = dsdA[:,:,2]

    O = thermo_acoustic_tensor(r, A, p, T, y, pINF, cs2, alpha2, 0, viscous, thermal)
    Xi1 = Xi1mat(r, p, T, pINF, sig0, Pi1)
    Xi2 = Xi2mat(r, p, A, T, y, alpha2)
    w, vl, vr = eig(O, left=1)
    sw = sqrt(w.real)
    D = diag(sw)
    Q = vl.T
    Q_1 = vr
    I = dot(Q,Q_1)
    Q = solve(I, Q, overwrite_a=1, check_finite=0)
    DQ = dot(D,Q)

    temp = solve(DQ.T, Xi2.T, overwrite_a=1, overwrite_b=1, check_finite=0).T
    temp2 = Q_1
    R[:5, :4] = temp
    R[:5, 4:8] = temp
    R[11:15, :4] = temp2
    R[11:15, 4:8] = -temp2

    temp = solve(D, dot(Q, Xi1), overwrite_b=1, check_finite=0)
    temp2 = -1/r * solve(D, dot(Q[:,:3], Pi2), overwrite_b=1, check_finite=0)
    temp3 = -1/r * solve(D, dot(Q[:,:3], Pi3), overwrite_b=1, check_finite=0)
    temp4 = Q
    L[:4, :5] = temp
    L[4:8, :5] = temp
    L[:4, 5:8] = temp2
    L[4:8, 5:8] = temp2
    L[:4, 8:11] = temp3
    L[4:8, 8:11] = temp3
    L[:4, 11:15] = temp4
    L[4:8, 11:15] = -temp4

    b = array([p+pINF, 0, 0]) - sig0
    Pi1A = dot(Pi1, A)
    c = 2 / (solve(Pi1A, b, overwrite_a=1, check_finite=0)[0] - 1)
    R[0, 8] = c * r
    R[1, 8] = c * (p + pINF)
    R[2:5, 8] = c * solve(Pi1, b, overwrite_b=1, check_finite=0)

    temp = solve(A.T, array([1, 0, 0]), overwrite_b=1, check_finite=0)
    L[8, 0] = -1 / r
    L[8, 2:5] = temp
    L[8, 5:8] = dot(temp, solve(Pi1, Pi2, check_finite=0))
    L[8, 8:11] = dot(temp, solve(Pi1, Pi3, check_finite=0))

    R[2:5, 9:12] = -2 * solve(Pi1, Pi2)
    R[2:5, 12:15] = -2 * solve(Pi1, Pi3)
    R[5:11, 9:15] = 2 * eye(6)
    R[15:18, 15:18] = 2 * eye(3)

    L[9:15, 5:11] = eye(6)
    L[15:18, 15:18] = eye(3)

    nonDegenList = [vd+sw[0], vd+sw[1], vd+sw[2], vd+sw[3], vd-sw[0], vd-sw[1], vd-sw[2], vd-sw[3]]
    return array(nonDegenList + [vd]*10).real, L, 0.5 * R
