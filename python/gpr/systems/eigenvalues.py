from numpy import outer, sqrt, zeros
from numpy.linalg import eigvals

from gpr.misc.functions import GdevG
from gpr.misc.structures import Cvec_to_Pclass
from gpr.variables.mg import dTdρ, dTdp
from gpr.variables.wavespeeds import c_0, c_h, c_s2

from options import VISCOUS, THERMAL, PERRON_FROB


def thermo_acoustic_tensor(P, d):
    """ Returns the tensor T_dij corresponding to the (i,j) component of the
        thermo-acoustic tensor in the dth direction
    """
    ret = zeros([4,4])

    ρ = P.ρ
    p = P.p()
    T = P.T()

    G = P.G()
    Gd = G[d]

    MP = P.MP
    cs2 = c_s2(ρ, MP)

    if VISCOUS:
        O = GdevG(G)
        O[:, d] *= 2
        O[d] *= 2
        O[d, d] *= 3/4
        O += Gd[d] * G + 1/3 * outer(Gd, Gd)
        O *= cs2
        ret[:3, :3] = O

    c0 = c_0(ρ, p, MP)
    ret[d, d] += c0**2

    if THERMAL:
        Tρ = dTdρ(ρ, p, MP)
        Tp = dTdp(ρ, MP)
        ch = c_h(ρ, T, MP)

        ret[3, 0] = Tρ + Tp * c0**2
        ret[0, 3] = ch**2 / Tp
        ret[3, 3] = ch**2

    return ret

def max_abs_eigs(Q, d, MP):
    """ Returns the maximum of the absolute values of the eigenvalues of the GPR system
    """
    P = Cvec_to_Pclass(Q, MP)
    vd = P.v[d]
    O = thermo_acoustic_tensor(P, d)

    if PERRON_FROB:
        rowSum = [sum(o) for o in O]
        colSum = [sum(oT) for oT in O.T]
        lam = sqrt(min(max(rowSum),max(colSum)))
    else:
        lam = sqrt(eigvals(O).max())

    if vd > 0:
        return vd + lam
    else:
        return lam - vd
