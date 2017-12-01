from numpy import outer, sqrt, zeros

from system.gpr.misc.functions import GdevG, eigvalsn
from system.gpr.misc.structures import Cvec_to_Pclass
from system.gpr.variables.mg import dTdρ, dTdp
from system.gpr.variables.wavespeeds import c_0, c_h
from options import VISCOUS, THERMAL, PERRON_FROB


def thermo_acoustic_tensor(P, d):
    """ Returns the tensor T_dij corresponding to the (i,j) component of the
        thermo-acoustic tensor in the dth direction
    """
    ret = zeros([4,4])

    ρ = P.ρ
    p = P.p
    T = P.T

    G = P.G()
    Gd = G[d]

    PAR = P.PAR
    cs2 = PAR.cs2

    if VISCOUS:
        O = GdevG(G)
        O[:, d] *= 2
        O[d] *= 2
        O[d, d] *= 3/4
        O += Gd[d] * G + 1/3 * outer(Gd, Gd)
        O *= cs2
        ret[:3, :3] = O

    c0 = c_0(ρ, p, PAR)
    ret[d, d] += c0**2

    if THERMAL:
        Tρ = dTdρ(ρ, p, PAR)
        Tp = dTdp(ρ, PAR)
        ch = c_h(ρ, T, PAR)

        ret[3, 0] = Tρ + Tp * c0**2
        ret[0, 3] = ch**2 / Tp
        ret[3, 3] = ch**2

    return ret

def max_abs_eigs(Q, d, PAR):
    """ Returns the maximum of the absolute values of the eigenvalues of the GPR system
    """
    P = Cvec_to_Pclass(Q, PAR)
    vd = P.v[d]
    O = thermo_acoustic_tensor(P, d)

    if PERRON_FROB:
        rowSum = [sum(o) for o in O]
        colSum = [sum(oT) for oT in O.T]
        lam = sqrt(min(max(rowSum),max(colSum)))
    else:
        lam = sqrt(eigvalsn(O, 4).max())

    if vd > 0:
        return vd + lam
    else:
        return lam - vd
