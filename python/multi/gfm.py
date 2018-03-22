from numpy import sum

from models.gpr.multi.riemann import star_states

from options import ISO_FIX, LSET


def get_levelset_root(u, i, m):
    """ return the location of interface i
    """
    φ = u[:, :, :, i - (m - 1)]
    n = len(φ)
    for j in range(n):
        if φ[j] >= 0:
            return j
    return n


def add_ghost_cells(mats, dt, *args):

    m = len(args[0])
    for i in range(m - 1):
        uL = mats[i]
        uR = mats[i + 1]
        ind = get_levelset_root(uL, i, m)
        MPL = MPs[i]
        MPR = MPs[i + 1]

        QL = uL[ind - 1 - ISO_FIX, 0, 0, :-(m - 1)]
        QR = uR[ind + ISO_FIX, 0, 0, :-(m - 1)]

        print(QL, QR)
        QL_, QR_ = star_states(QL, QR, dt, MPL, MPR)

        for j in range(ind, len(uL)):
            uL[j, 0, 0, :-(m - 1)] = QL_
        for j in range(ind):
            uR[j, 0, 0, :-(m - 1)] = QR_


def get_material_index(Q):
    NV = len(Q)
    return sum(Q[NV-LSET:] < 0)
