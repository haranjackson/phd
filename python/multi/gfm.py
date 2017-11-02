from numpy import concatenate, int64

from multi.riemann import star_states
from options import dx, ISO_FIX


def interface_inds(intLocs, n):
    ret = []
    for x0 in intLocs:
        for i in range(n):
            if (i+0.5) * dx >= x0:
                ret.append(i)
                break
    return concatenate([[0], ret, [n]]).astype(int64)

def add_ghost_cells(fluids, inds, vels, materialParameters, dt):

    for i in range(len(fluids)-1):
        uL = fluids[i]
        uR = fluids[i+1]
        ind = inds[i+1]
        PARL = materialParameters[i]
        PARR = materialParameters[i+1]

        QL = uL[ind-1-ISO_FIX, 0, 0]
        QR = uR[ind+ISO_FIX, 0, 0]
        QL_, QR_ = star_states(QL, QR, dt, PARL, PARR)

        for j in range(ind, len(uL)):
            uL[j] = QL_
        for j in range(ind):
            uR[j] = QR_

        vels[i] = (QL_[2] / QL_[0] + QR_[2] / QR_[0]) / 2   # average v*
