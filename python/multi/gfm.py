from numpy import concatenate, int64
from scipy.linalg import det

from solvers.basis import basis_polys
from gpr.variables.state import density, entropy, temperature
from gpr.variables.vectors import Cvec, Cvec_to_Pclass
from multi.approximate_riemann import star_states
from options import dx, RGFM, isoFix, SFix


psi, _, _ = basis_polys()


def interface_inds(intLocs, n):
    ret = []
    for x0 in intLocs:
        for i in range(n):
            if (i+0.5)*dx >= x0:
                ret.append(i)
                break
    return concatenate([[0], ret, [n]]).astype(int64)

def entropy_fix(Q0, PAR0, PAR1, S):
    """ Changes density and distortion of state Q0 of a fluid with parameters
        given by PAR0, so that the cell it describes has entropy S for a fluid
        with parameters given by PAR1
    """
    P = Cvec_to_Pclass(Q0, PAR0)
    ρ = density(S, P.p, PAR1)
    if (P.A==0).all():
        A = P.A
    else:
        A = (ρ / (det(P.A) * PAR1.ρ0))**(1/3) * P.A
    T0 = P.T
    T1 = temperature(ρ, P.p, PAR1.γ, PAR1.pINF, PAR1.cv)
    J1 = (T0 * PAR0.α2) / (T1 * PAR1.α2) * P.J
    return Cvec(ρ, P.p, P.v, A, J1, P.λ, PAR1)

def add_ghost_cells(fluids, inds, vels, materialParameters, dt):

    for i in range(len(fluids)-1):
        uL = fluids[i]
        uR = fluids[i+1]
        ind = inds[i+1]
        PARL = materialParameters[i]
        PARR = materialParameters[i+1]

        if RGFM:
            QL = uL[ind-1-isoFix, 0, 0]
            QR = uR[ind+isoFix, 0, 0]
            QL_, QR_ = star_states(QL, QR, dt, PARL, PARR)

            for j in range(ind, len(uL)):
                uL[j] = QL_
            for j in range(ind):
                uR[j] = QR_

            vels[i] = (QL_[2] / QL_[0] + QR_[2] / QR_[0]) / 2   # average v*

        else:
            for j in range(ind, len(uL)):
                uL[j] = uR[j, 0, 0]
            for j in range(ind):
                uR[j] = uL[j, 0, 0]

        if SFix:
            SL = entropy(uL[ind-isoFix-1, 0, 0], PARL)
            SR = entropy(uR[ind+isoFix, 0, 0], PARR)

            if isoFix:
                uL[ind-1] = entropy_fix(uL[ind-1,0,0], PARL, PARL, SL)
                uR[ind]  =  entropy_fix(uR[ind,0,0],   PARR, PARR, SR)

            if RGFM:
                for j in range(ind, len(uL)):
                    uL[j] = entropy_fix(uL[j,0,0], PARL, PARL, SL)
                for j in range(ind):
                    uR[j] = entropy_fix(uR[j,0,0], PARR, PARR, SR)
            else:
                for j in range(ind, len(uL)):
                    uL[j] = entropy_fix(uL[j,0,0], PARR, PARL, SL)
                for j in range(ind):
                    uR[j] = entropy_fix(uR[j,0,0], PARL, PARR, SR)
