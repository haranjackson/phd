from numpy import concatenate, int64, remainder, zeros
from scipy.linalg import det

from solvers.basis import basis_polys
from auxiliary.boundaries import temperature_fix_density
from gpr.variables.state import density, entropy, temperature
from gpr.variables.vectors import Qvec, Qvec_to_Pclass
from multi.approximate_riemann import star_states
from options import dx, Lx, N1, nx, UPDATE_STEP, RGFM, isoFix

import options


psi, _, _ = basis_polys()


def qhxt(qh, x, t):
    ret = zeros(18)
    qh = qh.reshape(N1,N1,18)
    for i in range(N1):
        for j in range(N1):
            ret += qh[i,j] * psi[i](t) * psi[j](x)
    return ret

def update_interface_locs(qh, interfaceLocations, dt):

    ddt = dt / UPDATE_STEP
    for i in range(len(interfaceLocations)):
        x0 = interfaceLocations[i]

        for j in range(UPDATE_STEP):
            qh0 = qh[int(x0*nx/Lx)].copy()
            x00 = remainder(x0, dx) / dx
            t00 = j*ddt
            u00 = qhxt(qh0, x00, t00)
            v = u00[2] / u00[0]
            x0 += ddt * v
        interfaceLocations[i] = x0 # + dt

    return interfaceLocations

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
    P = Qvec_to_Pclass(Q0, PAR0)
    ρ = density(S, P.p, PAR1)
    if (P.A==0).all():
        A = P.A
    else:
        A = (ρ / (det(P.A) * PAR1.ρ0))**(1/3) * P.A
    T0 = P.T
    T1 = temperature(ρ, P.p, PAR1.γ, PAR1.pINF, PAR1.cv)
    J1 = (T0 * PAR0.α2) / (T1 * PAR1.α2) * P.J
    return Qvec(ρ, P.p, P.v, A, J1, P.λ, PAR1)

def temperature_fix(Q0, PAR0, PAR1, T):
    """ Changes density and distortion of state Q0 of a fluid with parameters
        given by PAR0, so that the cell it describes has temperature T for a
        fluid with parameters given by PAR1
    """
    P = Qvec_to_Pclass(Q0, PAR0)
    ρ = temperature_fix_density(P.p, T, PAR1)
    if (P.A==0).all():
        A = P.A
    else:
        A = (ρ / det(P.A) * PAR1.ρ0)**(1/3) * P.A
    J1 = (P.T * PAR0.α2) / (T * PAR1.α2) * P.J
    return Qvec(ρ, P.p, P.v, A, J1, P.λ, PAR1)

def add_ghost_cells(fluids, inds, materialParameters, dt, SFix=None, TFix=None):

    if SFix is None:
        SFix = options.SFix
    if TFix is None:
        TFix = options.TFix

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

        elif TFix:
            TL = Qvec_to_Pclass(uL[ind-1, 0, 0], PARL).T
            TR = Qvec_to_Pclass(uR[ind, 0, 0], PARR).T
            T = (TL+TR)/2

            if isoFix:
                uL[ind-1] = temperature_fix(uL[ind-1,0,0], PARL, PARL, T)
                uR[ind]  =  temperature_fix(uR[ind,0,0],   PARR, PARR, T)

            if RGFM:
                for j in range(ind, len(uL)):
                    uL[j] = temperature_fix(uL[j, 0, 0], PARL, PARL, T)
                for j in range(ind):
                    uR[j] = temperature_fix(uR[j, 0, 0], PARR, PARR, T)
            else:
                for j in range(ind, len(uL)):
                    uL[j] = temperature_fix(uL[j, 0, 0], PARR, PARL, T)
                for j in range(ind):
                    uR[j] = temperature_fix(uR[j, 0, 0], PARL, PARR, T)
