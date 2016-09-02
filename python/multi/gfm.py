from numpy import concatenate, int64, remainder, zeros
from scipy.linalg import det

from ader.basis import basis_polys
from auxiliary.bc import temperature_fix_density
from gpr.functions import conserved, primitive
from gpr.variables import density, entropy, temperature
from multi.approximate_riemann import star_states
from options import dx, L, N1, nx, UPDATE_STEP, RGFM, isoFix, entropyFix, tempFix


psi, _, _ = basis_polys()


def qhxt(qh, x, t):
    ret = zeros(18)
    qh = qh.reshape(N1,N1,18)
    for i in range(N1):
        for j in range(N1):
            ret += qh[i,j] * psi[i](t) * psi[j](x)
    return ret

def update_interface_locations(qh, interfaceLocations, dt):

    ddt = dt / UPDATE_STEP
    for i in range(len(interfaceLocations)):
        x0 = interfaceLocations[i]

        for j in range(UPDATE_STEP):
            qh0 = qh[int(x0*nx/L)].copy()
            x00 = remainder(x0, dx) / dx
            t00 = j*ddt
            u00 = qhxt(qh0, x00, t00)
            v = u00[2] / u00[0]
            x0 += ddt * v
        interfaceLocations[i] = x0 # + dt

    return interfaceLocations

def interface_indices(intLocs, n):
    ret = []
    for x0 in intLocs:
        for i in range(n):
            if (i+0.5)*dx >= x0:
                ret.append(i)
                break
    return concatenate([[0], ret, [n]]).astype(int64)

def entropy_fix(Q0, params0, params1, S, viscous, thermal, reactive):
    """ Changes density and distortion of state Q0 of a fluid with parameters given by params0,
        so that the cell it describes has entropy S for a fluid with parameters given by params1
    """
    P = primitive(Q0, params0, viscous, thermal, reactive)
    r = density(S, P.p, params1)
    if (P.A==0).all():
        A = P.A
    else:
        A = (r / (det(P.A) * params1.r0))**(1/3) * P.A
    T0 = P.T
    T1 = temperature(r, P.p, params1.y, params1.pINF, params1.cv)
    J1 = (T0 * params0.alpha2) / (T1 * params1.alpha2) * P.J
    return conserved(r, P.p, P.v, A, J1, P.c, params1, viscous, thermal, reactive)

def temperature_fix(Q0, params0, params1, T, viscous, thermal, reactive):
    """ Changes density and distortion of state Q0 of a fluid with parameters given by params0,
        so that the cell it describes has temperature T for a fluid with parameters given by params1
    """
    P = primitive(Q0, params0, viscous, thermal, reactive)
    r = temperature_fix_density(P.p, T, params1)
    if (P.A==0).all():
        A = P.A
    else:
        A = (r / det(P.A) * params1.r0)**(1/3) * P.A
    J1 = (P.T * params0.alpha2) / (T * params1.alpha2) * P.J
    return conserved(r, P.p, P.v, A, J1, P.c, params1, viscous, thermal, reactive)

def add_ghost_cells(fluids, inds, materialParameters, dt, viscous, thermal, reactive):

    for i in range(len(fluids)-1):
        uL = fluids[i]
        uR = fluids[i+1]
        ind = inds[i+1]
        paramsL = materialParameters[i]
        paramsR = materialParameters[i+1]

        if RGFM:
            QL = uL[ind-1-isoFix, 0, 0]
            QR = uR[ind+isoFix, 0, 0]
            QLstar, QRstar = star_states(QL, QR, paramsL, paramsR, dt, viscous, thermal, reactive)
            for j in range(ind, len(uL)):
                uL[j] = QLstar
            for j in range(ind):
                uR[j] = QRstar

        else:
            for j in range(ind, len(uL)):
                uL[j] = uR[j, 0, 0]
            for j in range(ind):
                uR[j] = uL[j, 0, 0]

        if entropyFix:
            SL = entropy(uL[ind-isoFix-1, 0, 0], paramsL, viscous, thermal, reactive)
            SR = entropy(uR[ind+isoFix, 0, 0], paramsR, viscous, thermal, reactive)

            if isoFix:
                uL[ind-1] = entropy_fix(uL[ind-1,0,0], paramsL, paramsL, SL,
                                        viscous, thermal, reactive)
                uR[ind] = entropy_fix(uR[ind,0,0], paramsR, paramsR, SR,
                                      viscous, thermal, reactive)

            if RGFM:
                for j in range(ind, len(uL)):
                    uL[j] = entropy_fix(uL[j,0,0], paramsL, paramsL, SL, viscous, thermal, reactive)
                for j in range(ind):
                    uR[j] = entropy_fix(uR[j,0,0], paramsR, paramsR, SR, viscous, thermal, reactive)
            else:
                for j in range(ind, len(uL)):
                    uL[j] = entropy_fix(uL[j,0,0], paramsR, paramsL, SL, viscous, thermal, reactive)
                for j in range(ind):
                    uR[j] = entropy_fix(uR[j,0,0], paramsL, paramsR, SR, viscous, thermal, reactive)

        elif tempFix:
            TL = primitive(uL[ind-1, 0, 0], paramsL, viscous, thermal, reactive).T
            TR = primitive(uR[ind, 0, 0], paramsR, viscous, thermal, reactive).T
            T = (TL+TR)/2

            if isoFix:
                uL[ind-1] = temperature_fix(uL[ind-1,0,0], paramsL, paramsL, T,
                                            viscous, thermal, reactive)
                uR[ind] = temperature_fix(uR[ind,0,0], paramsR, paramsR, T,
                                          viscous, thermal, reactive)

            if RGFM:
                for j in range(ind, len(uL)):
                    uL[j] = temperature_fix(uL[j, 0, 0], paramsL, paramsL, T,
                                            viscous, thermal, reactive)
                for j in range(ind):
                    uR[j] = temperature_fix(uR[j, 0, 0], paramsR, paramsR, T,
                                            viscous, thermal, reactive)
            else:
                for j in range(ind, len(uL)):
                    uL[j] = temperature_fix(uL[j, 0, 0], paramsR, paramsL, T,
                                            viscous, thermal, reactive)
                for j in range(ind):
                    uR[j] = temperature_fix(uR[j, 0, 0], paramsL, paramsR, T,
                                            viscous, thermal, reactive)
