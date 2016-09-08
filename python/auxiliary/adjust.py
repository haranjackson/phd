from auxiliary.funcs import det3
from gpr.functions import primitive, conserved
from gpr.variables import E_1r, E_2J
from options import NOISE_LIM, reactiveEOS


def limit_noise(arr):
    """ Removes all elements of arr smaller in absolute value than the noise limit
    """
    arr[abs(arr) < NOISE_LIM] = 0
    return arr

def renormalise_density(fluids, materialParameters):
    """ Sets density to ρ0 * det(A)
    """
    for i in range(len(fluids)):
        fluid = fluids[i]
        params = materialParameters[i]
        for j in range(len(fluid)):
            uj = fluid[j,0,0]
            A = uj[5:14].reshape([3,3], order='F')
            uj[0] = det3(A) / params.ρ0

def thermal_conversion(fluids, materialParameters):
    for i in range(len(fluids)):
        fluid = fluids[i]
        params = materialParameters[i]
        γ = params.γ; pINF = params.pINF; cv = params.cv; α2 = params.α2; Qc = params.Qc
        n = len(fluid)
        Etot = sum(fluid[:, 0, 0, 1])
        temp = 0
        for j in range(n):
            Q = fluid[j, 0, 0]
            P = primitive(Q, params, 0, 1, 1)
            temp += E_2J(P.J, α2) / P.T
            if reactiveEOS:
                temp += E_1r(P.c, Qc) / P.T

        p_t = ((γ-1) * Etot - n * γ * pINF) / (n + temp / cv)
        for j in range(n):
            Q = fluid[j, 0, 0]
            P = primitive(Q, params, 0, 1, 1)
            ρ = p_t / ((γ-1) * cv * P.T)
            fluid[j, 0, 0] = conserved(ρ, p_t, P.v, P.A, P.J, P.λ, params, 0, 1, 1)
