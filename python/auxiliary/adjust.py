from auxiliary.funcs import det3
from gpr.functions import primitive, conserved
from gpr.variables import E_1r, E_2J
from options import NOISE_LIM, reactiveEOS


def limit_noise(arr):
    """ Removes all elements of arr smaller in absolute value than the noise limit
    """
    arr[abs(arr) < NOISE_LIM] = 0
    return arr

def renormalise_density(u):
    for i in range(len(u)):
        ui = u[i,0,0]
        A = ui[5:14].reshape([3,3], order='F')
        ui[0] = det3(A)

def thermal_conversion(u, params):
    y = params.y; pINF = params.pINF; cv = params.cv; alpha2 = params.alpha2; Qc = params.Qc
    n = len(u)
    Etot = sum(u[:, 0, 0, 1])
    temp = 0
    for i in range(n):
        Q = u[i, 0, 0]
        P = primitive(Q, params, 0, 1, 1)
        temp += E_2J(P.J, alpha2) / P.T
        if reactiveEOS:
            temp += E_1r(P.c, Qc) / P.T

    p_t = ((y-1) * Etot - n * y * pINF) / (n + temp / cv)
    for i in range(n):
        Q = u[i, 0, 0]
        P = primitive(Q, params, 0, 1, 1)
        r = p_t / ((y-1) * cv * P.T)
        u[i, 0, 0] = conserved(r, p_t, P.v, P.A, P.J, P.c, params, 0, 1, 1)
