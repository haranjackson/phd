from numpy import eye, zeros

from gpr.misc.structures import Cvec
from test.params.reactive import NM_JWL_Scaled


def steady_znd():

    n = 100
    MP = NM_JWL_Scaled

    Q = Cvec(1, 0, zeros(3), MP, A=eye(3), λ=1)
    Qs = Cvec(1, 0.52, zeros(3), MP, A=eye(3), λ=1)
    u = zeros([n, 15])
    for i in range(n):
        u[i] = Q
    u[0] = Qs

    return u, [MP], 0.1, [1/n], 'transitive'


def shock_detonation():
    """ tf = 0.5
        L = 1
        reactionType = 'd'
    """
    MP = material_parameters('sg', ρ0=1, y=1.4, cv=2.5,
                             b0=1e-8, cα=1e-8, μ=1e-4, Pr=0.75,
                             Qc=1, Kc=250, Ti=0.25)

    rL = 1.4
    pL = 1
    vL = zeros(3)
    AL = rL**(1/3) * eye(3)
    cL = 0

    rR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    AR = rR**(1/3) * eye(3)
    cR = 1

    J = zeros(3)

    QL = conserved(rL, pL, vL, AL, J, cL, params, 1, 1, 1)
    QR = conserved(rR, pR, vR, AR, J, cR, params, 1, 1, 1)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        if i*dx < L/4:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [MP], []


def heating_deflagration():
    pass


def heating_deflagration_bc():
    pass
