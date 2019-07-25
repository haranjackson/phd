from numpy import array, eye, zeros

from gpr.misc.objects import material_params
from gpr.misc.structures import Cvec
from tests.params.reactive import NM_JWL_SI


def steady_znd():

    n = 100
    MP = NM_JWL_SI

    L = 0.0062432
    ρ = 1137
    p = 23e9
    tf = 1e-6

    Q = Cvec(ρ, 0, zeros(3), MP, A=eye(3), λ=1)
    Qs = Cvec(ρ, p, zeros(3), MP, A=eye(3), λ=1)
    u = zeros([n, 15])
    for i in range(n):
        u[i] = Q
    u[0] = Qs

    return u, [MP], tf, [L/n], 'transitive'


def shock_detonation():

    tf = 0.5
    L = 1
    nx = 400

    MP = material_params('sg', ρ0=1, γ=1.4, cv=2.5, b0=1e-8, μ=1e-4,
                         Qc=1, Kc=250, Ti=0.25, REACTION='d')
    ρL = 1.4
    pL = 1
    vL = zeros(3)
    AL = ρL**(1/3) * eye(3)
    λL = 0

    ρR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    AR = ρR**(1/3) * eye(3)
    λR = 1

    QL = Cvec(ρL, pL, vL, MP, A=AL, λ=λL)
    QR = Cvec(ρR, pR, vR, MP, A=AR, λ=λR)

    u = zeros([nx, 15])

    for i in range(nx):
        if i < nx / 4:
            u[i] = QL
        else:
            u[i] = QR

    return u, [MP], tf, [L / nx], 'transitive'


def heating_deflagration():
    pass


def heating_deflagration_bc():
    pass
