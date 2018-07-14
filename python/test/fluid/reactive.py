def steady_znd():
    pass


def shock_detonation():
    """ tf = 0.5
        L = 1
        reactionType = 'd'
    """
    MP = material_parameters(y=1.4, pINF=0, cv=2.5, r0=1, p0=1,
                             cs=1e-8, cα=1e-8, μ=1e-4, Pr=0.75,
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
