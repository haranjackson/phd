def weno_stepper(pool, fluid, fluidBC, dt, PAR, SYS):

    t0 = time()

    wh = weno(fluidBC)
    t1 = time()

    nx,ny,nz = wh.shape[:3]
    qh = repeat(wh[:,:,:,newaxis], N1, 3).reshape([nx, ny, nz, NT, 18])
    t2 = time()

    fluid += fv_launcher(pool, qh, dt, PAR, SYS)
    t2 = time()

    print('WENO:', t1-t0, '\nFV:  ', t2-t1)
    return qh

def split_slic_stepper(fluid, dt, PAR, SYS):
    if linODE:
        ode_stepper(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    fluidn = standard_BC(standard_BC(fluid))
    slic_stepper(fluid, fluidn, dt, PAR, SYS)
    if linODE:
        ode_stepper(fluid, dt/2, PAR, SYS)
    else:
        ode_stepper_full(fluid, dt/2, PAR, SYS)
    return None

def new_stepper(fluid, fluidBC, dt, PAR, SYS):

    t0 = time()

    nx = len(fluidBC)
    for i in range(nx):
        P = primitive(fluidBC[i,0,0], PAR, SYS)
        fluidBC[i,0,0] = primitive_vector(P)
    wh = weno(fluidBC)
    t1 = time()

    qh = new_predictor(wh, dt, PAR, SYS)
    t2 = time()

    fluid += fv_terms(qh, dt, PAR, SYS)
    t3 = time()

    print('WENO:', t1-t0, '\nDG:  ', t2-t1, '\nFV:  ', t3-t2)

    return qh


def solver_distortion_lin(ρ, A, dt, PAR):
    """ A linearised solver for the distortion ODE
    """
    diff = tr(A)/3 * eye(3)
    ret1 = 0.5 * (A - A.T) + diff
    ret2 = 0.5 * (A + A.T) - diff
    return ret1 + exp(-6*dt/PAR.τ1 * (ρ/PAR.ρ0)**(7/3)) * ret2

def solver_thermal_analytic_constp(ρ, Q, dt, PAR, SYS):
    """ Returns the analytic solution to the thermal impulse ODE, assuming pressure is constant
        over the timescale of the ODE.
        NB This may not be a good assumption.
    """
    P0 = primitive(Q, PAR, SYS)
    return ρ * exp(-(P0.T * PAR.ρ0 * dt)/(PAR.T0 * ρ * PAR.τ2)) * P0.J