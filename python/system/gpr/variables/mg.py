from numpy import exp


STIFFENED_GAS = 0
SHOCK_MG = 1
JWL = 2
COCHRAN_CHAN = 3


def eos_text_to_code(text):
    if text == 'sg':
        return STIFFENED_GAS
    if text == 'smg':
        return SHOCK_MG
    if text == 'jwl':
        return JWL
    if text == 'cc':
        return COCHRAN_CHAN


def Γ_MG(ρ, PAR):
    """ Returns the Mie-Gruneisen parameter
    """
    EOS = PAR.EOS

    if EOS == STIFFENED_GAS:
        γ = PAR.γ
        return γ - 1

    elif EOS == SHOCK_MG:
        Γ0 = PAR.Γ0
        ρ0 = PAR.ρ0
        return Γ0 * ρ0 / ρ

    elif EOS == JWL or EOS == COCHRAN_CHAN:
        return PAR.Γ0

def p_ref(ρ, PAR):
    """ Returns the reference pressure in the Mie-Gruneisen EOS
    """
    EOS = PAR.EOS

    if EOS == STIFFENED_GAS:
        return - PAR.γ * PAR.pINF

    elif EOS == SHOCK_MG:
        c02 = PAR.c02
        ρ0 = PAR.ρ0
        s = PAR.s
        if ρ > ρ0:
            return c02 * (1/ρ0 - 1/ρ) / (1/ρ0 - s * (1/ρ0 - 1/ρ))**2
        else:
            return c02 * (ρ-ρ0)

    else:
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v_ = ρ0 / ρ

        if EOS == JWL:
            return A * exp(-R1 * v_) + B * exp(-R2 * v_)

        elif EOS == COCHRAN_CHAN:
            return A * v_**(-R1) - B * v_**(-R2)

def e_ref(ρ, PAR):
    """ Returns the reference energy for the Mie-Gruneisen EOS
    """
    EOS = PAR.EOS

    if EOS == STIFFENED_GAS:
        return 0

    elif EOS == SHOCK_MG:
        ρ0 = PAR.ρ0
        pr = p_ref(ρ, PAR)
        if ρ > ρ0:
            return 0.5 * pr * (1/ρ0 - 1/ρ)
        else:
            return 0

    else:
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v_ = ρ0 / ρ

        if EOS == JWL:
            return A/(ρ0*R1) * exp(-R1*v_)  +  B/(ρ0*R2) * exp(-R2*v_)

        elif EOS == COCHRAN_CHAN:
            return -A/(ρ0*(1-R1)) * (v_**(1-R1)-1) + B/(ρ0*(1-R2)) * (v_**(1-R2)-1)


def dΓ_MG(ρ, PAR):
    """ Returns the derivative of the Mie-Gruneisen parameter
    """
    EOS = PAR.EOS

    if EOS == STIFFENED_GAS:
        return 0

    elif EOS == SHOCK_MG:
        Γ0 = PAR.Γ0
        ρ0 = PAR.ρ0
        return - Γ0 * ρ0 / ρ**2

    elif EOS == JWL or EOS == COCHRAN_CHAN:
        return 0

def dp_ref(ρ, PAR):
    """ Returns the derivative of the reference pressure in the Mie-Gruneisen EOS
    """
    EOS = PAR.EOS

    if EOS == STIFFENED_GAS:
        return 0

    elif EOS == SHOCK_MG:
        c02 = PAR.c02
        ρ0 = PAR.ρ0
        s = PAR.s
        if ρ > ρ0:
            return c02 * ρ0**2 * (s*(ρ0-ρ) - ρ) / (s*(ρ-ρ0) - ρ)**3
        else:
            return c02

    else:
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v_ = ρ0 / ρ

        if EOS == JWL:
            return v_ / ρ * (A * R1 * exp(-R1 * v_) + B * R2 * exp(-R2 * v_))

        elif EOS == COCHRAN_CHAN:
            return v_/ρ * (A*R1 * v_**(-R1-1) - B*R2 * v_**(-R2-1))

def de_ref(ρ, PAR):
    """ Returns the derivative of the reference energy for the Mie-Gruneisen EOS
    """
    EOS = PAR.EOS

    if EOS == STIFFENED_GAS:
        return 0

    elif EOS == SHOCK_MG:
        c02 = PAR.c02
        ρ0 = PAR.ρ0
        s = PAR.s
        if ρ > ρ0:
            return - (ρ-ρ0) * ρ0 * c02 / (s * (ρ-ρ0) - ρ)**3
        else:
            return 0

    elif EOS == JWL or EOS == COCHRAN_CHAN:
        return e_ref(ρ, PAR) / ρ**2

def dedρ(ρ, p, PAR):
    """ Returns the derivative of the Mie-Gruneisen internal energy
        with respect to ρ
    """
    Γ = Γ_MG(ρ, PAR)
    dΓ  = dΓ_MG(ρ, PAR)
    pr  =  p_ref(ρ, PAR)
    dpr = dp_ref(ρ, PAR)
    der = de_ref(ρ, PAR)
    return der - (dpr*ρ*Γ + (Γ+ρ*dΓ)*(p-pr)) / (ρ*Γ)**2

def dedp(ρ, PAR):
    """ Returns the derivative of the Mie-Gruneisen internal energy
        with respect to p
    """
    Γ = Γ_MG(ρ, PAR)
    return 1 / (ρ*Γ)


def dTdρ(ρ, p, PAR):
    """ Returns the derivative of the Mie-Gruneisen temperature
        with respect to ρ
    """
    cv = PAR.cv
    Γ = Γ_MG(ρ, PAR)
    dΓ  = dΓ_MG(ρ, PAR)
    pr  =  p_ref(ρ, PAR)
    dpr = dp_ref(ρ, PAR)
    return - (dpr*ρ*Γ + (Γ+ρ*dΓ)*(p-pr)) / (ρ*Γ)**2 / cv

def dTdp(ρ, PAR):
    """ Returns the derivative of the Mie-Gruneisen temperature
        with respect to p
    """
    cv = PAR.cv
    return dedp(ρ, PAR) / cv
