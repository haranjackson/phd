from numpy import exp


def Γ_MG(ρ, PAR):
    """ Returns the Mie-Gruneisen parameter
    """
    if PAR.EOS == 'sg':
        return PAR.γ - 1
    elif PAR.EOS == 'jwl':
        return PAR.Γ0
    elif PAR.EOS == 'smg':
        Γ0 = PAR.Γ0
        ρ0 = PAR.ρ0
        return Γ0 * ρ0 / ρ

def p_ref(ρ, PAR):
    """ Returns the reference pressure in the Mie-Gruneisen EOS
    """
    if PAR.EOS == 'sg':
        return - PAR.γ * PAR.pINF
    elif PAR.EOS == 'jwl':
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v_ = - ρ0 / ρ
        return A * exp(R1 * v_) + B * exp(R2 * v_)
    elif PAR.EOS == 'smg':
        c02 = PAR.c02
        v0 = PAR.v0
        s = PAR.s
        v = 1 / ρ
        return c02 * (v0 - v) / (v0 - s * (v0 - v))**2

def e_ref(ρ, PAR):
    """ Returns the reference energy for the Mie-Gruneisen EOS
    """
    if PAR.EOS == 'sg':
        return 0
    elif PAR.EOS == 'jwl':
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v0 = PAR.v0
        v_ = - ρ0 / ρ
        return A * v0 * exp(R1 * v_) / R1  +  B * v0 * exp(R2 * v_) / R2
    elif PAR.EOS == 'smg':
        v0 = PAR.v0
        v = 1 / ρ
        p0 = p_ref(ρ, PAR)
        return 0.5 * p0 * (v0 - v)

def dΓ_MG(ρ, PAR):
    """ Returns the derivative of the Mie-Gruneisen parameter
    """
    if PAR.EOS == 'sg':
        return 0
    elif PAR.EOS == 'jwl':
        return 0
    elif PAR.EOS == 'smg':
        Γ0 = PAR.Γ0
        ρ0 = PAR.ρ0
        return - Γ0 * ρ0 / ρ**2

def dp_ref(ρ, PAR):
    """ Returns the derivative of the reference pressure in the Mie-Gruneisen EOS
    """
    if PAR.EOS == 'sg':
        return 0
    elif PAR.EOS == 'jwl':
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v_ = - ρ0 / ρ
        return -v_ / ρ * (A * R1 * exp(R1 * v_) + B * R2 * exp(R2 * v_))
    elif PAR.EOS == 'smg':
        c02 = PAR.c02
        ρ0 = PAR.ρ0
        s = PAR.s
        return c02 * ρ0**2 * (s*(ρ0-ρ) - ρ) / (s*(ρ-ρ0) - ρ)**3

def de_ref(ρ, PAR):
    """ Returns the derivative of the reference energy for the Mie-Gruneisen EOS
    """
    if PAR.EOS == 'sg':
        return 0
    elif PAR.EOS == 'jwl':
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v0 = PAR.v0
        v_ = - ρ0 / ρ
        return (A * exp(R1 * v_)  +  B * exp(R2 * v_)) / ρ**2
    elif PAR.EOS == 'smg':
        c02 = PAR.c02
        ρ0 = PAR.ρ0
        s = PAR.s
        return - (ρ-ρ0) * ρ0 * c02 / (s * (ρ-ρ0) - ρ)**3

def dedρ(ρ, p, PAR):
    """ Returns the derivative of the Mie-Gruneisen internal energy
        with respect to ρ
    """
    Γ = Γ_MG(ρ, PAR)
    dΓ  = dΓ_MG(ρ, PAR)
    p0  =  p_ref(ρ, PAR)
    dp0 = dp_ref(ρ, PAR)
    e0  =  e_ref(ρ, PAR)
    de0 = de_ref(ρ, PAR)
    return de0 - (dp0*ρ*Γ + (Γ+ρ*dΓ)*(p-p0)) / (ρ*Γ)**2

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
    return - (dp0*ρ*Γ + (Γ+ρ*dΓ)*(p-p0)) / (ρ*Γ)**2 / cv

def dTdp(ρ, PAR):
    """ Returns the derivative of the Mie-Gruneisen temperature
        with respect to p
    """
    cv = PAR.cv
    return dedp(ρ, PAR) / cv
