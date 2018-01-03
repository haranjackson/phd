from system.gpr.variables.state import temperature
from system.gpr.variables.mg import eos_text_to_code
from options import USE_CPP


class hyperelastic_params():
    def __init__(self, ρ0, α, β, γ, cv, T0, b0, c0):
        self.ρ0 = ρ0
        self.α = α
        self.β = β
        self.γ = γ
        self.cv = cv
        self.T0 = T0
        self.K0 = c0**2 - 4/3 * b0**2
        self.B0 = b0**2

class EOS_params():
    def __init__(self, EOS, ρ0, cv, p0, γ, pINF, c0, Γ0, s, A, B, R1, R2):

        self.EOS = eos_text_to_code(EOS)

        self.ρ0 = ρ0
        self.cv = cv
        self.p0 = p0

        if EOS == 'sg':
            self.γ = γ
            self.pINF = pINF

        elif EOS == 'smg':
            self.Γ0 = Γ0
            self.c02 = c0**2
            self.s = s

        elif EOS == 'jwl' or EOS == 'cc':
            self.Γ0 = Γ0
            self.A = A
            self.B = B
            self.R1 = R1
            self.R2 = R2

def params(PAR, Rc, EOS, ρ0, p0, T0, cv,
           γ, pINF,
           c0, Γ0, s, e0,
           A, B, R1, R2,
           cs, β, τ1, μ, σY, n, PLASTIC,
           α, τ2,
           REACTION, Qc,
           Kc, Ti,
           Bc, Ea,
           I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2):

    PAR.Rc = Rc
    PAR.EOS = eos_text_to_code(EOS)

    PAR.ρ0 = ρ0
    PAR.p0 = p0
    PAR.T0 = T0
    PAR.cv = cv

    if EOS == 'sg':
        PAR.γ = γ
        PAR.pINF = pINF

    elif EOS == 'smg':
        PAR.Γ0 = Γ0
        PAR.c02 = c0**2
        PAR.s = s
        PAR.e0 = e0

    elif EOS == 'jwl' or EOS == 'cc':
        PAR.Γ0 = Γ0
        PAR.A = A
        PAR.B = B
        PAR.R1 = R1
        PAR.R2 = R2

    if cs is not None:
        PAR.cs2 = cs**2
        PAR.β = β
        PAR.τ1 = τ1
        PAR.PLASTIC = PLASTIC
        if PLASTIC:
            PAR.σY = σY
            PAR.n = n

    if α is not None:
        PAR.α2 = α**2
        PAR.τ2 = τ2

    if REACTION is not None:
        PAR.REACTION = REACTION
        PAR.Qc = Qc

        if REACTION == 'd':
            PAR.Kc = Kc
            PAR.Ti = Ti

        elif REACTION == 'a':
            PAR.Ea = Ea
            PAR.Bc = Bc

        elif REACTION == 'i':
            PAR.I = I
            PAR.G1 = G1
            PAR.G2 = G2
            PAR.a = a
            PAR.b = b
            PAR.c = c
            PAR.d = d
            PAR.e = e
            PAR.g = g
            PAR.x = x
            PAR.y = y
            PAR.z = z
            PAR.φIG = φIG
            PAR.φG1 = φG1
            PAR.φG2 = φG2

def material_parameters(EOS, ρ0, cv, p0=None,
                        γ=None, pINF=None,
                        c0=None, Γ0=None, s=None, e0=None,
                        A=None, B=None, R1=None, R2=None,
                        cs=None, β=None, μ=None, τ1=None, σY=None, n=None, PLASTIC=False,
                        α=None, κ=None, Pr=None,
                        REACTION=None, Qc=None,
                        Kc=None, Ti=None,
                        Bc=None, Ea=None,
                        I=None, G1=None, G2=None, a=None, b=None, c=None,
                        d=None, e=None, g=None, x=None, y=None, z=None,
                        φIG=None, φG1=None, φG2=None,
                        Rc=8.31445985):

    """ An object to hold the material constants
    """
    assert(EOS in ['sg', 'smg', 'jwl', 'cc'])
    assert(REACTION in ['a', 'd', 'ig', None])

    if (γ is not None) and (pINF is None):
        pINF = 0

    P = EOS_params(EOS, ρ0, cv, p0, γ, pINF, c0, Γ0, s, A, B, R1, R2)
    T0 = temperature(ρ0, p0, P)

    if cs is not None:
        if (not PLASTIC) and (τ1 is None):
            τ1 = 6 * μ / (ρ0 * cs**2)
        if β is None:
            β = 0

    if α is not None:
        if Pr is not None:
            κ = μ * γ * cv / Pr
        τ2 = κ * ρ0 / (T0 * α**2)
    else:
        τ2 = None

    if USE_CPP:
        import GPRpy
        PAR = GPRpy.classes.Par()
    else:
        class PAR: pass

    params(PAR, Rc, EOS, ρ0, p0, T0, cv,
           γ, pINF,
           c0, Γ0, s, e0,
           A, B, R1, R2,
           cs, β, τ1, μ, σY, n, PLASTIC,
           α, τ2,
           REACTION, Qc,
           Kc, Ti,
           Bc, Ea,
           I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2)

    return PAR
