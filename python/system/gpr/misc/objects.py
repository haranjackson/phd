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
    def __init__(self, EOS, ρ0, cv, p0, α, β, γ, pINF, c0, Γ0, s, A, B, R1, R2):

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

        elif EOS == 'gr':
            self.α = α
            self.β = β
            self.γ = γ

        elif EOS == 'jwl' or EOS == 'cc':
            self.Γ0 = Γ0
            self.A = A
            self.B = B
            self.R1 = R1
            self.R2 = R2

def params(MP, Rc, EOS, ρ0, p0, Tref, T0, cv,
           α, β, γ, pINF,
           c0, Γ0, s, e0,
           A, B, R1, R2,
           b0, β, τ1, μ, σY, n, PLASTIC,
           cα, τ2,
           REACTION, Qc,
           Kc, Ti,
           Bc, Ea,
           I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2):

    MP.Rc = Rc
    MP.EOS = eos_text_to_code(EOS)

    MP.ρ0 = ρ0
    MP.p0 = p0
    MP.Tref = Tref
    MP.T0 = T0
    MP.cv = cv

    if EOS == 'sg':
        MP.γ = γ
        MP.pINF = pINF

    elif EOS == 'smg':
        MP.Γ0 = Γ0
        MP.c02 = c0**2
        MP.s = s
        MP.e0 = e0

    if EOS == 'gr':
        MP.α = α
        MP.β = β
        MP.γ = γ

    elif EOS == 'jwl' or EOS == 'cc':
        MP.Γ0 = Γ0
        MP.A = A
        MP.B = B
        MP.R1 = R1
        MP.R2 = R2

    if b0 is not None:
        MP.B0 = b0**2
        MP.β = β
        MP.τ1 = τ1
        MP.PLASTIC = PLASTIC
        if PLASTIC:
            MP.σY = σY
            MP.n = n

    if cα is not None:
        MP.cα2 = cα**2
        MP.τ2 = τ2

    if REACTION is not None:
        MP.REACTION = REACTION
        MP.Qc = Qc

        if REACTION == 'd':
            MP.Kc = Kc
            MP.Ti = Ti

        elif REACTION == 'a':
            MP.Ea = Ea
            MP.Bc = Bc

        elif REACTION == 'i':
            MP.I = I
            MP.G1 = G1
            MP.G2 = G2
            MP.a = a
            MP.b = b
            MP.c = c
            MP.d = d
            MP.e = e
            MP.g = g
            MP.x = x
            MP.y = y
            MP.z = z
            MP.φIG = φIG
            MP.φG1 = φG1
            MP.φG2 = φG2

def material_parameters(EOS, ρ0, cv, p0,
                        Tref=None, α=None, β=None, γ=None, pINF=None,
                        c0=None, Γ0=None, s=None, e0=None,
                        A=None, B=None, R1=None, R2=None,
                        b0=None, β=None, μ=None, τ1=None,
                        σY=None, n=None, PLASTIC=False,
                        cα=None, κ=None, Pr=None,
                        REACTION=None, Qc=None,
                        Kc=None, Ti=None,
                        Bc=None, Ea=None,
                        I=None, G1=None, G2=None, a=None, b=None, c=None,
                        d=None, e=None, g=None, x=None, y=None, z=None,
                        φIG=None, φG1=None, φG2=None,
                        Rc=8.31445985):

    """ An object to hold the material constants
    """
    assert(EOS in ['sg', 'smg', 'jwl', 'cc', 'gr'])
    assert(REACTION in ['a', 'd', 'ig', None])

    if Tref is None:
        Tref = 0

    if (γ is not None) and (pINF is None):
        pINF = 0

    P = EOS_params(EOS, ρ0, cv, p0, α, β, γ, pINF, c0, Γ0, s, A, B, R1, R2)
    T0 = temperature(ρ0, p0, P)

    if b0 is not None:
        if (not PLASTIC) and (τ1 is None):
            τ1 = 6 * μ / (ρ0 * b0**2)
        if β is None:
            β = 0

    if cα is not None:
        if Pr is not None:
            κ = μ * γ * cv / Pr
        τ2 = κ * ρ0 / (T0 * cα**2)
    else:
        τ2 = None

    if USE_CPP:
        import GPRpy
        MP = GPRpy.classes.Par()
    else:
        class MP: pass

    params(MP, Rc, EOS, ρ0, p0, Tref, T0, cv,
           α, β, γ, pINF,
           c0, Γ0, s, e0,
           A, B, R1, R2,
           b0, β, τ1, μ, σY, n, PLASTIC,
           cα, τ2,
           REACTION, Qc,
           Kc, Ti,
           Bc, Ea,
           I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2)

    return MP
