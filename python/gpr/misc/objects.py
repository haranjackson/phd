import GPRpy

from numpy import zeros

from gpr.vars.mg import eos_text_to_code


class hyperelastic_params():
    def __init__(self, ρ0, α, β, γ, cv, T0, b0, c0):
        self.ρ0 = ρ0
        self.α = α
        self.β = β
        self.γ = γ
        self.cv = cv
        self.T0 = T0
        self.K0 = c0**2 - 4 / 3 * b0**2
        self.b02 = b0**2


def params(MP, Rc, EOS,
           ρ0, Tref, cv,
           α, β, γ, pINF,
           c0, Γ0, s,
           A, B, R1, R2,
           b0, τ0, μ, σY, n, POWER_LAW, SOLID,
           cα, κ,
           REACTION, Qc,
           Kc, Ti,
           Bc, Ea,
           I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2,
           δp):

    MP.Rc = Rc
    MP.EOS = eos_text_to_code(EOS)

    MP.ρ0 = ρ0

    if cv is not None:
        MP.cv = cv
        MP.Tref = Tref

    if EOS == 'sg':
        MP.γ = γ
        MP.pINF = pINF

    elif EOS == 'smg':
        MP.Γ0 = Γ0
        MP.c02 = c0**2
        MP.s = s

    elif EOS == 'gr':
        MP.c02 = c0**2
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
        MP.b02 = b0**2
        MP.β = β
        MP.POWER_LAW = POWER_LAW
        MP.SOLID = SOLID
        if μ:
            MP.μ = μ
        if n:
            MP.n = n
        if σY:
            MP.σY = σY
            MP.τ0 = τ0

    if cα is not None:
        MP.cα2 = cα**2
        MP.κ = κ

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

    if δp is None:
        MP.δp = zeros(3)
    else:
        MP.δp = δp


def material_params(EOS, ρ0,
                    cv=None, Tref=None,
                    α=None, β=None, γ=None, pINF=None,
                    c0=None, Γ0=None, s=None,
                    A=None, B=None, R1=None, R2=None,
                    b0=None, μ=None, τ0=None, σY=None, n=None,
                    cα=None, κ=None, Pr=None,
                    REACTION=None, Qc=None,
                    Kc=None, Ti=None,
                    Bc=None, Ea=None,
                    I=None, G1=None, G2=None, a=None, b=None, c=None,
                    d=None, e=None, g=None, x=None, y=None, z=None,
                    φIG=None, φG1=None, φG2=None,
                    δp=None, Rc=8.31445985):

    assert(EOS in ['sg', 'smg', 'jwl', 'cc', 'gr', 'vac'])
    assert(REACTION in ['a', 'd', 'ig', None])

    MP = GPRpy.classes.Par()

    if EOS == 'vac':
        MP.EOS = -1

    else:
        if Tref is None:
            Tref = 0

        if (γ is not None) and (pINF is None):
            pINF = 0

        if n is None:
            n = 1
            POWER_LAW = False
            SOLID = False if μ else True
        else:
            POWER_LAW = True
            SOLID = True if σY else False

        if β is None:
            β = 0

        if Pr is not None:
            κ = μ * γ * cv / Pr

        params(MP, Rc, EOS,
               ρ0, Tref, cv,
               α, β, γ, pINF,
               c0, Γ0, s,
               A, B, R1, R2,
               b0, τ0, μ, σY, n, POWER_LAW, SOLID,
               cα, κ,
               REACTION, Qc,
               Kc, Ti,
               Bc, Ea,
               I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2,
               δp)

    return MP
