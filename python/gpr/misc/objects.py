import GPRpy

from numpy import zeros

from gpr.vars.mg import EOS_CODES


REACTION_CODES = {'d': 0,
                  'a': 1,
                  'i': 2}


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
           G1, a, b, d, λ0,
           δp,
           MULTI, EOS_2,
           Tref_2, cv_2,
           α_2, β_2, γ_2, pINF_2,
           c0_2, Γ0_2, s_2,
           A_2, B_2, R1_2, R2_2,
           b0_2, μ_2, n_2,
           cα_2, κ_2):

    MP.EOS = EOS_CODES[EOS]

    MP.ρ0 = ρ0

    if cv:
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

    if b0:
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

    if cα:
        MP.cα2 = cα**2
        MP.κ = κ

    if REACTION:

        MP.REACTION = REACTION_CODES[REACTION]
        MP.Qc = Qc

        if REACTION == 'd':
            MP.Kc = Kc
            MP.Ti = Ti

        elif REACTION == 'a':
            MP.Ea = Ea
            MP.Bc = Bc
            MP.Rc = Rc

        elif REACTION == 'i':
            MP.G1 = G1
            MP.a = a
            MP.b = b
            MP.d = d
            MP.λ0 = λ0

    else:
        MP.REACTION = -1

    if δp is not None:
        MP.δp = δp
    else:
        MP.δp = zeros(3)

    if MULTI:

        MP.MULTI = MULTI
        MP.MP2.EOS = EOS_CODES[EOS_2]
        MP.MP2.ρ0 = ρ0

        if cv_2:
            MP.MP2.cv = cv_2
            MP.MP2.Tref = Tref_2

        if EOS == 'sg':
            MP.MP2.γ = γ_2
            MP.MP2.pINF = pINF_2

        elif EOS == 'smg':
            MP.MP2.Γ0 = Γ0_2
            MP.MP2.c02 = c0_2**2
            MP.MP2.s = s_2

        elif EOS == 'gr':
            MP.MP2.c02 = c0_2**2
            MP.MP2.α = α_2
            MP.MP2.β = β_2
            MP.MP2.γ = γ_2

        elif EOS == 'jwl' or EOS == 'cc':
            MP.MP2.Γ0 = Γ0_2
            MP.MP2.A = A_2
            MP.MP2.B = B_2
            MP.MP2.R1 = R1_2
            MP.MP2.R2 = R2_2

        if b0 is not None:
            MP.MP2.b02 = b0_2**2
            MP.MP2.β = β_2
            MP.MP2.μ = μ_2
            if n_2:
                MP.MP2.n = n_2

        if cα:
            MP.MP2.cα2 = cα_2**2
            MP.MP2.κ = κ_2


def material_params(EOS, ρ0,
                    cv=None, Tref=None,

                    α=None, β=None, γ=None, pINF=None,
                    c0=None, Γ0=None, s=None,
                    A=None, B=None, R1=None, R2=None,
                    b0=None, μ=None, τ0=None, σY=None, n=None,
                    cα=None, κ=None, Pr=None,

                    REACTION=None, Qc=None,
                    Kc=None, Ti=None,
                    Bc=None, Ea=None, Rc=None,
                    G1=None, a=None, b=None, d=None, λ0=None,
                    δp=None,

                    MULTI=False, EOS_2=None,
                    Tref_2=None, cv_2=None,
                    α_2=None, β_2=None, γ_2=None, pINF_2=None,
                    c0_2=None, Γ0_2=None, s_2=None,
                    A_2=None, B_2=None, R1_2=None, R2_2=None,
                    b0_2=None, μ_2=None, n_2=None,
                    cα_2=None, κ_2=None, Pr_2=None):

    MP = GPRpy.classes.Par()

    if EOS == 'vac':
        MP.EOS = -1

    else:
        if Tref is None:
            Tref = 0

        if γ and (pINF is None):
            pINF = 0

        if n:
            POWER_LAW = True
            SOLID = True if σY else False
        else:
            n = 1
            POWER_LAW = False
            SOLID = False if μ else True

        if β is None:
            β = 0

        if Pr:
            κ = μ * γ * cv / Pr

        if Tref_2 is None:
            Tref_2 = 0

        if β_2 is None:
            β_2 = 0

        if Pr_2:
            κ_2 = μ_2 * γ_2 * cv_2 / Pr_2

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
               G1, a, b, d, λ0,
               δp,
               MULTI, EOS_2,
               Tref_2, cv_2,
               α_2, β_2, γ_2, pINF_2,
               c0_2, Γ0_2, s_2,
               A_2, B_2, R1_2, R2_2,
               b0_2, μ_2, n_2,
               cα_2, κ_2)

    return MP
