from system.gpr.variables.state import temperature


def CParameters(GPRpy, PAR):
    MP = GPRpy.classes.Par()
    MP.gamma = PAR.γ
    MP.cv = PAR.cv
    MP.pinf = PAR.pINF
    MP.r0 = PAR.ρ0
    MP.p0 = PAR.p0
    MP.T0 = PAR.T0
    MP.cs2 = PAR.cs2
    MP.mu = PAR.μ
    MP.tau1 = PAR.τ1
    MP.alpha2 = PAR.α2
    MP.kappa = PAR.κ
    MP.tau2 = PAR.τ2
    return MP

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
    def __init__(self, EOS, ρ0, cv, p0, γ, pINF, c0, Γ0, s, A, B, R1, R2, ε1, ε2):
        self.EOS = EOS
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

        elif EOS == 'jwl':
            self.Γ0 = Γ0
            self.A = A
            self.B = B
            self.R1 = R1
            self.R2 = R2

        elif EOS == 'cc':
            self.Γ0 = Γ0
            self.A = A
            self.B = B
            self.ε1 = ε1
            self.ε2 = ε2

class params():
    def __init__(self, Rc, EOS, ρ0, p0, T0, cv,
                 γ, pINF,
                 c0, Γ0, s, e0,
                 A, B, R1, R2, ε1, ε2,
                 cs, β, τ1, μ, σY, n, PLASTIC,
                 α, τ2,
                 REACTION, Qc,
                 Kc, Ti,
                 Bc, Ea,
                 I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2):

        self.Rc = Rc
        self.EOS = EOS

        self.ρ0 = ρ0
        self.p0 = p0
        self.T0 = T0
        self.cv = cv

        if EOS == 'sg':
            self.γ = γ
            self.pINF = pINF

        elif EOS == 'smg':
            self.Γ0 = Γ0
            self.c02 = c0**2
            self.s = s
            self.e0 = e0

        elif EOS == 'jwl':
            self.Γ0 = Γ0
            self.A = A
            self.B = B
            self.R1 = R1
            self.R2 = R2

        elif EOS == 'cc':
            self.Γ0 = Γ0
            self.A = A
            self.B = B
            self.ε1 = ε1
            self.ε2 = ε2

        if cs is not None:
            self.cs2 = cs**2
            self.β = β
            self.τ1 = τ1
            self.PLASTIC = PLASTIC
            if PLASTIC:
                self.σY = σY
                self.n = n
            else:
                self.μ = μ

        if α is not None:
            self.α2 = α**2
            self.τ2 = τ2

        if REACTION is not None:
            self.REACTION = REACTION
            self.Qc = Qc

            if REACTION == 'd':
                self.Kc = Kc
                self.Ti = Ti

            elif REACTION == 'a':
                self.Ea = Ea
                self.Bc = Bc

            elif REACTION == 'i':
                self.I = I
                self.G1 = G1
                self.G2 = G2
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.e = e
                self.g = g
                self.x = x
                self.y = y
                self.z = z
                self.φIG = φIG
                self.φG1 = φG1
                self.φG2 = φG2

def material_parameters(EOS, ρ0, cv, p0=None,
                        γ=None, pINF=None,
                        c0=None, Γ0=None, s=None, e0=None,
                        A=None, B=None, R1=None, R2=None, ε1=None, ε2=None,
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

    P = EOS_params(EOS, ρ0, cv, p0, γ, pINF, c0, Γ0, s, A, B, R1, R2, ε1, ε2)
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

    return params(Rc, EOS, ρ0, p0, T0, cv,
                  γ, pINF,
                  c0, Γ0, s, e0,
                  A, B, R1, R2, ε1, ε2,
                  cs, β, τ1, μ, σY, n, PLASTIC,
                  α, τ2,
                  REACTION, Qc,
                  Kc, Ti,
                  Bc, Ea,
                  I, G1, G2, a, b, c, d, e, g, x, y, z, φIG, φG1, φG2)
