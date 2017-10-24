from numba import jitclass, float64 as f64
from numpy import array, expand_dims

from options import PARA_DG, PARA_FV, USE_CPP
from gpr.variables.state import temperature


if USE_CPP:
    import GPRpy
    def CParameters(PAR):
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

if PARA_DG or PARA_FV:
    class material_params():
        def __init__(self, Rc, γ, pINF, cv, ρ0, p0, T0, cs, α, μ, Pr, κ,
                     τ1, τ2, Qc, Kc, Ti, Ea, Bc):
            self.Rc = Rc

            self.γ = γ
            self.pINF = pINF
            self.cv = cv

            self.ρ0 = ρ0
            self.p0 = p0
            self.T0 = T0

            self.cs = cs
            self.α = α
            self.μ = μ
            self.Pr = Pr
            self.τ1 = τ1
            self.κ = κ
            self.τ2 = τ2

            self.Qc = Qc
            self.Kc = Kc
            self.Ti = Ti
            self.Ea = Ea
            self.Bc = Bc

            self.cs2 = self.cs**2
            self.α2 = self.α**2
else:
    @jitclass([('Rc',f64),
               ('γ', f64),('pINF',f64), ('cv',f64), ('ρ0',f64), ('p0',f64),
               ('T0',f64), ('cs',f64), ('α',f64), ('μ',f64), ('Pr',f64),
               ('τ1',f64), ('κ',f64), ('τ2',f64), ('Qc',f64), ('Kc',f64),
               ('Ti',f64), ('Ea',f64), ('Bc',f64), ('cs2',f64), ('α2',f64)])
    class material_params():
        def __init__(self, Rc, γ, pINF, cv, ρ0, p0, T0, cs, α, μ, Pr, κ,
                     τ1, τ2, Qc, Kc, Ti, Ea, Bc):
            self.Rc = Rc

            self.γ = γ
            self.pINF = pINF
            self.cv = cv

            self.ρ0 = ρ0
            self.p0 = p0
            self.T0 = T0

            self.cs = cs
            self.α = α
            self.μ = μ
            self.Pr = Pr
            self.τ1 = τ1
            self.κ = κ
            self.τ2 = τ2

            self.Qc = Qc
            self.Kc = Kc
            self.Ti = Ti
            self.Ea = Ea
            self.Bc = Bc

            self.cs2 = self.cs**2
            self.α2 = self.α**2


def material_parameters(Rc=8.314459848, γ=None, pINF=None, cv=None, ρ0=None,
                        p0=None, cs=None, α=None, μ=None, κ=None, Pr=None,
                        Qc=None, Kc=None, Ti=None, ε=None, Bc=None):
    """ An object to hold the material constants
    """
    if pINF is None:
        pINF = 0

    T0 = temperature(ρ0, p0, γ, pINF, cv)

    if cs is not None:
        τ1 = 6 * μ / (ρ0 * cs**2)
    else:
        cs = 0
        τ1 = 0

    if α is not None:
        if Pr is None:
            κ = κ
            Pr = μ * γ * cv / κ
        else:
            κ = μ * γ * cv / Pr
        τ2 = κ * ρ0 / (T0 * α**2)
    else:
        α = 0
        Pr = 0
        κ = 0
        τ2 = 0

    if Qc is None:
        Qc = 0
    if Kc is None:
        Kc = 0
    if Ti is None:
        Ti = 0
    if Bc is None:
        Bc = 0

    if ε is None:
        Ea = 0
    else:
        Ea = Rc * T0 / ε

    return material_params(Rc, γ, pINF, cv, ρ0, p0, T0, cs, α, μ, Pr, κ,
                           τ1, τ2, Qc, Kc, Ti, Ea, Bc)

class Data():
    """ An object to hold the arrays in which simulation data are saved
    """
    def __init__(self, u, interfaceLocs, t):
        self.grid = u.copy()
        self.time = t
        self.int = array(interfaceLocs)
