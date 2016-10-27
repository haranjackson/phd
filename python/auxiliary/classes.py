from numba import boolean, float64, jitclass
from numpy import array, expand_dims

from options import Rc
import gpr


@jitclass([('γ', float64), ('pINF', float64), ('cv', float64), ('ρ0', float64), ('p0', float64),
           ('T0', float64), ('cs', float64), ('α', float64), ('μ', float64), ('Pr', float64),
           ('t1', float64), ('κ', float64), ('t2', float64), ('Qc', float64), ('Kc', float64),
           ('Ti', float64), ('Ea', float64), ('Bc', float64), ('cs2', float64), ('α2', float64)])
class material_params():
    def __init__(self, γ, pINF, cv, ρ0, p0, T0, cs, α, μ, Pr, κ, t1, t2, Qc, Kc, Ti, Ea, Bc):
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
        self.t1 = t1
        self.κ = κ
        self.t2 = t2

        self.Qc = Qc
        self.Kc = Kc
        self.Ti = Ti
        self.Ea = Ea
        self.Bc = Bc

        self.cs2 = self.cs**2
        self.α2 = self.α**2

def material_parameters(γ=None, pINF=None, cv=None, ρ0=None, p0=None,
                 cs=None, α=None, μ=None, κ=None, Pr=None,
                 Qc=None, Kc=None, Ti=None, ε=None, Bc=None):
    """ An object to hold the material constants
    """
    if pINF is None:
        pINF = 0

    T0 = gpr.variables.state.temperature(ρ0, p0, γ, pINF, cv)

    if cs is not None:
        t1 = 6 * μ / (ρ0 * cs**2)
    else:
        cs = 0
        t1 = 0

    if α is not None:
        if Pr is None:
            κ = κ
            Pr = μ * γ * cv / κ
        else:
            κ = μ * γ * cv / Pr
        t2 = κ * ρ0 / (T0 * α**2)
    else:
        α = 0
        Pr = 0
        κ = 0
        t2 = 0

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

    return material_params(γ, pINF, cv, ρ0, p0, T0, cs, α, μ, Pr, κ, t1, t2, Qc, Kc, Ti, Ea, Bc)

@jitclass([('mechanical', boolean), ('viscous', boolean), ('thermal', boolean),
           ('reactive', boolean)])
class active_subsystems():
    """ An object to hold information on which components of the GPR model are activated
    """
    def __init__(self, mechanical=1, viscous=1, thermal=1, reactive=1):
        self.mechanical = mechanical
        self.viscous = viscous
        self.thermal = thermal
        self.reactive = reactive

class save_arrays():
    """ An object to hold the arrays in which simulation data are saved
    """
    def __init__(self, u, intLocs):
        self.data = expand_dims(u.copy(), axis=0)
        self.time = array([0])
        self.interfaces = expand_dims(array(intLocs), axis=0)
