from numpy import array, expand_dims

from options import Rc
import gpr.variables


class material_parameters():
    """ An object to hold the material constants
    """
    def __init__(self, γ=None, pINF=None, cv=None, ρ0=None, p0=None,
                 cs=None, α=None, μ=None, κ=None, Pr=None,
                 Qc=None, Kc=None, Ti=None, ε=None, Bc=None):

        self.γ = γ
        self.pINF = pINF
        self.cv = cv

        self.ρ0 = ρ0
        self.p0 = p0
        self.T0 = gpr.variables.state.temperature(ρ0, p0, γ, pINF, cv)

        self.cs = cs
        self.α = α
        self.μ = μ
        self.Pr = Pr

        if cs is not None:
            self.cs2 = self.cs**2
            self.t1 = 6 * μ / (ρ0 * self.cs2)

        if α is not None:
            self.α2 = self.α**2
            if Pr is None:
                self.κ = κ
            else:
                self.κ = μ * γ * cv / Pr
            self.t2 = self.κ * ρ0 / (self.T0 * self.α2)

        self.Qc = Qc
        if ε is None:
            self.Kc = Kc
            self.Ti = Ti
        else:
            self.Ea = Rc * self.T0 / ε
            self.Bc = Bc

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
