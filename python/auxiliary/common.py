from options import Rc

from gpr.variables import temperature


class material_parameters():
    """ An object to hold the material constants
    """
    def __init__(self, y=None, pINF=None, cv=None, r0=None, p0=None,
                 cs=None, alpha=None, mu=None, kappa=None, Pr=None,
                 Qc=None, Kc=None, Ti=None, epsilon=None, Bc=None):
        self.y = y
        self.pINF = pINF
        self.cv = cv

        self.r0 = r0
        self.p0 = p0
        self.T0 = temperature(r0, p0, y, pINF, cv)

        self.cs = cs
        self.alpha = alpha
        self.mu = mu
        self.Pr = Pr

        if cs is not None:
            self.cs2 = self.cs**2
            self.t1 = 6 * mu / (r0 * self.cs2)

        if alpha is not None:
            self.alpha2 = self.alpha**2
            if Pr is None:
                self.kappa = kappa
            else:
                self.kappa = mu * y * cv / Pr
            self.t2 = self.kappa * r0 / (self.T0 * self.alpha2)

        self.Qc = Qc
        if epsilon is None:
            self.Kc = Kc
            self.Ti = Ti
        else:
            self.Ea = Rc * self.T0 / epsilon
            self.Bc = Bc
