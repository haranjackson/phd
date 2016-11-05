from gpr.variables.eos import E_1r, E_2J
from gpr.variables.vectors import primitive, conserved
from options import reactiveEOS


def thermal_conversion(fluids, PARs, SYS):
    """ Sets the pressure and density across the domain in the isobaric cookoff technique,
        given the temperature calculated with the reduced thermal conduction system
    """
    for i in range(len(fluids)):
        fluid = fluids[i]
        PAR = PARs[i]
        γ = PAR.γ; cv = PAR.cv;

        n = len(fluid)
        Etot = sum(fluid[:, 0, 0, 1])   # Total specific energy in domain
        temp = 0

        for j in range(n):
            Q = fluid[j, 0, 0]
            P = primitive(Q, PAR, SYS)
            temp += E_2J(P.J, PAR.α2) / P.T
            if reactiveEOS:
                temp += E_1r(P.c, PAR.Qc) / P.T

        p_t = ((γ-1) * Etot - n * γ * PAR.pINF) / (n + temp / cv)   # Average pressure

        for j in range(n):
            Q = fluid[j, 0, 0]
            P = primitive(Q, PAR, SYS)
            ρ = p_t / ((γ-1) * cv * P.T)
            fluid[j, 0, 0] = conserved(ρ, p_t, P.v, P.A, P.J, P.λ, PAR, SYS)
