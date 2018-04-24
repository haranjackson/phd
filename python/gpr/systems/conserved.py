from numpy import dot, zeros

from gpr.misc.structures import State
from gpr.systems.jacobians import dFdP, dPdQ
from gpr.variables.eos import total_energy


class SystemConserved():

    def __init__(self, VISCOUS, THERMAL, REACTIVE=False, MULTI=False, LSET=0):

        self.VISCOUS = VISCOUS
        self.THERMAL = THERMAL
        self.REACTIVE = REACTIVE
        self.MULTI = MULTI
        self.LSET = LSET

        self.NV = 5 + int(VISCOUS) * 9 + int(THERMAL) * 3 + int(REACTIVE) + \
            int(MULTI) * 2 + LSET

    def F(self, Q, d, MP):

        NV = len(Q)
        ret = zeros(NV)

        P = State(Q, MP)

        ρ = P.ρ
        p = P.p()
        E = P.E
        v = P.v

        vd = v[d]
        ρvd = ρ * vd

        if self.MULTI:
            ρ1 = P.ρ1
            z = P.z
            ret[0] = z * ρ1 * vd
        else:
            ret[0] = ρvd

        ret[1] = ρvd * E + p * vd
        ret[2:5] = ρvd * v
        ret[2 + d] += p

        if self.VISCOUS:

            A = P.A
            σ = P.σ()

            σd = σ[d]
            ret[1] -= dot(σd, v)
            ret[2:5] -= σd

            Av = dot(A, v)
            ret[5 + d] = Av[0]
            ret[8 + d] = Av[1]
            ret[11 + d] = Av[2]

        if self.THERMAL:

            cα2 = MP.cα2

            J = P.J
            T = P.T()
            q = P.q()

            ret[1] += q[d]
            ret[14:17] = ρvd * J
            ret[14 + d] += T

        if self.MULTI:

            λ = P.λ
            ρ2 = P.ρ2

            ret[17] = (1 - z) * ρ2 * vd
            ret[18] = ρvd * z

            if self.REACTIVE:
                ret[19] = (1 - z) * ρ2 * vd * λ

        elif self.REACTIVE:
            ret[17] = ρ * vd * λ

        return ret

    def S(self, Q, MP):

        NV = len(Q)
        ret = zeros(NV)

        P = State(Q, MP)

        ρ = P.ρ

        ret[2:5] = P.f_body()

        if self.VISCOUS:
            ψ = P.ψ()
            θ1_1 = P.θ1_1()
            ret[5:14] = - ψ.ravel() * θ1_1

        if self.THERMAL:
            H = P.H()
            θ2_1 = P.θ2_1()
            ret[14:17] = - ρ * H * θ2_1

        if self.REACTIVE:
            K = - P.K()
            if self.MULTI:
                z = P.z
                ρ2 = P.ρ2
                ret[19] = (1 - z) * ρ2 * K
            else:
                ret[17] = ρ * K

        return ret

    def B(self, Q, d, MP):

        NV = len(Q)
        ret = zeros([NV, NV])

        if self.VISCOUS:

            P = State(Q, MP)

            v = P.v
            vd = v[d]

            for i in range(5, 14):
                ret[i, i] = vd
            ret[5 + d, 5 + d:8 + d] -= v
            ret[8 + d, 8 + d:11 + d] -= v
            ret[11 + d, 11 + d:14 + d] -= v

        for i in range(1, self.LSET + 1):
            ret[-i, -i] = vd

        return ret

    def M(self, Q, d, MP):
        """ Returns the Jacobian in the dth direction
        """
        NV = len(Q)
        P = State(Q, MP)
        DFDP = dFdP(self, P, d, MP)
        DPDQ = dPdQ(self, P, MP)
        return dot(DFDP, DPDQ) + self.B_cons(Q, d, MP)

    def Cvec(self, ρ1, p, v, A, J, MP, ρ2=None, z=1, λ=None):
        """ Returns vector of conserved variables, given primitive variables
        """
        Q = zeros(self.NV)

        if self.MULTI:
            ρ = z * ρ1 + (1 - z) * ρ2
            Q[0] = z * ρ1
            Q[17] = (1 - z) * ρ2
            Q[18] = z * ρ

            if self.REACTIVE:
                Q[19] = (1 - z) * ρ2 * λ
        else:
            ρ = ρ1
            Q[0] = ρ

            if self.REACTIVE:
                Q[17] = ρ * λ

        Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, MP, self.VISCOUS,
                                self.THERMAL, self.REACTIVE)
        Q[2:5] = ρ * v

        if self.VISCOUS:
            Q[5:14] = A.ravel()

        if self.THERMAL:
            Q[14:17] = ρ * J

        return Q
