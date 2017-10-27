#include "../../etc/globals.h"
#include "../../system/eig.h"
#include "../../system/equations.h"


VecV Bint(VecVr qL, VecVr qR, int d)
{   // Returns the jump matrix for B, in the dth direction.
    VecV ret = VecV::Zero();
    VecV qJump = qR - qL;
    VecV q, tmp;
    for (int i=0; i<N+1; i++)
    {
        q = qL + NODES(i) * qJump;
        Bdot(tmp, q, qJump, d);
        ret += WGHTS(i) * tmp;
    }
    return ret;
}

VecV Smax(VecVr qL, VecVr qR, int d, bool PERRON_FROBENIUS, Par & MP)
{
    double max1 = max_abs_eigs(qL, d, PERRON_FROBENIUS, MP);
    double max2 = max_abs_eigs(qR, d, PERRON_FROBENIUS, MP);
    return std::max(max1, max2) * (qL - qR);
}



/*
def Aint(pL, pR, qL, qR, d, PAR, SYS):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(18, dtype=complex128)
    Δq = qR - qL
    for i in range(N1):
        q = qL + nodes[i] * Δq
        J = system_conserved(q, d, PAR, SYS)
        λ, R = eig(J, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += weights[i] * dot(R, abs(λ)*b)
    return ret.real
*/
