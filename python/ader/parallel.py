from joblib import delayed
from numpy import array, concatenate

from ader.dg import predictor
from ader.fv import finite_volume_terms
from options import ncore


def parallel_predictor(pool, wh, params, dt, mechanical, viscous, thermal, reactive):
    """ Controls the parallel computation of the Galerkin predictor
    """
    nx = wh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    n = len(chunk) - 1
    qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i+1]], params, dt,
                                     mechanical, viscous, thermal, reactive) for i in range(n))
    return concatenate(qhList)

def parallel_finite_volume_terms(pool, qh, params, dt, mechanical, viscous, thermal, reactive):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    nx = qh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    chunk[0] += 1
    chunk[-1] -= 1
    n = len(chunk) - 1
    qhList = pool(delayed(finite_volume_terms)(qh[chunk[i]-1:chunk[i+1]+1], params, dt,
                                               mechanical, viscous, thermal, reactive)
                  for i in range(n))
    return concatenate(qhList)
