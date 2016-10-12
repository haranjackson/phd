from joblib import delayed
from numpy import array, concatenate

from ader.dg import predictor
from ader.fv import fv_terms
from ader.fv_space_only import fv_terms_space_only
from options import ncore


def para_predictor(pool, wh, params, dt, subsystems):
    """ Controls the parallel computation of the Galerkin predictor
    """
    nx = wh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    n = len(chunk) - 1
    qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i+1]], params, dt, subsystems)
                  for i in range(n))
    return concatenate(qhList)

def para_fv_terms(pool, qh, params, dt, subsystems):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    nx = qh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    chunk[0] += 1
    chunk[-1] -= 1
    n = len(chunk) - 1
    qhList = pool(delayed(fv_terms)(qh[chunk[i]-1:chunk[i+1]+1], params, dt, subsystems)
                  for i in range(n))
    return concatenate(qhList)

def para_fv_terms_space_only(pool, wh, params, dt, subsystems):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    nx = wh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    chunk[0] += 1
    chunk[-1] -= 1
    n = len(chunk) - 1
    qhList = pool(delayed(fv_terms_space_only)(wh[chunk[i]-1:chunk[i+1]+1], params, dt, subsystems)
                  for i in range(n))
    return concatenate(qhList)
