from os import makedirs, path
from time import time

from numpy import array, concatenate, expand_dims, linspace, int64, save

from options import tf, L, nx, ny, nz
from options import mechanical, viscous, thermal, reactive
from options import Ms, W, doubleTime, fullBurn, burnProp
from options import reactionType, reactiveEOS, minE
from options import GFM, RGFM, isoFix, entropyFix, tempFix, UPDATE_STEP
from options import N, CFL, method, renormaliseRho, convertTemp, NOISE_LIM
from options import hidalgo, stiff, superStiff, failLim, TOL
from options import rc, lam, lams, eps
from options import MAX_ITER, minParaDGLen, minParaFVLen, ncore, reducedDomain


def record_data(u, t, interfaceLocations, dataArray, timeArray, interArray):
    """ Appends the latest data and timestep to the recording arrays
    """
    dataArray = concatenate([dataArray, expand_dims(u, axis=0)])
    timeArray = concatenate([timeArray, array([t])])
    interArray = concatenate([interArray, expand_dims(array(interfaceLocations), axis=0)])
    return dataArray, timeArray, interArray

def save_config(path):
    with open(path, 'w+') as f:
        f.write('tf = %e\n' % tf)
        f.write('L  = %e\n' % L)
        f.write('nx = %i\n' % nx)
        f.write('ny = %i\n' % ny)
        f.write('nz = %i\n\n' % nz)

        f.write('mechanical = %i\n' % mechanical)
        f.write('viscous    = %i\n' % viscous)
        f.write('thermal    = %i\n' % thermal)
        f.write('reactive   = %i\n\n' % reactive)

        f.write('Ms = %f\n' % Ms)
        f.write('W  = %e\n' % W)
        f.write('doubleTime = %e\n' % doubleTime)
        f.write('fullBurn = %i\n' % fullBurn)
        f.write('burnProp = %f\n\n' % burnProp)

        f.write('reactionType = %s\n' % reactionType)
        f.write('reactiveEOS  = %i\n' % reactiveEOS)
        f.write('minE = %i\n\n' % minE)

        f.write('GFM    = %i\n' % GFM)
        f.write('RGFM   = %i\n' % RGFM)
        f.write('isoFix     = %i\n' % isoFix)
        f.write('entropyFix = %i\n' % entropyFix)
        f.write('tempFix    = %i\n' % tempFix)
        f.write('UPDATE_STEP = %i\n\n' % UPDATE_STEP)

        f.write('N      = %i\n' % N)
        f.write('CFL    = %f\n' % CFL)
        f.write('method = %s\n' % method)
        f.write('renormaliseRho = %i\n' % renormaliseRho)
        f.write('convertTemp    = %i\n' % convertTemp)
        f.write('NOISE_LIM = %e\n\n' % NOISE_LIM)

        f.write('hidalgo    = %i\n' % hidalgo)
        f.write('stiff      = %i\n' % stiff)
        f.write('superStiff = %i\n' % superStiff)
        f.write('failLim    = %i\n' % failLim)
        f.write('TOL        = %e\n' % TOL)
        f.write('MAX_ITER   = %i\n\n' % MAX_ITER)

        f.write('rc   = %f\n' % rc)
        f.write('lam  = %e\n' % lam)
        f.write('lams = %e\n' % lams)
        f.write('eps  = %e\n\n' % eps)

        f.write('minParaDGLen = %i\n' % minParaDGLen)
        f.write('minParaFVLen = %i\n' % minParaFVLen)
        f.write('ncore = %i\n' % ncore)
        f.write('reducedDomain = %i\n\n' % reducedDomain)

def save_all(dataArray, timeArray, interArray):
    if not path.exists('_dump'):
        makedirs('_dump')
    save('_dump/dataArray%d.npy' % time(), dataArray)
    save('_dump/timeArray%d.npy' % time(), timeArray)
    save('_dump/interArray%d.npy' % time(), interArray)
    save_config('_dump/options%d.txt' % time())

def compress_arrays(dataArray, timeArray, interArray, N):
    n = len(timeArray)
    inds = linspace(0,n-1,N,dtype=int64)
    return [dataArray[inds], timeArray[inds], interArray[inds]]
