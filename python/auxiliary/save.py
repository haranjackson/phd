from codecs import open
from os import makedirs, path
from time import time

from numpy import array, concatenate, expand_dims, linspace, int64, save, zeros

from options import tf, Lx, Ly, Lz, nx, ny, nz
from options import mechanical, viscous, thermal, reactive
from options import W, doubleTime, reactionType, fullBurn, burnProp
from options import GFM, RGFM, isoFix, SFix, TFix, UPDATE_STEP
from options import solver, convertTemp, altThermSolve
from options import fullODE, wenoHalfStep, StrangSplit, approxInterface, reconstructPrim
from options import  N, CFL, method, perronFrob
from options import hidalgo, stiff, superStiff, failLim, TOL
from options import rc, λc, λs, eps
from options import MAX_ITER, paraDG, paraFV, ncore


def print_stats(count, t, dt, interfaceLocations, SYS):
    print(count+1)
    print('t  =', t)
    print('dt =', dt)
    print('M,V,T,R =', SYS.mechanical, SYS.viscous, SYS.thermal, SYS.reactive)
    if GFM:
        print('Interfaces =', interfaceLocations)

def record_data(fluids, inds, t, interfaceLocations, saveArrays):
    """ Appends the latest data and timestep to the recording arrays
    """
    u = zeros(fluids[0].shape)
    for i in range(len(fluids)):
        l = inds[i]
        r = inds[i+1]
        u[l:r] = fluids[i][l:r]
    saveArrays.data = concatenate([saveArrays.data, expand_dims(u, axis=0)])
    saveArrays.time = concatenate([saveArrays.time, array([t])])
    saveArrays.interfaces = concatenate([saveArrays.interfaces,
                                         expand_dims(array(interfaceLocations), axis=0)])

def save_config(path):
    with open(path, 'w+', encoding='utf-8') as f:
        f.write('tf = %e\n' % tf)
        f.write('Lx = %e\n' % Lx)
        f.write('Ly = %e\n' % Ly)
        f.write('Lz = %e\n' % Lz)
        f.write('nx = %i\n' % nx)
        f.write('ny = %i\n' % ny)
        f.write('nz = %i\n\n' % nz)

        f.write('mechanical = %i\n' % mechanical)
        f.write('viscous    = %i\n' % viscous)
        f.write('thermal    = %i\n' % thermal)
        f.write('reactive   = %i\n\n' % reactive)

        f.write('W  = %e\n' % W)
        f.write('doubleTime = %e\n' % doubleTime)
        f.write('reactionType = %s\n' % reactionType)
        f.write('fullBurn = %i\n' % fullBurn)
        f.write('burnProp = %f\n\n' % burnProp)

        f.write('GFM  = %i\n' % GFM)
        f.write('RGFM = %i\n' % RGFM)
        f.write('isoFix = %i\n' % isoFix)
        f.write('SFix   = %i\n' % SFix)
        f.write('TFix   = %i\n' % TFix)
        f.write('UPDATE_STEP = %i\n\n' % UPDATE_STEP)

        f.write('solver  = %s\n' % solver)
        f.write('convertTemp   = %i\n' % convertTemp)
        f.write('altThermSolve = %i\n\n' % altThermSolve)

        f.write('fullODE = %i\n' % fullODE)
        f.write('wenoHalfStep = %i\n' % wenoHalfStep)
        f.write('StrangSplit  = %i\n' % StrangSplit)
        f.write('approxInterface = %i\n' % approxInterface)
        f.write('reconstructPrim = %i\n\n' % reconstructPrim)

        f.write('N      = %i\n' % N)
        f.write('CFL    = %f\n' % CFL)
        f.write('method = %s\n' % method)
        f.write('perronFrob = %i\n\n' % perronFrob)

        f.write('hidalgo    = %i\n' % hidalgo)
        f.write('stiff      = %i\n' % stiff)
        f.write('superStiff = %i\n' % superStiff)
        f.write('failLim    = %i\n' % failLim)
        f.write('TOL        = %e\n' % TOL)
        f.write('MAX_ITER   = %i\n\n' % MAX_ITER)

        f.write('rc  = %f\n' % rc)
        f.write('λc  = %e\n' % λc)
        f.write('λs  = %e\n' % λs)
        f.write('eps = %e\n\n' % eps)

        f.write('paraDG = %i\n' % paraDG)
        f.write('paraFV = %i\n' % paraFV)
        f.write('ncore  = %i\n\n' % ncore)

def save_all(saveArrays):
    if not path.exists('_dump'):
        makedirs('_dump')
    save('_dump/dataArray%d.npy' % time(), saveArrays.data)
    save('_dump/timeArray%d.npy' % time(), saveArrays.time)
    save('_dump/interArray%d.npy' % time(), saveArrays.interfaces)
    save_config('_dump/options%d.txt' % time())

def compress_arrays(saveArrays, N):
    n = len(saveArrays.time)
    inds = linspace(0,n-1,N,dtype=int64)
    return [saveArrays.data[inds], saveArrays.time[inds], saveArrays.interfaces[inds]]
