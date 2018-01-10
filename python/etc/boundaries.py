from solvers.weno.weno import extend

from options import ndim


def standard_BC(u, reflect=0):
    ret = extend(u, 1, 0)
    if reflect:
        ret[0, :, :, 2:5] *= -1
        ret[0, :, :, 14:17] *= -1
        ret[-1, :, :, 2:5] *= -1
        ret[-1, :, :, 14:17] *= -1
    if ndim > 1:
        ret = extend(ret, 1, 1)
        if reflect:
            ret[:, 0, :, 2:5] *= -1
            ret[:, 0, :, 14:17] *= -1
            ret[:, -1, :, 2:5] *= -1
            ret[:, -1, :, 14:17] *= -1
        if ndim > 2:
            ret = extend(ret, 1, 2)
            if reflect:
                ret[:, :, 0, 2:5] *= -1
                ret[:, :, 0, 14:17] *= -1
                ret[:, :, -1, 2:5] *= -1
                ret[:, :, -1, 14:17] *= -1
    return ret

def periodic_BC(u):
    ret = extend(u, 1, 0)
    ret[0] = ret[-2]
    ret[-1] = ret[1]
    if ndim > 1:
        ret = extend(ret, 1, 1)
        ret[:,0] = ret[:,-2]
        ret[:,-1] = ret[:,1]
    if ndim > 2:
        ret = extend(ret, 1, 2)
        ret[:,:,0] = ret[:,:,-2]
        ret[:,:,-1] = ret[:,:,1]
    return ret
