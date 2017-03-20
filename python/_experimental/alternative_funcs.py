from numpy import cos, sin, zeros


def rotmat2(x,y,z):
    cx = cos(x)
    sx = sin(x)
    cy = cos(y)
    sy = sin(y)
    cz = cos(z)
    sz = sin(z)

    ret = zeros([3,3])
    ret[0,0] = cy*cz
    ret[0,1] = sx*sy*cz - cx*sz
    ret[0,2] = cx*sy*cz + sx*sz
    ret[1,0] = cy*sz
    ret[1,1] = sx*sy*sz + cx*cz
    ret[1,2] = cx*sy*sz - sx*cz
    ret[2,0] = -sy
    ret[2,1] = sx*cy
    ret[2,2] = cx*cy
    return ret
