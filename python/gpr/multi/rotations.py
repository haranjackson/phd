from numpy import array, dot, sqrt, vstack, zeros


def rotation_matrix(n):
    """ returns the matrix that rotates vector quantities into a coordinate
        system defined by e1=n, e2,e3 ⟂ n
    """
    e1 = zeros(3)
    e1[:len(n)] = n

    if abs(e1[1] + e1[2]) <= abs(e1[1] - e1[2]):
        den = sqrt(2 * (1 - e1[0] * e1[1] - e1[1] * e1[2] - e1[2] * e1[0]))
        e2 = array([e1[1] - e1[2], e1[2] - e1[0], e1[0] - e1[1]]) / den
        e3 = (e1 * sum(e1) - dot(e1, e1)) / den

    else:
        den = sqrt(2 * (1 + e1[0] * e1[1] + e1[1] * e1[2] - e1[2] * e1[0]))
        e2 = array([e1[1] + e1[2], e1[2] - e1[0], - e1[0] - e1[1]]) / den
        Sum = e1[0] - e1[1] + e1[2]
        Sq = dot(e1, e1)
        e3 = array([e1[0] * Sum - Sq, e1[1] * Sum + Sq, e1[2] * Sum - Sq]) / den

    return vstack([e1, e2, e3])


def rotate_tensors(Q, R):

    Q[2:5] = dot(R, Q[2:5])

    A = Q[5:14].reshape([3,3])
    A_ = dot(R, dot(A, R.T))
    Q[5:14] = A_.ravel()

    if THERMAL:
        Q[14:17] = dot(R, Q[14:17])
