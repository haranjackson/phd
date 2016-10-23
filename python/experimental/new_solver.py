from numpy import array
from numpy.polynomial.legendre import leggauss


nodes, weights = leggauss(2)
nodes += 1
nodes /= 2
weights /= 2


c1 = nodes[0]
c2 = nodes[1]

M = 3 * array([[c2**2, -c1*c2, -c1*c2, c1**2],
               [-c2,       c2,     c1,   -c1],
               [-c2,       c1,     c2,   -c1],
               [  1,       -1,     -1,     1]])
