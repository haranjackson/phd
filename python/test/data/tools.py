from numpy import genfromtxt


def from_csv(fname):

    data = genfromtxt(fname, delimiter=',')
    return data[:,0], data[:,1]
