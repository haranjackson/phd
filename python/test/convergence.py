from numpy import amax, inf, log, mean, prod, sqrt, sum, take


def convergence_rate(NDIM, exact1, exact2, results1, results2, L=2, var=0):

    if results1.ndim > NDIM:
        diff1 = take(results1 - exact1, var, axis=-1)
        diff2 = take(results2 - exact2, var, axis=-1)
    else:
        diff1 = results1 - exact1
        diff2 = results2 - exact2

    n1 = prod(diff1.shape[:NDIM])
    n2 = prod(diff2.shape[:NDIM])

    if L == 1:
        ε1 = sum(abs(diff1)) / n1
        ε2 = sum(abs(diff2)) / n2
    elif L == 2:
        ε1 = sqrt(sum(diff1**2) / n1)
        ε2 = sqrt(sum(diff2**2) / n2)
    elif L == inf:
        ε1 = amax(abs(diff1))
        ε2 = amax(abs(diff2))

    print(ε1)
    print(ε2)
    return log((ε2) / (ε1)) / log(n1 / n2)


def reduce(u, n):
    """ averages cell values of u to produce array of length n
    """
    N = int(len(u) / n)
    return mean(u.reshape([n, N, -1]), axis=1)
