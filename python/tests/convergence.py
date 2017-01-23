from numpy import amax, inf, log, sqrt, sum


def convergence_rate(exact, results1, results2, n1, n2, L=2):
    diff1 = results1 - exact
    diff2 = results2 - exact
    if L==1:
        ε1 = sum(abs(diff1))
        ε2 = sum(abs(diff2))
    elif L==2:
        ε1 = sqrt(sum(diff1**2))
        ε2 = sqrt(sum(diff2**2))
    elif L==inf:
        ε1 = amax(abs(diff1))
        ε2 = amax(abs(diff2))
    return log(ε2/ε1) / log(n1/n2)
