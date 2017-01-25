from numpy import amax, inf, log, sqrt, sum


def convergence_rate(exact1, exact2, results1, results2, n1, n2, L=2):
    diff1 = results1 - exact1
    diff2 = results2 - exact2

    if L==1:
        ε1 = sum(abs(diff1)) / n1
        ε2 = sum(abs(diff2)) / n2
    elif L==2:
        ε1 = sqrt(sum(diff1**2) / n1)
        ε2 = sqrt(sum(diff2**2) / n2)
    elif L==inf:
        ε1 = amax(abs(diff1))
        ε2 = amax(abs(diff2))

    print(ε1/n1)
    print(ε2/n2)
    return log((ε2/n2)/(ε1/n1)) / log(n1/n2)
