from numpy import amax, inf, log, sqrt, sum, zeros


def convergence_rate(exact1, exact2, results1, results2, L=2, var=0):

    diff1 = (results1 - exact1)[:,:,:,var]
    diff2 = (results2 - exact2)[:,:,:,var]

    nx1,ny1,nz1 = diff1.shape[:3]
    nx2,ny2,nz2 = diff2.shape[:3]

    n1 = nx1*ny1*nz1
    n2 = nx2*ny2*nz2

    if L==1:
        ε1 = sum(abs(diff1)) / n1
        ε2 = sum(abs(diff2)) / n2
    elif L==2:
        ε1 = sqrt(sum(diff1**2) / n1)
        ε2 = sqrt(sum(diff2**2) / n2)
    elif L==inf:
        ε1 = amax(abs(diff1))
        ε2 = amax(abs(diff2))

    print(ε1)
    print(ε2)
    return log((ε2)/(ε1)) / log(nx1/nx2)

def resize_array(arr,n):
    # reduces size of arr along dimension 0 to length n
    n0 = len(arr)
    q = int(n0/n)
    shape = [n] + list(arr.shape[1:])
    shape2 = [1] + list(arr.shape[1:])
    ret = zeros(shape)
    for i in range(n):
        tmp = zeros(shape2)
        for j in range(q):
            tmp += arr[i*q+j]
        ret[i] = tmp/q
    return ret
