# Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
include("weno_matrices.jl")

function extend(arr, N, nx, ny, nz, nvar, d)
  # Extends the input array by M cells on each surface
  if d==1
    ret = zeros(nx+2N, ny, nz, nvar)
    ret[N+1:nx+N, :, :, :] = arr
    in0 = arr[1, :, :, :]
    in1 = arr[nx, :, :, :]
    for i = 1:N
      ret[i, :, :, :] = in0
      ret[nx+N+i, :, :, :] = in1
    end
    return ret

  elseif d==2
    ret = zeros(nx, ny+2N, nz, N+1, nvar)
    ret[:, N+1:ny+N, :, :, :] = arr
    in0 = arr[:, 1, :, :]
    in1 = arr[:, ny, :, :]
    for i = 1:N
      ret[:, i, :, :] = in0
      ret[:, ny+N+i, :, :] = in1
    end
    return ret

  else
    ret = zeros(nx, ny, nz+2N, N+1, N+1, nvar)
    ret[:, :, N+1:nz+N, :, :, :] = arr
    in0 = arr[:, :, 1, :]
    in1 = arr[:, :, nz, :]
    for i = 1:N
      ret[:, :, i, :] = in0
      ret[:, :, nz+N+i, :] = in1
    end
    return ret
  end
end

function coeffs(w0list, N, n, Mlist, λlist, Σ, ε, r)
  # Calculate coefficients of basis polynomials and weights
  oSum = 0.
  ret = zeros(N+1)
  for i in 1:n
    w = Mlist[i] \ w0list[i]
    σ = w' * Σ * w
    o = λlist[i] / (abs(σ[1]) + ε)^r
    oSum += o
    ret += o * w
  end
  return ret / oSum
end

function weno(u, N, λc=1e5, λs=1, r=8, ε=1e-14)
  # Find reconstruction coefficients of u to order N+1
  nx, ny, nz, nvar = size(u)
  fHalfN = Int(floor(N/2))
  cHalfN = Int(ceil(N/2))

  if Bool(N%2)
    n = 4
    λlist = [λc, λc, λs, λs]
  else
    n = 3
    λlist = [λc, λs, λs]
  end

  Mlist = coefficient_matrices(N, n)
  Σ = oscillation_indicator(N)

  Wx = zeros(nx, ny, nz, N+1, nvar)
  Wx0 = extend(u, N, nx, ny, nz, nvar, 1)
  for v=1:nvar, k=1:nz, j=1:ny, i=1:nx
    ii = i+N
    if n==3
      w1 = view(Wx0, ii-fHalfN : ii+fHalfN, j, k, v)
      w2 = view(Wx0, ii-N : ii,             j, k, v)
      w3 = view(Wx0, ii : ii+N,             j, k, v)
      Wx[i, j, k, :, v] = coeffs([w1, w2, w3], N, n, Mlist, λlist, Σ, ε, r)
    else
      w1 = view(Wx0, ii-fHalfN : ii+cHalfN, j, k, v)
      w2 = view(Wx0, ii-cHalfN : ii+fHalfN, j, k, v)
      w3 = view(Wx0, ii-N : ii,             j, k, v)
      w4 = view(Wx0, ii : ii+N,             j, k, v)
      Wx[i, j, k, :, v] = coeffs([w1, w2, w3, w4], N, n, Mlist, λlist, Σ, ε, r)
    end
  end
  if ny==1 && nz==1
    return Wx
  end

  Wxy = zeros(nx, ny, nz, N+1, N+1, nvar)
  Wxy0 = extend(Wx, N, nx, ny, nz, nvar, 2)
  for v=1:nvar, a=1:N+1, k=1:nz, j=1:ny, i=1:nx
    jj = j + N
    if n==3
      w1 = Wxy0[i, jj-fHalfN : jj+fHalfN, k, a, v]
      w2 = Wxy0[i, jj-N : jj,             k, a, v]
      w3 = Wxy0[i, jj : jj+N,             k, a, v]
      Wxy[i, j, k, a, :, v] = coeffs([w1, w2, w3], N, n, Mlist, λlist, Σ, ε, r)
    else
      w1 = Wxy0[i, jj-fHalfN : jj+cHalfN, k, a, v]
      w2 = Wxy0[i, jj-cHalfN : jj+fHalfN, k, a, v]
      w3 = Wxy0[i, jj-N : jj,             k, a, v]
      w4 = Wxy0[i, jj : jj+N,             k, a, v]
      Wxy[i, j, k, a, :, v] = coeffs([w1, w2, w3, w4], N, n, Mlist, λlist, Σ, ε, r)
    end
  end
  if nz==1
    return Wxy
  end

  Wxyz = zeros(nx, ny, nz, N+1, N+1, N+1, nvar)
  Wxyz0 = extend(Wxy, N, nx, ny, nz, nvar, 3)
  for v=1:nvar, b=1:N+1, a=1:N+1, k=1:nz, j=1:ny, i=1:nx
    kk = k + N
    if n==3
      w1 = Wxyz0[i, j, kk-fHalfN : kk+fHalfN, a, b, v]
      w2 = Wxyz0[i, j, kk-N : kk,             a, b, v]
      w3 = Wxyz0[i, j, kk : kk+N,             a, b, v]
      Wxyz[i, j, k, a, b, :, v] = coeffs([w1, w2, w3], N, n, Mlist, λlist, Σ, ε, r)
    else
      w1 = Wxyz0[i, j, kk-fHalfN : kk+cHalfN, a, b, v]
      w2 = Wxyz0[i, j, kk-cHalfN : kk+fHalfN, a, b, v]
      w3 = Wxyz0[i, j, kk-N : kk,             a, b, v]
      w4 = Wxyz0[i, j, kk : kk+N,             a, b, v]
      Wxyz[i, j, k, a, b, :, v] = coeffs([w1, w2, w3, w4], N, n, Mlist, λlist, Σ, ε, r)
    end
  end
  return Wxyz
end

u = rand(100,1,1,18)
@time weno(u,2);
@time weno(u,2);
wh = @time weno(u,2);
