# Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
include("weno_matrices.jl")

function extend(inarray, N, nx, ny, nz, nvar, d)
  # Extends the input array by M cells on each surface
  if d==1
    ret = zeros(nx+2N, ny, nz, nvar)
    ret[N+1:nx+N, :, :, :] = inarray
    for i = 1:N
      ret[i, :, :, :] = inarray[1, :, :, :]
      ret[nx+N+i, :, :, :] = inarray[nx, :, :, :]
    end
    return ret

  elseif d==2
    ret = zeros(nx, ny+2N, nz, N+1, nvar)
    ret[:, N+1:ny+N, :, :, :] = inarray
    for i = 1:N
      ret[:, i, :, :] = inarray[:, 1, :, :]
      ret[:, ny+N+i, :, :] = inarray[:, ny, :, :]
    end
    return ret

  else
    ret = zeros(nx, ny, nz+2N, N+1, N+1, nvar)
    ret[:, :, N+1:nz+N, :, :, :] = inarray
    for i = 1:N
      ret[:, :, i, :] = inarray[:, :, 1, :]
      ret[:, :, nz+N+i, :] = inarray[:, :, nz, :]
    end
    return ret
  end
end

function coeffs(w0list, N, nvar, n, Mlist, λlist, Σ, ε, r)
  # Calculate coefficients of basis polynomials and weights
  wlist = [Mlist[i,:,:] \ w0list[i] for i = 1:n]
  σlist = [diag(w' * Σ * w) for w in wlist]
  olist = [λlist[i]  ./ (abs(σlist[i]) + ε).^r for i = 1:n]
  oSum = zeros(nvar)
  num = zeros(N+1, nvar)
  for i = 1:n
    oSum += olist[i]
    for j = 1:N+1
      num[j,:] += wlist[i][j,:] .* olist[i]
    end
  end
  for j = 1:N+1
    num[j,:] ./= oSum
  end
  return num
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
  for i = 1:nx
    for j = 1:ny
      for k = 1:nz
        ii = i + N
        if n==3
          w1 = Wx0[ii-fHalfN : ii+fHalfN, j, k, :]
          w2 = Wx0[ii-N : ii,             j, k, :]
          w3 = Wx0[ii : ii+N,             j, k, :]
          Wx[i, j, k, :, :] = coeffs([w1, w2, w3], N, nvar, n, Mlist, λlist, Σ, ε, r)
        else
          w1 = Wx0[ii-fHalfN : ii+cHalfN, j, k, :]
          w2 = Wx0[ii-cHalfN : ii+fHalfN, j, k, :]
          w3 = Wx0[ii-N : ii,             j, k, :]
          w4 = Wx0[ii : ii+N,             j, k, :]
          Wx[i, j, k, :, :] = coeffs([w1, w2, w3, w4], N, nvar, n, Mlist, λlist, Σ, ε, r)
        end
      end
    end
  end
  if ny==1 && nz==1
    return Wx
  end

  Wxy = zeros(nx, ny, nz, N+1, N+1, nvar)
  Wxy0 = extend(Wx, N, nx, ny, nz, nvar, 2)
  for i = 1:nx
    for j = 1:ny
      for k = 1:nz
        jj = j + N
        for a = 1:N+1
          if n==3
            w1 = Wxy0[i, jj-fHalfN : jj+fHalfN, k, a, :]
            w2 = Wxy0[i, jj-N : jj,             k, a, :]
            w3 = Wxy0[i, jj : jj+N,             k, a, :]
            Wxy[i, j, k, a, :, :] = coeffs([w1, w2, w3], N, nvar, n, Mlist, λlist, Σ, ε, r)
          else
            w1 = Wxy0[i, jj-fHalfN : jj+cHalfN, k, a, :]
            w2 = Wxy0[i, jj-cHalfN : jj+fHalfN, k, a, :]
            w3 = Wxy0[i, jj-N : jj,             k, a, :]
            w4 = Wxy0[i, jj : jj+N,             k, a, :]
            Wxy[i, j, k, a, :, :] = coeffs([w1, w2, w3, w4], N, nvar, n, Mlist, λlist, Σ, ε, r)
          end
        end
      end
    end
  end
  if nz==1
    return Wxy
  end

  Wxyz = zeros(nx, ny, nz, N+1, N+1, N+1, nvar)
  Wxyz0 = extend(Wxy, N, nx, ny, nz, nvar, 3)
  for i in 1:nx
    for j = 1:ny
      for k = 1:nz
        kk = k + N
        for a = 1:N+1
          for b = 1:N+1
            if n==3
              w1 = Wxyz0[i, j, kk-fHalfN : kk+fHalfN, a, b, :]
              w2 = Wxyz0[i, j, kk-N : kk,             a, b, :]
              w3 = Wxyz0[i, j, kk : kk+N,             a, b, :]
              Wxyz[i, j, k, a, b, :, :] = coeffs([w1, w2, w3], N, nvar, n, Mlist, λlist, Σ, ε, r)
            else
              w1 = Wxyz0[i, j, kk-fHalfN : kk+cHalfN, a, b, :]
              w2 = Wxyz0[i, j, kk-cHalfN : kk+fHalfN, a, b, :]
              w3 = Wxyz0[i, j, kk-N : kk,             a, b, :]
              w4 = Wxyz0[i, j, kk : kk+N,             a, b, :]
              Wxyz[i, j, k, a, b, :, :] = coeffs([w1, w2, w3, w4], N, nvar, n, Mlist, λlist, Σ, ε, r)
            end
          end
        end
      end
    end
  end
  return Wxyz
end
