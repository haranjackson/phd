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

function coeffs4(ret, N, w1,w2,w3,w4, M1,M2,M3,M4, λs,λc, Σ, ε, r)
  # Calculate coefficients of basis polynomials and weights
  A_ldiv_B!(M1, w1)
  σ1 = dot(w1, Σ * w1)
  o1 = λs / (abs(σ1) + ε)^r
  A_ldiv_B!(M2, w2)
  σ2 = dot(w2, Σ * w2)
  o2 = λs / (abs(σ2) + ε)^r
  A_ldiv_B!(M3, w3)
  σ3 = dot(w3, Σ * w3)
  o3 = λc / (abs(σ3) + ε)^r
  A_ldiv_B!(M4, w4)
  σ4 = dot(w4, Σ * w4)
  o4 = λc / (abs(σ4) + ε)^r
  oSum = o1 + o2 + o3 + o4
  for i = 1:N+1
    ret[i] = (o1*w1[i] + o2*w2[i] + o3*w3[i] + o4*w4[i]) / oSum
  end
end

function coeffs3(ret, N, w1,w2,w3, M1,M2,M3, λs,λc, Σ, ε, r)
  # Calculate coefficients of basis polynomials and weights
  A_ldiv_B!(M1, w1)
  σ1 = dot(w1, Σ * w1)
  o1 = λs / (abs(σ1) + ε)^r
  A_ldiv_B!(M2, w2)
  σ2 = dot(w2, Σ * w2)
  o2 = λs / (abs(σ2) + ε)^r
  A_ldiv_B!(M3, w3)
  σ3 = dot(w3, Σ * w3)
  o3 = λc / (abs(σ3) + ε)^r
  oSum = o1 + o2 + o3
  for i = 1:N+1
    ret[i] = (o1*w1[i] + o2*w2[i] + o3*w3[i]) / oSum
  end
end

function weno(u, N, λc=1e5, λs=1, r=8, ε=1e-14)
  # Find reconstruction coefficients of u to order N+1
  nx, ny, nz, nvar = size(u)
  fHalfN = Int(floor(N/2))
  cHalfN = Int(ceil(N/2))
  n = Bool(N%2)

  Mlist = coefficient_matrices(N, n)
  Σ = oscillation_indicator(N)
  M1 = factorize(Mlist[1])
  M2 = factorize(Mlist[2])
  M3 = factorize(Mlist[3])
  M4 = factorize(Mlist[4])

  Wx = zeros(nx, ny, nz, N+1, nvar)
  Wx0 = extend(u, N, nx, ny, nz, nvar, 1)
  w1 = zeros(N+1)
  w2 = zeros(N+1)
  w3 = zeros(N+1)
  w3 = zeros(N+1)

  for v=1:nvar, k=1:nz, j=1:ny
    w0 = view(Wx0, :, j, k, v)
    for i=1:nx
      for ind = 0:N
        w1[ind+1] = w0[i+ind]
        w2[ind+1] = w0[i+N+ind]
        w3[ind+1] = w0[i+cHalfN+ind]
      end
      if n
        for ind = 0:N
          w4[ind+1] = w0[i+fHalfN+ind]
        end
        coeffs4(view(Wx,i,j,k,:,v), N, w1,w2,w3,w4, M1,M2,M3,M4, λs,λc, Σ,ε,r)
      else
        coeffs3(view(Wx,i,j,k,:,v), N, w1,w2,w3, M1,M2,M3, λs,λc, Σ,ε,r)
      end
    end
  end
  if ny==1 && nz==1
    return Wx
  end

  """ REQUIRES MODIFICATION TO FORM ABOVE

  Wxy = zeros(nx, ny, nz, N+1, N+1, nvar)
  Wxy0 = extend(Wx, N, nx, ny, nz, nvar, 2)
  for v=1:nvar, a=1:N+1, k=1:nz, j=1:ny, i=1:nx
    jj = j + N
    if n==3
      w1 = Wxy0[i, jj-fHalfN : jj+fHalfN, k, a, v]
      w2 = Wxy0[i, jj-N : jj,             k, a, v]
      w3 = Wxy0[i, jj : jj+N,             k, a, v]
      Wxy[i, j, k, a, :, v] = coeffs([w1,w2,w3], N, n, Mlist, λlist, Σ, ε, r)
    else
      w1 = Wxy0[i, jj-fHalfN : jj+cHalfN, k, a, v]
      w2 = Wxy0[i, jj-cHalfN : jj+fHalfN, k, a, v]
      w3 = Wxy0[i, jj-N : jj,             k, a, v]
      w4 = Wxy0[i, jj : jj+N,             k, a, v]
      Wxy[i, j, k, a, :, v] = coeffs([w1,w2,w3,w4], N, n, Mlist, λlist, Σ, ε, r)
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
      Wxyz[i, j, k, a, b, :, v] = coeffs([w1,w2,w3], N, n, Mlist, λlist, Σ, ε, r)
    else
      w1 = Wxyz0[i, j, kk-fHalfN : kk+cHalfN, a, b, v]
      w2 = Wxyz0[i, j, kk-cHalfN : kk+fHalfN, a, b, v]
      w3 = Wxyz0[i, j, kk-N : kk,             a, b, v]
      w4 = Wxyz0[i, j, kk : kk+N,             a, b, v]
      Wxyz[i, j, k, a, b, :, v] = coeffs([w1,w2,w3,w4], N, n, Mlist, λlist, Σ, ε, r)
    end
  end
  return Wxyz

  """

end



u = rand(91,1,1,18)
u[:,1,1,1] = sin(1:0.1:10)
u[:,1,1,2] = cos(1:0.1:10)
@time weno(u,2);
@time weno(u,2);
wh = @time weno(u,2);
