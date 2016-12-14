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

@inline function indicator(chΣT, w, n)
  # The indicator is w' * Σ * w, ie |w' * chol(Σ)'|^2
  # NB The Cholesky decomposition always exists as Σ is SPD
  ret = 0.
  for j=1:n
    tmp = 0.
    for i=j:n
      @inbounds tmp += w[i] * chΣT[i,j]
    end
    ret += tmp^2
  end
  return ret
end

function coeffs3(ret, N, w1,w2,w3, M1,M2,M3, λs,λc, chΣT, ε, r)
  # Calculate coefficients of basis polynomials and weights
  A_ldiv_B!(M1, w1)
  A_ldiv_B!(M2, w2)
  A_ldiv_B!(M3, w3)
  o1 = λs / (indicator(chΣT, w1, N+1) + ε)^r
  o2 = λs / (indicator(chΣT, w2, N+1) + ε)^r
  o3 = λc / (indicator(chΣT, w3, N+1) + ε)^r
  oSum = o1 + o2 + o3
  for i = 1:N+1
    @inbounds ret[i] = (o1*w1[i] + o2*w2[i] + o3*w3[i]) / oSum
  end
end

function coeffs4(ret, N, w1,w2,w3,w4, M1,M2,M3,M4, λs,λc, chΣT, ε, r)
  # Calculate coefficients of basis polynomials and weights
  A_ldiv_B!(M1, w1)
  A_ldiv_B!(M2, w2)
  A_ldiv_B!(M3, w3)
  A_ldiv_B!(M4, w4)
  o1 = λs / (indicator(chΣT, w1, N+1) + ε)^r
  o2 = λs / (indicator(chΣT, w2, N+1) + ε)^r
  o3 = λc / (indicator(chΣT, w3, N+1) + ε)^r
  o4 = λc / (indicator(chΣT, w4, N+1) + ε)^r
  oSum = o1 + o2 + o3 + o4
  for i = 1:N+1
    @inbounds ret[i] = (o1*w1[i] + o2*w2[i] + o3*w3[i] + o4*w4[i]) / oSum
  end
end

function weno(u, N, M1, M2, M3, M4, chΣT, λc=1e5, λs=1, r=8, ε=1e-14)
  # Find reconstruction coefficients of u to order N+1
  nx, ny, nz, nvar = size(u)
  fHalfN = Int(floor(N/2))
  cHalfN = Int(ceil(N/2))
  n = Bool(N%2)

  w1 = Array{Float64}(N+1)
  w2 = Array{Float64}(N+1)
  w3 = Array{Float64}(N+1)
  w4 = Array{Float64}(N+1)

  Wx = zeros(nx, ny, nz, N+1, nvar)
  Wx0 = extend(u, N, nx, ny, nz, nvar, 1)

  for v=1:nvar, k=1:nz, j=1:ny
    w0 = view(Wx0, :, j, k, v)
    for i=1:nx
      for ind = 0:N
        @inbounds w1[ind+1] = w0[i+ind]
        @inbounds w2[ind+1] = w0[i+N+ind]
        @inbounds w3[ind+1] = w0[i+cHalfN+ind]
      end
      if n
        for ind = 0:N
          @inbounds w4[ind+1] = w0[i+fHalfN+ind]
        end
        coeffs4(view(Wx,i,j,k,:,v), N, w1,w2,w3,w4, M1,M2,M3,M4, λs,λc, chΣT,ε,r)
      else
        coeffs3(view(Wx,i,j,k,:,v), N, w1,w2,w3, M1,M2,M3, λs,λc, chΣT,ε,r)
      end
    end
  end
  if ny==1 && nz==1
    return Wx
  end

  Wxy = zeros(nx, ny, nz, N+1, N+1, nvar)
  Wxy0 = extend(Wx, N, nx, ny, nz, nvar, 2)

  for v=1:nvar, a=1:N+1, k=1:nz, i=1:nx
    w0 = view(Wxy0, i, :, k, a, v)
    for j=1:ny
      for ind = 0:N
        @inbounds w1[ind+1] = w0[j+ind]
        @inbounds w2[ind+1] = w0[j+N+ind]
        @inbounds w3[ind+1] = w0[j+cHalfN+ind]
      end
      if n
        for ind = 0:N
          @inbounds w4[ind+1] = w0[j+fHalfN+ind]
        end
        coeffs4(view(Wxy,i,j,k,a,:,v), N, w1,w2,w3,w4, M1,M2,M3,M4, λs,λc, chΣT,ε,r)
      else
        coeffs3(view(Wxy,i,j,k,a,:,v), N, w1,w2,w3, M1,M2,M3, λs,λc, chΣT,ε,r)
      end
    end
  end
  if nz==1
    return Wxy
  end

  Wxyz = zeros(nx, ny, nz, N+1, N+1, N+1, nvar)
  Wxyz0 = extend(Wxy, N, nx, ny, nz, nvar, 3)

  for v=1:nvar, b=1:N+1, a=1:N+1, j=1:ny, i=1:nx
    w0 = view(Wxyz0, i, j, :, a, b, v)
    for k=1:nz
      for ind = 0:N
        @inbounds w1[ind+1] = w0[k+ind]
        @inbounds w2[ind+1] = w0[k+N+ind]
        @inbounds w3[ind+1] = w0[k+cHalfN+ind]
      end
      if n
        for ind = 0:N
          @inbounds w4[ind+1] = w0[k+fHalfN+ind]
        end
        coeffs4(view(Wxyz,i,j,k,a,b,:,v), N, w1,w2,w3,w4, M1,M2,M3,M4, λs,λc, chΣT,ε,r)
      else
        coeffs3(view(Wxyz,i,j,k,a,b,:,v), N, w1,w2,w3, M1,M2,M3, λs,λc, chΣT,ε,r)
      end
    end
  end
  return Wxyz

end

function main(u, N)
  n = Bool(N%2)
  Mlist = coefficient_matrices(N, n)
  Σ = oscillation_indicator(N)
  M1 = factorize(Mlist[1])
  M2 = factorize(Mlist[2])
  M3 = factorize(Mlist[3])
  M4 = factorize(Mlist[4])
  chΣT = chol(Σ)'
  @time weno(u, N, M1, M2, M3, M4, chΣT)
end

function test()
  include("plot.jl")
  u = rand(91,1,1,18)
  u[:,1,1,1] = sin(1:0.1:10)
  u[:,1,1,2] = cos(1:0.1:10)
  test(u,2)
  wh = test(u,2)
  plot_weno(wh,1,2)
end
