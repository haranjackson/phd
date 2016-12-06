import Polynomials: Poly, polyint, polyder

function lagrange(x, w)
  """
  Taken from SciPy 0.18.1:

  Return a Lagrange interpolating polynomial.
  Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
  polynomial through the points ``(x, w)``.
  Warning: This implementation is numerically unstable. Do not expect to
  be able to use more than about 20 points even if they are chosen optimally.
  Parameters
  ----------
  x : array_like
      `x` represents the x-coordinates of a set of datapoints.
  w : array_like
      `w` represents the y-coordinates of a set of datapoints, i.e. f(`x`).
  Returns
  -------
      The Lagrange interpolating polynomial.
  """
  M = size(x)[1]
  p = Poly([0])
  for j in 1:M
    pt = Poly([w[j]])
    for k in 1:M
      if k != j
        fac = x[j]-x[k]
        pt *= Poly([1.0, -x[k]]) / fac
      end
    end
    p += pt
  end
  return p
end

function nodes(N)
  # Returns Legendre-Gauss nodes, scaled to [0,1]
  if N==1
    return [0.5]
  elseif N==2
    return 0.5 * (1 + [-sqrt(1/3), sqrt(1/3)])
  elseif N==3
    tmp = sqrt(3/5)
    return 0.5 * (1 + [-tmp, 0, tmp])
  elseif N==4
    tmp = 2/7 * sqrt(6/5)
    tmp1 = sqrt(3/7 - tmp)
    tmp2 = sqrt(3/7 + tmp)
    return 0.5 * (1 + [-tmp2, -tmp1, tmp1, tmp2])
  elseif N==5
    tmp = 2 * sqrt(10/7)
    tmp1 = 1/3 * sqrt(5-tmp)
    tmp2 = 1/3 * sqrt(5+tmp)
    return 0.5 * (1 + [-tmp2, -tmp1, 0, tmp1, tmp2])
  else
    print("Invalid input")
  end
end

function basis(N)
  # Returns basis polynomials
  nodeArray = nodes(N+1)
  return [lagrange(nodeArray, eye(N+1)[:,i]) for i in 1:N+1]
end

function coefficient_matrices(N, n)
  # Generate linear systems governing the coefficients of the basis polynomials
  fHalfN = floor(N/2)
  cHalfN = ceil(N/2)
  Mlist = zeros(n, N+1, N+1)
  ψ = basis(N)

  for a in 1:N+1
    for p in 1:N+1
      ψintp = polyint(ψ[p])
      if Bool(N%2)
        Mlist[1,a,p] = ψintp(a-fHalfN) - ψintp(a-fHalfN-1)
        Mlist[2,a,p] = ψintp(a-cHalfN) - ψintp(a-cHalfN-1)
        Mlist[3,a,p] = ψintp(a-N)      - ψintp(a-N-1)
        Mlist[4,a,p] = ψintp(a)        - ψintp(a-1)
      else
        Mlist[1,a,p] = ψintp(a-fHalfN) - ψintp(a-fHalfN-1)
        Mlist[2,a,p] = ψintp(a-N)      - ψintp(a-N-1)
        Mlist[3,a,p] = ψintp(a)        - ψintp(a-1)
      end
    end
  end
  return Mlist
end

function oscillation_indicator(N)
  # Generate the oscillation indicator matrix
  Σ = zeros(N+1, N+1)
  ψ = basis(N)

  for a in 1:N
    ψdera = [polyder(ψp, a) for ψp in ψ]
    for p in 1:N+1
      for m in 1:N+1
        antiderivative = polyint(ψdera[p] * ψdera[m])
        Σ[p,m] += antiderivative(1) - antiderivative(0)
      end
    end
  end
  return Σ
end
