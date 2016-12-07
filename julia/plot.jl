import PyPlot: plot
include("weno_matrices.jl")

function plot_weno(wh, var, N)
  n = size(wh)[1]
  x = float(repeat(0:n-1, inner=N+1))
  for i = 1:n
    x[1+(i-1)*(N+1) : i*(N+1)] += nodes(N)
  end
  y = zeros(n*(N+1))
  for i = 1:n
    for j = 1:N+1
      y[(i-1)*(N+1)+j] = wh[i, 1, 1, j, var]
    end
  end

  plot(x,y)
  plot(x,y,marker='x')
end
