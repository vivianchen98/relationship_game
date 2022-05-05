using Plots
using Zygote, ChainRulesCore
include("game_solvers/entropy_nash_solver.jl")
include("gradient_entropy_nash.jl")
# include("game_solvers/bimatrix_mixed_nash_solver.jl")

# Plot of E[V] on w
function expectedValue(u, phi, w, V)
    m, n, N = size(u)
    u_tilde = u + [ (phi[k,:,:] * w)' * u[i,j,:] for i in 1:m, j in 1:n, k in 1:N]

    solver = EntropySolver()
    x, y, info = solve_entropy_nash(solver, u_tilde, actions)
    # solver = LemkeHowsonGameSolver()
    # x, y, info = solve_mixed_nash(solver, u_tilde[:,:,1], u_tilde[:,:,2])

    return x' * V * y
end

function plot_surface(u, phi, V, output_name)
    w1 = range(0, stop=1, length=10)
    w2 = range(0, stop=1, length=10)
    f(w1, w2) = expectedValue(u, phi, [w1, w2], V)
    s = surface(w1, w2, f, camera=(30,40))
    plot(s)
    savefig("results/$(output_name)_surface.png")
end

# Gradient-Heatmap
function plot_heatmap(u, actions, phi, output_name, c)
    x = range(0, stop=1, length=1000)  # w1
    y = range(0, stop=1, length=1000)  # w2

    f(x, y) = evaluate(u, actions, phi, [x, y])

    fdot(x, y) = gradient(f, x, y)
    ∂xf(x, y) = fdot(x, y)[1]
    ∂yf(x, y) = fdot(x, y)[2]

    xlim = (-0, 1)
    ylim = (-0, 1)
    xs = range(xlim...; length=200)
    ys = range(ylim...; length=200)

    # c = 0.05 # for visual scaling of the quivers (GRADIENTS ARE MASSIVE!!!? WHY :( )
    x = range(-0, 1; length=11)
    y = range(-0, 1; length=11)
    X, Y = reim(complex.(x', y)) # meshgrid
    U, V = c*∂xf.(x', y), c*∂yf.(x', y)

    h = heatmap(xs, ys, f)
    quiver!(vec(X-U/2), vec(Y-V/2); quiver=(vec(U), vec(V)), color=:cyan)
    plot(h; xlim, ylim, xlabel = "w1", ylabel = "w2",size=(450, 400))
    savefig("results/$(output_name)_heatmap.png")

end

# ************* EXAMPLES *************
# prisoner's dilemma
# u = [[1 3; 0 2];;;[1 0; 3 2]]
# actions = [["C", "D"], ["C", "D"]]
# phi = [[0 1; 2 0];;;[0 0; 1 0]]
# V = sum(u[:,:,i] for i in 1:2)

# 2-route traffic
# u = [[2 4; 1.5 3];;;[2 1.5; 4 3]]
# actions = [["A", "B"], ["A", "B"]]
# phi = [[0 1; 2 0];;;[0 0; 1 0]]
# V = sum(u[:,:,i] for i in 1:2)

# 3-route traffic
u = [[2 4 4; 1.5 3 1.5; 2.5 2.5 5];;;[2 1.5 2.5; 4 3 2.5; 4 1.5 5]] 
actions = [["A", "B", "C"], ["A", "B", "C"]]
phi = [[0 1; 2 0];;;[0 0; 1 0]]
V = sum(u[:,:,i] for i in 1:2)

## 3 PLAYER EXAMPLE: Clean or Pollute
# u1 = [[1 1; 0 3];;;[1 4; 3 3]]
# u2 = [[1 0; 1 3];;;[1 3; 4 3]]
# u3 = [[1 1; 1 4];;;[0 3; 3 3]]
# u = [u1, u2, u3]
# d = [2,2,2]; N = 3; cost_tensors = [ randn(d...) for i = 1:N]; # random N-player matrix generation

# ************* PLOTTING *************
# plot_surface(u, phi, V, "traffic3_0.8")
plot_heatmap(u, actions, phi, "traffic3_0.8", 0.05)
