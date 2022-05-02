using Plots
using Zygote, ChainRulesCore
include("game_solvers/entropy_nash_solver.jl")
include("gradient_entropy_nash.jl")

# function modify_u(u, w, phi)
#     w_phi = sum(w[i]*phi[i] for i in eachindex(w))
#     u_tilde = u + w_phi * u
#     return u_tilde
# end

# function expectedValue(w)
#     u_tilde = modify_u(u, w, phi)

#     solver = EntropySolver()
#     x, y, proper_termination, max_iter = solve_entropy_nash(solver, u_tilde, actions)
#     return x' * V * y
# end

# # evaluate effect of w on E[V]
# V = [3 2; 2 1]
# phi = [[0 1; 1 0], [0 1; 0 0], [0 0; 1 0]]
#
# gr()
#
# w1 = range(0, stop=1, length=1000)
# w2 = range(0, stop=1, length=1000)
# f(w1, w2) = expectedValue([w1, w2])
# p = surface(w1, w2, f)
# display(plot(p))


# Gradient-Heatmap
u = [[1 3; 0 2];;;[1 0; 3 2]]
actions = [["C", "D"], ["C", "D"]]
phi = [[0 1; 2 0];;;[0 0; 1 0]]

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

c = 0.0000000002  # for visual scaling of the quivers (GRADIENTS ARE MASSIVE!!!? WHY :( )
x = range(-0, 1; length=11)
y = range(-0, 1; length=11)
X, Y = reim(complex.(x', y)) # meshgrid
U, V = c*∂xf.(x', y), c*∂yf.(x', y)

heatmap(xs, ys, f)
quiver!(vec(X-U/2), vec(Y-V/2); quiver=(vec(U), vec(V)), color=:cyan)
plot!(; xlim, ylim, size=(450, 400))
savefig("result_udpated.png")
