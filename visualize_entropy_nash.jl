using Plots
include("game_solvers/entropy_nash_solver.jl")

function modify_u(u, w, phi)
    w_phi = sum(w[i]*phi[i] for i in eachindex(w))
    u_tilde = u + w_phi * u
    return u_tilde
end

function expectedValue(w)
    u_tilde = modify_u(u, w, phi)

    solver = EntropySolver()
    x, y, proper_termination, max_iter = solve_entropy_nash(solver, u_tilde, actions)
    return x' * V * y
end

# evaluate effect of w on E[V]
V = [3 2; 2 1]
phi = [[0 1; 1 0], [0 1; 0 0], [0 0; 1 0]]

gr()

w1 = range(0, stop=1, length=1000)
w2 = range(0, stop=1, length=1000)
f(w1, w2) = expectedValue([w1, w2])
p = surface(w1, w2, f)
display(plot(p))