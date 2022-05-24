include("gradient_entropy_nash.jl")
include("trafficN.jl")
using Plots

# ************* EXAMPLES *************
# prisoner's dilemma
function prisoner()
    name = "prisoner"
    u = [[1 3; 0 2];;;[1 0; 3 2]]
    phi = [[0 1; 1 0];;;[0 1; 0 0]]
    V = [0 0; 0 -1]
    (; name=name, u=u, phi=phi, V=V)
end 

# M-route traffic
function trafficM(M)
    name = "traffic" * string(M)
    u = generate_traffic(2, [M,M]); u = [u[1];;;u[2]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    V = zeros(M,M); V[1,1] = -1
    (; name=name, u=u, phi=phi, V=V)
end

# ************* DEFINE PROBLEM *************
# game structure
name, u, phi, V = trafficM(3)
print("\n------ Game -------\n")
@show name
@show u
@show phi
@show V

# 2 player relationship weights
# w = [0.8, 0.2]


# ************* SOLUTIONS *************
print("\n------ Orignal Game Sol -------\n")
solver = LemkeHowsonGameSolver()
x, y, info, _ = solve_mixed_nash(solver, u[:,:,1], u[:,:,2])
println("player 1", x)
println("player 2", y)

# print("\n------ New Game Sol -------\n")
# x, y, info = solve_relationship_game(u, phi, w)
# @show x
# @show y

# expected_V = evaluate(u, phi, w, V) # Remember V is defined inside evaluate()!
# @show expected_V

# grad = gradient(evaluate, u, phi, w, V)
# @show grad

# ************* Gradient Descent *************
print("\n****** Gradient Descent ******\n")
w, w_list, exp_val_list, terminate_step = GradientDescent(trafficM(3), 0.01, 10000)

print("\n------ New Game Sol -------\n")
x_entropy, y_entropy, info = solve_relationship_game(u, phi, w)
@show x_entropy
@show y_entropy

m, n, N = size(u)
u_tilde = u + [ (phi[k,:,:] * w)' * u[i,j,:] for i in 1:m, j in 1:n, k in 1:N]
x, y, info, _ = solve_mixed_nash(solver, u_tilde[:,:,1], u_tilde[:,:,2])
@show x
@show y

@show x' * V * y

plot(exp_val_list)
savefig("gradient training_3.png")
