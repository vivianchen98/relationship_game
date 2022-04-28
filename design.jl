using JuMP
using Ipopt
using LinearAlgebra
# using MathOptInterface
include("julia_solvers/bimatrix_mixed_nash_solver.jl")

function design(A1, A2, u, a1, a2, phi, k)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # variable: w, eps
    @variable(model, 0<=w[1:length(phi)]<=1)
    @variable(model, eps>=0)

    # matrix computation: u_tilde = w^T phi u
    w_phi = sum(w[i]*phi[i] for i in eachindex(w))
    u_tilde = u + [w_phi * u[i] for i in eachindex(u)]

    # constraints
    A1_minus_a1 = delete!(A1, a1)
    A2_minus_a2 = delete!(A2, a2)

    # 1) to guarantee a Nash sol
    for a1_minus in A1_minus_a1
        @constraint(model, u_tilde[1][a1,a2] <= u_tilde[1][a1_minus, a2])
    end

    for a2_minus in A2_minus_a2
        @constraint(model, u_tilde[2][a1,a2] <= u_tilde[2][a1, a2_minus])
    end

    #2) to guarantee uniqueness of the Nash sol
    # for a1_minus in A1_minus_a1, a2_minus in A2_minus_a2
    #     @constraint(model, (u_tilde[1][a1_minus, a2] + eps) <= u_tilde[1][a1_minus, a2_minus])
    #     @constraint(model, (u_tilde[2][a1, a2_minus] + eps) <= u_tilde[2][a1_minus, a2_minus])
    # end

    # 3) sparsity constraint for w
    @NLconstraint(model, sum(abs(w[i]) for i in eachindex(w)) <= k)


    # objective: min eps^2 (slack var in uniqueness constraints)
    @objective(model, Min, eps^2)

    # print and solve
    print("\n------ MODEL -------\n")
    print(model)
    optimize!(model)

    # show result
    print("\n------ RESULT -------\n")
    @show termination_status(model)
    @show objective_value(model)
    @show value.(w)
    @show value.(eps)
    
    (; w = value.(w), eps = value.(eps), V = objective_value(model))
end

# given problem
A1 = Set([1, 2])
A2 = Set([1, 2])
u = [[2 4; 1.5 3], [2 1.5; 4 3]]
(a1, a2) = (1,2) # C,C
phi = [[0 1; 1 0], [0 1; 0 0], [0 0; 1 0]]
k = 2

# design the weight w for the given problem
w, eps, V = design(A1, A2, u, a1, a2, phi, k)

# VERIFICATION
print("\n------ Orignal Game Sol -------\n")
x, y, info, _ = solve_mixed_nash(u[1], u[2])
println("player 1", x)
println("player 2", y)

print("\n------ New Game Sol -------\n")
w_phi = sum(w[i]*phi[i] for i in eachindex(w))
u_tilde = u + [w_phi * u[i] for i in eachindex(u)]

println(u_tilde)

x, y, info, _ = solve_mixed_nash(u_tilde[1], u_tilde[2])
println("player 1", x)
println("player 2", y)