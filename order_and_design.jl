using JuMP
using Gurobi
using LinearAlgebra
using MathOptInterface
include("julia_solvers/bimatrix_mixed_nash_solver.jl")

# order and design
function order_and_design(N, A, u, V, phi, k)
    # order
    MIN = minimum(V)

    while !all(x->x==(MIN-1), V)
        a = convert(Tuple, argmax(V))
        a = [i for i in a]
        print("\n****** Checking $a ******\n")

       # call design
        found, w, V = design(N, A, u, a, phi, k)

        if found
            return w
        else
            print("w not found! move to next best a")
            V[a] = MIN - 1
        end
    end

    return false
end

function design(N, A, u, a, phi, k)
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # variable: w, eps
    @variable(model, w[1:length(phi)])
    @variable(model, z[1:length(phi)])

    # matrix computation: u_tilde = w^T phi u
    w_phi = sum(w[i]*phi[i] for i in eachindex(w))
    # u_tilde = u + [w_phi * u[i] for i in eachindex(u)]
    u_tilde = u + w_phi * u

    # constraints
    A_minus_a = deepcopy(A)
    for i = 1:N
        A_minus_a[i] = deleteat!(A[i], findall(x->x==a[i], A[i]))
    end

    # 1) to guarantee a Nash sol
    for i = 1:N
        for a_minus in A_minus_a[i]
            a_ = deepcopy(a)
            a_[i] = a_minus
            @constraint(model, u_tilde[i][CartesianIndex(Tuple(a))] <= u_tilde[i][CartesianIndex(Tuple(a_))])
        end
    end

    # 2) sparsity constraint for w
    for i in eachindex(w)
        @constraint(model, w[i] <= z[i])
        @constraint(model, w[i] >= -z[i])
    end
    @constraint(model, sum(z[i] for i in eachindex(z)) <= k)

    # objective: min ||w||_1
    @objective(model, Min, sum(z[i] for i in eachindex(z)))

    # print and solve
    print("\n------ MODEL -------\n")
    print(model)
    optimize!(model)

    # show result
    print("\n------ RESULT -------\n")
    @assert termination_status(model)== MathOptInterface.OPTIMAL
    @show objective_value(model)
    @show value.(w)
    @show value.(z)
    
    (; found = (termination_status(model)== MathOptInterface.OPTIMAL), w = value.(w), V = objective_value(model))
end

# Given problem
N = 2
A = [[1,2], [1,2]]
u = [[2 4; 1.5 3], [2 1.5; 4 3]]
V = [3 2; 2 1]
phi = [[0 1; 1 0], [0 1; 0 0], [0 0; 1 0]]
k = length(phi)

# call order_and_design
w = order_and_design(N, A, u, V, phi, k)

w = [1.5, 0, 0]
# VERIFICATION
print("\n****** VERIFICATION ******\n")
print("\n------ Orignal Game Sol -------\n")
x, y, info, _ = solve_mixed_nash(u[1], u[2])
println("player 1", x)
println("player 2", y)

print("\n------ New Game Sol -------\n")
w_phi = sum(w[i]*phi[i] for i in eachindex(w))
# u_tilde = u + [w_phi * u[i] for i in eachindex(u)]
u_tilde = u + w_phi * u

x, y, info, _ = solve_mixed_nash(u_tilde[1], u_tilde[2])
println("player 1", x)
println("player 2", y)

println(u_tilde)