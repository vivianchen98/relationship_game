using JuMP
using Gurobi
using LinearAlgebra
using MathOptInterface
# using BenchmarkTools
include("game_solvers/bimatrix_mixed_nash_solver.jl")
include("trafficN.jl")

# order and design
function order_and_design(N, A, u, V_matrix, phi, k)
    # order
    MIN = minimum(V)
    
    while !all(x->x==(MIN-1), V)
        a = convert(Tuple, argmax(V))
        a = [i for i in a]
        print("\n****************** Checking $a ******************\n")

       # call design
        @time found, w, z, obj_val = design(N, A, u, a, phi, k)

        if found
            z_phi = sum(z[i]*phi[i] for i in eachindex(z)); u_tilde = u + z_phi * u

            solver = LemkeHowsonGameSolver()
            x, y, info, _ = solve_mixed_nash(solver, u_tilde[1], u_tilde[2])
            println("player 1", x)
            println("player 2", y)
            
            


            if x[a[1]] == 1 && y[a[2]] == 1 # Nash sol = a
                return w
            else
                println("\n z=($z) does not lead to a=($a)! Move to next best a.")
                V[a...] = MIN - 1
            end
        else
            println("w not found! Move to next best a.")
            V[a...] = MIN - 1
        end
        # break
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
    u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]
    # u_tilde = u + w_phi * u


    # constraints
    A_minus_a = deepcopy(A)
    for i = 1:N
        A_minus_a[i] = deleteat!(A_minus_a[i], findall(x->x==a[i], A_minus_a[i]))
    end

    # 1) to guarantee a Nash sol
    for i = 1:N
        for a_minus in A_minus_a[i]
            a_ = deepcopy(a)
            a_[i] = a_minus
            @constraint(model, u_tilde[i][a...] <= u_tilde[i][a_...])
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
    # print(latex_formulation(model))
    total_num_constraints = num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) + num_constraints(model, AffExpr, MOI.LessThan{Float64})
    optimize!(model)

    # show result
    print("\n------ RESULT -------\n")
    w, obj_val = zeros(length(phi)), 0
    found = termination_status(model)== MathOptInterface.OPTIMAL; @show found
    if found
        obj_val = objective_value(model); @show obj_val
        w = value.(w); @show w
        z = value.(z); @show z
    end
    @show total_num_constraints
    # @assert false

    (; found, w, z, obj_val)
end

# ************* EXAMPLES *************
# prisoner's dilemma
function prisoner()
    name = "prisoner"
    u = [[1 3; 0 2],[1 0; 3 2]]
    A = [[1,2] for i in 1:2]
    phi = [[0 1; 1 0],[0 1; 0 0]]
    V = [0 0; 1 0]
    (; name=name, N=2, u=u, A=A, phi=phi, V=V)
end 

# M-route traffic
function trafficM(M)
    name = "traffic" * string(M)
    u = generate_traffic(2, [M,M])
    A = [[i for i in 1:M] for N=1:2]
    phi = [[0 1; 0 0],[0 0; 1 0]]
    V = zeros(M,M); V[2,1] = 1; V[1,1] = -1
    (; name=name, N=2, u=u, A=A, phi=phi, V=V)
end

function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    N = N
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]
    phi = [[0 1 0; 1 0 0; 0 0 0], [0 0 0; 0 0 1; 0 1 0], [0 0 1; 0 0 0; 1 0 0]]
    V = zeros([M for i in 1:N]...); V[1,1,1] = 1
    (; name=name, N=N, u=u, A=A, phi=phi, V=V)
end

# call order_and_design
name, N, u, A, phi, V = playerN_trafficM(3, 2)
k = length(phi)
w = order_and_design(N, A, u, V, phi, k)

# VERIFICATION
# print("\n****** VERIFICATION ******\n")
# print("\n------ Orignal Game Sol -------\n")
# solver = LemkeHowsonGameSolver()
# x, y, info, _ = solve_mixed_nash(solver, u[1], u[2])
# println("player 1", x)
# println("player 2", y)

# print("\n------ New Game Sol -------\n")
# w_phi = sum(w[i]*phi[i] for i in eachindex(w))
# u_tilde = u + w_phi * u

# x, y, info, _ = solve_mixed_nash(solver, u_tilde[1], u_tilde[2])
# println("player 1", x)
# println("player 2", y)