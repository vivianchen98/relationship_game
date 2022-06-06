using JuMP, Gurobi, MathOptInterface
using LinearAlgebra
using GameTheory
using TensorGames

include("trafficN.jl")

# order and design
function order_and_design(N, A, u, V_matrix, phi, k)

    MAX = maximum(V)
    
    while !all(x->x==(MAX+1), V)
        a = convert(Tuple, argmin(V)); a = [i for i in a]

        print("\n****************** Checking $a ******************\n")

       # call design
        @time found, w, z, obj_val = design(N, A, u, a, phi, k)

        if found
            return w
            # compute resulting nash sol
            # z_phi = sum(z[i]*phi[i] for i in eachindex(z)) 
            # w_phi = sum(w[i]*phi[i] for i in eachindex(w)) 
            # u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]
            # sol = compute_equilibrium(u_tilde); @show sol.x

            # # determine if the nash equals expected a
            # correct_nash = true
            # for i in 1:N
            #     if sol.x[i][a[i]] <= 0.99
            #         correct_nash = false
            #         break
            #     end
            # end

            # # if nash correct, return w; elsewise inspect next best a
            # if correct_nash
            #     return w
            # else
            #     println("\n w=($w) does not lead to a=($a)! Move to next best a.")
            #     V[a...] = MAX + 1
            # end
        else
            println("w not found! Move to next best a.")
            V[a...] = MAX + 1
        end
        break
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
    total_num_constraints = num_constraints(model, AffExpr, MOI.GreaterThan{Float64}) + num_constraints(model, AffExpr, MOI.LessThan{Float64})
    
    optimize!(model)

    # show result
    print("\n------ RESULT -------\n")
    @show total_num_constraints
    found = termination_status(model)== MathOptInterface.OPTIMAL; @show found
    obj_val = objective_value(model); @show obj_val
    w = value.(w); @show w
    z = value.(z); @show z

    # if found
    #     obj_val = objective_value(model); @show obj_val
    #     w = value.(w); @show w
    #     z = value.(z); @show z
    # else
    #     w, obj_val = zeros(length(phi)), 0
    # end
    # @show total_num_constraints
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

# N-player (<=3), M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    N = N
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]
    if N == 2
        phi = [[0 1; 0 0],[0 0; 1 0]]
    elseif N == 3
        phi = [[0 1 0; 0 0 0; 0 0 0], [0 0 1; 0 0 0; 0 0 0], [0 0 0; 1 0 0; 0 0 0], [0 0 0; 0 0 1; 0 0 0], [0 0 0; 0 0 0; 1 0 0], [0 0 0; 0 0 0; 0 1 0]]
    end
    V = zeros([M for i in 1:N]...)
    if N == 2
        V[1,1] = -1
    elseif N == 3
        V[1,1,1] = -1
    end
    # V = sum(u[i] for i in 1:N)
    @show V
    (; name=name, u=u, A=A, phi=phi, V=V)
end

# call order_and_design
N, M = 3, 2
name, u, A, phi, V = playerN_trafficM(N, M)
k = length(phi)

# design weights
w = order_and_design(N, A, u, V, phi, k)

# VERIFICATION
println()
print("\n****** SUMMARY ******\n")
@show N
@show M
@show V
@show argmin(V)

print("\n------ Orignal Game Sol -------\n")
# g = NormalFormGame([Player(u[i]) for i in 1:N]...)
# @show pure_nash(g)
@show compute_equilibrium(u).x

print("\n------ Modifed Game Sol -------\n")
@show w
w_phi = sum(w[i]*phi[i] for i in eachindex(w)) 
u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]
# g_tilde = NormalFormGame([Player(u_tilde[i]) for i in 1:N]...)
# @show pure_nash(g_tilde)
@show compute_equilibrium(u_tilde).x