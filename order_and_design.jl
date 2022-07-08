using JuMP, Gurobi, MathOptInterface
using LinearAlgebra
# using GameTheory
using TensorGames
using ArgParse
include("trafficN.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--N"
            help = "number of players"
            arg_type = Int64
            default = 2
        "--M"
            help = "number of actions"
            arg_type = Int64
            default = 2
    end

    return parse_args(s)
end

args = parse_commandline()

# order and design
function order_and_design(N, A, u, V_matrix, phi, k)
    total_constr = 0

    MAX = maximum(V)
    
    while !all(x->x==(MAX+1), V)
        a = convert(Tuple, argmin(V)); a = [i for i in a]

        print("\n****************** Checking $a ******************\n")

       # call design
        @time found, w, z, obj_val, num_constr = design(N, A, u, a, phi, k)
        total_constr += num_constr

        if found
            # compute resulting nash sol
            w_phi = sum(w[i]*phi[i] for i in eachindex(w)) 
            u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]
            x_tilde = compute_equilibrium(u_tilde).x
            @show x_tilde

            # determine if the a_tilde ~ a
            a_tilde = [argmax(x_tilde[i]) for i in 1:N]
            if a_tilde == a
                return w
                break
            else
                println("\n w=($w) does not lead to a=($a)! Move to next best a.")
                V[a...] = MAX + 1
            end
        else
            println("w not found! Move to next best a.")
            V[a...] = MAX + 1
        end
        # break
    end
    println("total_constr= $(total_constr)")
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
    if found
        obj_val = objective_value(model); @show obj_val
        w = value.(w); @show w
        z = value.(z); @show z
    else
        w, obj_val = zeros(length(phi)), 0
    end

    (; found, w, z, obj_val, total_num_constraints)
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
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]

    # phi_individual
    phi_individual = Matrix{Int64}[]
    for i in 1:N, j in 1:N
        phi = zeros(Int64, N,N)
        if i != j
            phi[i,j] = 1
            push!(phi_individual, phi)
        end
    end
    # phi_individual = phi_individual[1:3]

    # phi_all_people
    phi_all_people = Matrix{Int64}[]
    for i in 1:N
        phi = zeros(Int64, N,N)
        for j in 1:N
            if i != j
                phi[i,j] = 1
            end
        end
        push!(phi_all_people, phi)
    end

    # phi_reciprocity
    phi_reciprocity = Matrix{Int64}[]
    for i in 1:N, j in 1:N
        phi = zeros(Int64, N,N)
        if i != j
            phi[i,j] = 1
            phi[j,i] = 1
            if !(phi in phi_reciprocity)    
                push!(phi_reciprocity, phi)
            end
        end
    end


    # V
    V = sum(u[i] for i in 1:N)

    (; name=name, u=u, A=A, phi=phi_all_people[1:3], V=V)
end

# call order_and_design
N, M = args["N"], args["M"]
name, u, A, phi, V = playerN_trafficM(N, M)
k = length(phi)

# design weights
@time w = order_and_design(N, A, u, V, phi, k)

# VERIFICATION
println()
print("\n****** SUMMARY ******\n")
@show N
@show M
@show V
V_min_a = argmin(V)
@show V_min_a

print("\n------ Orignal Game Sol -------\n")
x = compute_equilibrium(u).x
@show x
a = [argmax(x[i]) for i in 1:N]
@show a 

print("\n------ Modifed Game Sol -------\n")
@show w
w_phi = sum(w[i]*phi[i] for i in eachindex(w)) 
u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]
x_tilde = compute_equilibrium(u_tilde).x
@show x_tilde

a_tilde = [argmax(x_tilde[i]) for i in 1:N]
@show a_tilde

# open("order_and_design_output/player_$(N)_action_$(M)_$(V_min_a).txt", "w") do file
#     println(file, "N=$(N), M=$(M)")
#     println(file, "V: ", V)
#     println(file, "a: ", V_min_a)
#     println(file, "x: ", x)
#     println(file, "a: ", a)
#     println(file, "w: ", w)
#     println(file, "x_tilde: ", x_tilde)
#     println(file, "a_tilde: ", a_tilde)
# end