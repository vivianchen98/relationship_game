include("game_solvers/entropy_nash_solver_general.jl")
include("gradient_entropy_nash_general.jl")
include("trafficN.jl")
using TensorGames, Plots

# N-player, M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    N = N
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

    (; name=name, N=N, u=u, A=A, phi=phi_individual, V=V)
end

# create traffic example
# name, N, u, A, phi, V = playerN_trafficM(3,2)
# k = length(phi)

# x, info = solve_entropy_nash_general(EntropySolver(), u)
# @show x
# @show info.total_iter

# w = [0, 1, 0, 1, 0, 1]
# val = evaluate(u, phi, w, V)
# @show val

# grad = gradient(evaluate, u, phi, w, V)[3]
# @show grad
name, N, u, A, phi, V = playerN_trafficM(3,2)
w, w_list, exp_val_list, terminate_step = GradientDescent(playerN_trafficM(3,2), 0.01, 1000)
# @show w
# @show terminate_step

plot(exp_val_list, label="Entropy-Nash GD")
plot!(5 * ones(length(exp_val_list)), label="Optimal")
plot!(10.3 * ones(length(exp_val_list)), label="Nash")
savefig("exp_val_list.png")

print("\n------ Modifed Game Sol -------\n")
w_phi = sum(w[i]*phi[i] for i in eachindex(w)) 
u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]
x_tilde = compute_equilibrium(u_tilde).x
@show x_tilde

a_tilde = [argmax(x_tilde[i]) for i in 1:N]
@show a_tilde