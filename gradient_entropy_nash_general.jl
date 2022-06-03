using Zygote, ChainRulesCore
using LinearAlgebra
include("game_solvers/entropy_nash_solver_general.jl")
# include("game_solvers/bimatrix_mixed_nash_solver.jl")
include("trafficN.jl")

# Given a RG, find its entropy_nash solution
function solve_relationship_game(u, phi, w)
    N = length(u)
    w_phi = w' * phi
    u_tilde = u + [sum(w_phi[n,:][i] * u[i] for i in 1:N) for n in 1:N]

    solver = EntropySolver()
    res = solve_entropy_nash_general(solver, u_tilde)

    return res
end

function evaluate(u, phi, w, V)
    x, info = solve_relationship_game(u, phi, w)
    return x[1]' * h(1, u, x)
end


function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, phi, w)
    res = solve_relationship_game(u, phi, w)

    function solve_relationship_game_pullback(∂res)
        x = res.x
        proper_termination, total_iter, max_iter, λ, N = res.info
        K = size(phi)[3]
        # A, B = u[:,:,1], u[:,:,2]

        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂phi = NoTangent()

        # J_F
        J_submatrix(i,j) = (i == j) ? (zeros((size(u[i])[i], size(u[i])[i]))) : (softmax_jacobian(x[j]) * (-g(i,j,u,x) ./ λ))
        J_softmax = []
        for i in 1:N
            row = []
            for j in 1:length(x)
                if isempty(row)
                    row = J_submatrix(i,j)
                else
                    row = hcat(row, J_submatrix(i,j))
                end
            end
            if isempty(J_softmax)
                J_softmax = row
            else
                J_softmax = vcat(J_softmax, row)
            end
        end
        m, n = size(J_softmax)
        total_actions = sum(size(u[i])[i] for i in 1:N)
        @assert m == total_actions && n == total_actions
        J_F = I(total_actions) - J_softmax

        # J_F_wrt_w
        J_f_wrt_w = [ - (phi[1,:,k]' * [A[i,:]' ; B[i,:]'] * y ./ λ)
                for i in 1:m, k in 1:K]
        J_g_wrt_w = [ - (phi[2,:,k]' * [A[:,i]' ; B[:,i]'] * x ./ λ)
                    for i in 1:n, k in 1:K]
        J_F_wrt_w = - [J_s * J_f_wrt_w; J_u * J_g_wrt_w]

        # ∂w = [∂x; ∂y] - (J_F)^-1 * J_F_wrt_w
        ∂w = ([∂res.x; ∂res.y]' * inv(J_F) * J_F_wrt_w)'

        ∂self, ∂u, ∂phi, ∂w
    end

    res, solve_relationship_game_pullback
end


# Gradient Descent of social cost V on weight vector w
function GradientDescent(g, stepsize, max_iter)
    w_list = Vector{Vector{Float64}}()
    exp_val_list = Vector{Float64}()
    terminate_step = 0

    # init w
    K = size(g.phi)[3]
    w = [0/K for i in 1:K] # unifrom distribution of length K
    push!(w_list, w)
    push!(exp_val_list, evaluate(g.u, g.phi, w, g.V))
    println("start with w=($w)")

    for i in 1:max_iter 
        ∂w = gradient(evaluate, g.u, g.phi, w, g.V)[3]
        w = w + stepsize .* ∂w
        push!(w_list, w)
        push!(exp_val_list, evaluate(g.u, g.phi, w, g.V))
        if i % 1000 == 0
            @show w
            # @show norm(∂w)
            # @show evaluate(g.u, g.phi, w, g.V)
        end
        if norm(∂w) < 0.01 # stopping criteria
        # if evaluate(g.u, g.phi, w, g.V) - (-1) ≤ 0.01
            println("terminate with w=($w) in $(i) steps")
            terminate_step = i
            # @show evaluate(g.u, g.phi, w, g.V)
            break
        end
        if i == max_iter
            println("Does not converge within ($max_iter) iterations")
        end
    end

    return w, w_list, exp_val_list, terminate_step
end    

# phi = [[0 1 0; 0 0 0; 0 0 0];;;[0 0 1; 0 0 0; 0 0 0];;;[0 0 0; 1 0 0; 0 0 0];;;[0 0 0; 0 0 1; 0 0 0];;;[0 0 0; 0 0 0; 1 0 0];;;[0 0 0; 0 0 0; 0 1 0]]
# phi = [[0 1 0; 0 0 0; 0 0 0], [0 0 1; 0 0 0; 0 0 0], [0 0 0; 1 0 0; 0 0 0], [0 0 0; 0 0 1; 0 0 0], [0 0 0; 0 0 0; 1 0 0], [0 0 0; 0 0 0; 0 1 0]]