using Zygote, ChainRulesCore
using LinearAlgebra
include("game_solvers/entropy_nash_solver.jl")
include("game_solvers/bimatrix_mixed_nash_solver.jl")
include("trafficN.jl")

# Given a RG, find its entropy_nash solution
function solve_relationship_game(u, phi, w)
    # m, n, N = size(u)
    # u_tilde = u + [ (phi[k,:,:] * w)' * u[i,j,:] for i in 1:m, j in 1:n, k in 1:N]
    create_u_tilde(u, phi, w)
    solver = EntropySolver()
    res = solve_entropy_nash(solver, u_tilde)
    return res
end

function evaluate(u, phi, w, V, gamma = 1)
    x, y, info = solve_relationship_game(u, phi, w)
    return x' * V * y + gamma * norm(w, 1)
end


function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, phi, w)
    res = solve_relationship_game(u, phi, w)

    function solve_relationship_game_pullback(∂res)
        x, y = res.x, res.y
        proper_termination, max_iter, λ, m, n, N = res.info
        K = size(phi)[3]
        A, B = u[:,:,1], u[:,:,2]

        ∂self = NoTangent()
        ∂u = NoTangent()
        # ∂actions = NoTangent()
        ∂phi = NoTangent()

        # J_F
        # s = softmax(-A * y ./ λ)
        # u = softmax(-B' * x ./ λ)
        # J_s = softmax_jacobian(s)
        # J_u = softmax_jacobian(u)

        J_s = softmax_jacobian(x)
        J_u = softmax_jacobian(y)

        J_s_wrt_y =  J_s * (-A ./ λ)
        J_u_wrt_x =  J_u * (-B' ./ λ)
        J_softmax = [zeros(m,m) J_s_wrt_y; J_u_wrt_x zeros(n,n)]
        J_F = I(m+n) - J_softmax

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
