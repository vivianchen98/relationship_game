using Zygote, ChainRulesCore
using LinearAlgebra
include("game_solvers/entropy_nash_solver.jl")

# Given a RG, find its entropy_nash solution
function solve_relationship_game(u, actions, phi, w)  
    u_tilde = u + sum(w .* phi) * u

    solver = EntropySolver()
    res = solve_entropy_nash(solver, u_tilde, actions)

    return res
end

function evaluate(u, actions, phi, w)
    V = [3 2; 2 1]

    x, y, info = solve_relationship_game(u, actions, phi, w)
    return x' * V * y
end


function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, actions, phi, w)
    res = solve_relationship_game(u, actions, phi, w)

    function solve_relationship_game_pullback(∂res)
        x, y = res.x, res.y
        proper_termination, max_iter, λ, m, n = res.info
        A, B = u[1], u[2]

        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂actions = NoTangent()
        ∂phi = NoTangent()
        
        # J_F
        s = softmax(-A * y ./ λ)
        u = softmax(-B' * x ./ λ)
        J_s_wrt_y =  softmax_jacobian(s) * (-A ./ λ)
        J_u_wrt_x =  softmax_jacobian(u) * (-B' ./ λ)
        J_softmax = [zeros(m,m) J_s_wrt_y; J_u_wrt_x zeros(n,n)]
        J_F = I(m+n) - J_softmax

        # J_F_wrt_w
        J_s = softmax_jacobian(-A * y ./ λ)
        J_u = softmax_jacobian(-B' * x ./ λ)
        J_f_wrt_w = - (phi[1,:,:]'[:,:] * [(A*y)'; (B*y)'] ./ λ)'
        J_g_wrt_w = - (phi[2,:,:]'[:,:] * [x'*A; y'*B] ./ λ)'
        J_F_wrt_w = [-J_s * J_f_wrt_w; -J_u * J_g_wrt_w]

        ∂w = ([∂res.x; ∂res.y]' * inv(J_F) * J_F_wrt_w)'
        @show ∂w

        ∂self, ∂u, ∂actions, ∂phi, ∂w
    end

    res, solve_relationship_game_pullback
end


# test
u = [[1 3; 0 2];;; [1 0; 3 2]]
actions = [["C", "D"], ["C", "D"]]
phi = [[0 1; 2 0];;;[0 0; 1 0]]
V = [3 2; 2 1]
w = [.2, .8]

# @show evaluate(u, actions, phi, w)
# @show gradient(evaluate, u, actions, phi, w)

x, y, info = solve_relationship_game(u, actions, phi, w)
@show x
@show y
grad = gradient(solve_relationship_game, u, actions, phi, w)
@show grad