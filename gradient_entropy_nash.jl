using Zygote, ChainRulesCore
using LinearAlgebra
include("game_solvers/entropy_nash_solver.jl")
include("game_solvers/bimatrix_mixed_nash_solver.jl")

# Given a RG, find its entropy_nash solution
function solve_relationship_game(u, actions, phi, w)
    m, n, N = size(u)
    u_tilde = u + [ (phi[k,:,:] * w)' * u[i,j,:] for i in 1:m, j in 1:n, k in 1:N]

    solver = EntropySolver()
    res = solve_entropy_nash(solver, u_tilde, actions)

    return res
end


function evaluate(u, actions, phi, w, V)
    x, y, info = solve_relationship_game(u, actions, phi, w)
    return x' * V * y
end


function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, actions, phi, w)
    res = solve_relationship_game(u, actions, phi, w)

    function solve_relationship_game_pullback(∂res)
        x, y = res.x, res.y
        proper_termination, max_iter, λ, m, n, N = res.info
        K = size(phi)[3]
        A, B = u[:,:,1], u[:,:,2]

        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂actions = NoTangent()
        ∂phi = NoTangent()


        # J_F
        s = softmax(-A * y ./ λ)
        u = softmax(-B' * x ./ λ)
        J_s = softmax_jacobian(s)
        J_u = softmax_jacobian(u)

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

        ∂self, ∂u, ∂actions, ∂phi, ∂w
    end

    res, solve_relationship_game_pullback
end



# ************* EXAMPLES *************
# prisoner's dilemma
function prisoner()
    name = "prisoner"
    u = [[1 3; 0 2];;;[1 0; 3 2]]
    actions = [["C", "D"], ["C", "D"]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    V = [-1 0; 0 0]
    (; name=name, u=u, actions=actions, phi=phi, V=V)
end 

# 2-route traffic
function traffic2()
    name = "traffic2"
    u = [[2 4; 1.5 3];;;[2 1.5; 4 3]]
    actions = [["A", "B"], ["A", "B"]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    # V = sum(u[:,:,i] for i in 1:2)
    V = [-1 0; 0 0]
    (; name=name, u=u, actions=actions, phi=phi, V=V)
end

# 3-route traffic
function traffic3()
    name = "traffic3"
    u = [[2 4 4; 1.5 3 1.5; 2.5 2.5 5];;;[2 1.5 2.5; 4 3 2.5; 4 1.5 5]] 
    actions = [["A", "B", "C"], ["A", "B", "C"]]
    phi = [[0 1; 2 0];;;[0 0; 1 0]]
    V = sum(u[:,:,i] for i in 1:2)
    (; name=name, u=u, actions=actions, phi=phi, V=V)
end


# ************* DEFINE PROBLEM *************
# game structure
name, u, actions, phi, V = prisoner()

# 2 player relationship weights
w = [1, 1]


# ************* SOLUTIONS *************
# x, y, info = solve_relationship_game(u, actions, phi, w)
# @show x
# @show y

# expected_V = evaluate(u, actions, phi, w, V) # Remember V is defined inside evaluate()!
# @show expected_V

# grad = gradient(evaluate, u, actions, phi, w, V)
# @show grad
