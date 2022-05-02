using Distributions
using LinearAlgebra

# entropy-regularized Nash solver
Base.@kwdef struct EntropySolver
    "The maximum number of iterations allowed."
    max_iter::Int = 1000
end

# stable softmax
function softmax(arr; θ=0.1)
    e = exp.((arr .- maximum(arr)) * θ)
    return e./ sum(e)
end

# jacobian matrix for a softmax distribution
function softmax_jacobian(s)
    len_s = length(s)
    J_s = zeros(len_s, len_s)
    for i in 1:len_s, j in 1:len_s
        J_s[i,j] = (i == j) ? (s[i] * (1-s[i])) : (-s[i] * s[j])
    end
    return J_s
end

"""
Compute entropy-regularized Nash mixed strategies by Newton's method

Inputs:
- u: cost matrices for all players
- actions: action choices for all players

Returns:
- x: mixed eq strategies for player 1
- y: mixed eq strategies for player 2
- proper_termination: if the algorithm converges within given max_iter
- max_iter: maximum number of iterations allowed
"""
function solve_entropy_nash(solver::EntropySolver, u, actions; λ = 0.005, ϵ = 0.01)
    A, B = u[:,:,1], u[:,:,2]
    m, n = length(actions[1]), length(actions[2])

    # initialize random mixed strategies
    x = rand(Dirichlet(m, 1.0))     # for player 1 with m actions
    y = rand(Dirichlet(n, 1.0))     # for player 2 with n actions

    total_iter = 0
    for i in 1:solver.max_iter
        total_iter = i

        s = softmax(-A * y ./ λ)
        u = softmax(-B' * x ./ λ)

        # Jacobian of F([x;y]) = [x;y] - [s;u]
        J_s_wrt_y =  softmax_jacobian(s) * (-A ./ λ)
        J_u_wrt_x =  softmax_jacobian(u) * (-B' ./ λ)
        J_softmax = [zeros(m,m) J_s_wrt_y; J_u_wrt_x zeros(n,n)]
        J_F = I(m+n) - J_softmax

        # step if not convergent yet
        step = inv(J_F) * ([x;y] - [s;u])
        if norm(step, 2) < ϵ
            break
        else
            x -= step[1:m]
            y -= step[m+1:m+n]
        end
    end

    proper_termination = (total_iter < solver.max_iter)

    (;
        x,
        y,
        info = (; proper_termination, solver.max_iter, λ, m, n),
    )
end


# example: prisoner's dilemma
# u = [[1 3; 0 2];;; [1 0; 3 2]]
# actions_1 = ["C", "D"]
# actions_2 = ["C", "D"]
# actions = [actions_1, actions_2]
#
#
# solver = EntropySolver()
# x, y, info = solve_entropy_nash(solver, u, actions)
# @show x
# @show y
# @show info
# @show info.λ
