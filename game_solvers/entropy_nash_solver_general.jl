using Distributions
using LinearAlgebra
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--lambda"
            help = "temperature"
            arg_type = Float64
            default = 0.1
        "--epsilon"
            help = "Newton's convergence threshold"
            arg_type = Float64
            default = 0.001
        "--nash_type"
            help = "Nash or Entropy-Nash"
            arg_type = String
            default = "entropy_nash"
        "--plot-gradient"
            help = "surface/heatmap/gradient_heatmap"
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end

args = parse_commandline()

# entropy-regularized Nash solver
Base.@kwdef struct EntropySolver
    "The maximum number of iterations allowed."
    max_iter::Int = 10000
end

# stable softmax
function softmax(arr; θ=1)
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

# -i covector
function prob_prod(x, player_indices, cartesian_indices)
    out = zeros(size(cartesian_indices))
    for idx in cartesian_indices
        out[idx] = prod(x[i][idx[i]] for i in player_indices)
    end
    return out
end

function h(i, u, x)
    n = length(u)
    list_without_i = [s for s in 1:n]; deleteat!(list_without_i, i)
    pt_cost = u[i] .* prob_prod(x, list_without_i, CartesianIndices(u[i]))
    out = sum(pt_cost, dims=list_without_i)
    return vec(out)
end

function g(i, j, u, x)
    n = length(u)
    list_without_i_j = [s for s in 1:n]; deleteat!(list_without_i_j, sort([i,j]))
    pt_cost = u[i] .* prob_prod(x, list_without_i_j, CartesianIndices(u[i]))
    h = sum(pt_cost, dims=list_without_i_j)
    # h = vec(h)
    return h
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
function solve_entropy_nash(solver::EntropySolver, u; λ = args["lambda"], ϵ = args["epsilon"])
    N = length(u)

    # initialize random mixed strategies
    x = [[1/size(u[i])[i] for counter in 1:size(u[i])[i]] for i in 1:N] # list of uniform distributions
    x_vec = collect(Iterators.flatten(x))

    total_iter = 0
    J_F = nothing
    for i in 1:solver.max_iter
        total_iter = i

        # s = softmax(-h(x_-i) ./ λ)
        s = [softmax(- h(i, u, x) ./ λ) for i in 1:N]    

        # J_s(j)_wrt_x(i)
        # J(i,j) = (i == j) ? (zeros(size(u[i])[i])) : (softmax_jacobian(s[j]) * (-g(i,j,u,x) ./ λ))
        J_softmax = zeros((length(x), length(s)))
        for i in 1:length(x), j in 1:length(s)
            if i == j
                J_softmax[i,j] = zeros(size(u[i])[i])
            else
                J_softmax[i,j] = softmax_jacobian(s[j]) * (-g(i,j,u,x) ./ λ)
        end

        # Jacobian of F(x) = x - s
        total_actions = sum(size(u[i])[i] for i in 1:N)
        J_F = I(total_actions) - J_softmax

        # compute step
        # step = inv(J_F) * ([x;y] - [s;u])
        β = 0.1
        step = (J_F' * J_F + β * I(total_actions)) \ J_F' * (x - s)

        # step if not convergent yet
        if norm(step, 2) < ϵ
            break
        else
            x -= step
        end
    end

    proper_termination = (total_iter < solver.max_iter)

    (;
        x = [softmax(- h(i, u, x) ./ λ) for i in 1:N]
        info = (; proper_termination, solver.max_iter, λ, N),
    )
end