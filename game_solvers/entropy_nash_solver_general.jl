using Distributions
using LinearAlgebra
using ArgParse
include("entropy_nash_solver.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--lambda"
            help = "temperature"
            arg_type = Float64
            default = 0.01
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
    max_iter::Int = 1000
end

# stable softmax
function softmax(arr; θ=1)
    e = exp.((arr .- maximum(arr)) * θ)
    return e./ sum(e)
end

# jacobian matrix for a softmax distribution (given s = softmax(x))
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
    sum_dims = [s for s in 1:n]
    list_without_i = [s for s in 1:n]; deleteat!(list_without_i, i)
    pt_cost = u[i] .* prob_prod(x, list_without_i, CartesianIndices(u[i]))
    out = sum(pt_cost, dims=list_without_i)
    return vec(out)
end

function g(i, j, u, x)
    n = length(u)
    if n == 2
        if i < j
            return u[i]
        else
            return u[i]'
        end
    end
    list_without_i_j = [s for s in 1:n]; deleteat!(list_without_i_j, sort([i,j]))
    pt_cost = u[i] .* prob_prod(x, list_without_i_j, CartesianIndices(u[i]))
    out = sum(pt_cost, dims=list_without_i_j)
    out = permutedims(out, (i,j,list_without_i_j...))
    out = reshape(out, (length(x[i]), length(x[j])))
    return out
end

function x_vec_to_x(x_vec, u)
    num_actions = size(u[1])
    idx_counter = 1
    x = Vector{Vector{Float64}}()  ## Empty list of strategies
    for ii in 1:length(num_actions)
        x_ii = x_vec[idx_counter: idx_counter + num_actions[ii] - 1]
        push!(x, x_ii)
        idx_counter += num_actions[ii]
    end
    return x
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
function solve_entropy_nash_general(solver::EntropySolver, u; λ = args["lambda"], ϵ = args["epsilon"])
    N = length(u)

    if N == 2
        x1, x2, info = solve_entropy_nash(solver, u[1], u[2])
        x = [x1, x2]
        proper_termination, total_iter = info.proper_termination, info.total_iter
    else
        # initialize random mixed strategies
        x = [[1/size(u[i])[i] for counter in 1:size(u[i])[i]] for i in 1:N] # list of uniform distributions
        x_vec = collect(Iterators.flatten(x))

        total_iter = 0
        J_F = nothing
        for i in 1:solver.max_iter
            total_iter = i

            # s = softmax(-h(x_-i) ./ λ)
            s = [softmax(- h(i, u, x) ./ λ) for i in 1:N]
            s_vec = collect(Iterators.flatten(s))

            # J_s(j)_wrt_x(i) : try two sets of counters
            J_submatrix(i,j) = (i == j) ? (zeros((size(u[i])[i], size(u[i])[i]))) : (softmax_jacobian(s[j]) * (-g(i,j,u,x) ./ λ))
            J_softmax = []
            for i in 1:length(s)
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
            # @show J_softmax
            # @show size(J_softmax)
            m, n = size(J_softmax)
            total_actions = sum(size(u[i])[i] for i in 1:N)
            @assert m == total_actions && n == total_actions

            # Jacobian of F(x) = x - s
            J_F = I(total_actions) - J_softmax

            # compute step = inv(J_F) * ([x;y] - [s;u])
            β = 0.1
            step = (J_F' * J_F + β * I(total_actions)) \ J_F' * (x_vec - s_vec)

            # step if not convergent yet
            if norm(step, 2) < ϵ
                break
            else
                x_vec -= step
            end

            x_vec_to_x(x_vec, u) # x <- x_vec?? (update x here)
        end

        proper_termination = (total_iter < solver.max_iter)

    end

    (;
        x = [softmax(- h(i, u, x) ./ λ) for i in 1:N],
        info = (; proper_termination, total_iter, solver.max_iter, λ, N),
    )
end
