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
function solve_entropy_nash(solver::EntropySolver, u, actions; λ = args["lambda"], ϵ = args["epsilon"])
    A, B = u[:,:,1], u[:,:,2]
    m, n, N = size(u)

    # initialize random mixed strategies
    # x = rand(Dirichlet(m, 1.0))     # for player 1 with m actions
    # y = rand(Dirichlet(n, 1.0))     # for player 2 with n actions
    x = [1/m for i in 1:m] # unifrom distribution of length K
    y = [1/n for i in 1:n] # unifrom distribution of length K



    total_iter = 0
    J_F = nothing
    for i in 1:solver.max_iter
        total_iter = i

        s = softmax(-A * y ./ λ)
        u = softmax(-B' * x ./ λ)

        # Jacobian of F([x;y]) = [x;y] - [s;u]
        J_s_wrt_y =  softmax_jacobian(s) * (-A ./ λ)
        J_u_wrt_x =  softmax_jacobian(u) * (-B' ./ λ)
        J_softmax = [zeros(m,m) J_s_wrt_y; J_u_wrt_x zeros(n,n)]
        J_F = I(m+n) - J_softmax
        # @show J_F

        # step if not convergent yet
        # step = inv(J_F) * ([x;y] - [s;u])
        β = 1
        step = (J_F' * J_F + β * I(m+n)) \ J_F' * ([x;y] - [s;u])


        if norm(step, 2) < ϵ
            break
        else
            x -= step[1:m]
            y -= step[m+1:m+n]
        end
    end

    proper_termination = (total_iter < solver.max_iter)
    # if !proper_termination
    #     @show J_F
    #     @show det(J_F)
    # end

    (;
        x = softmax(-A * y ./ λ),
        y = softmax(-B' * x ./ λ),
        # x,
        # y,
        info = (; proper_termination, solver.max_iter, λ, m, n, N),
    )
end