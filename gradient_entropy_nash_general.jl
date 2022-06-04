using Zygote, ChainRulesCore
using LinearAlgebra
include("game_solvers/entropy_nash_solver_general.jl")
# include("game_solvers/bimatrix_mixed_nash_solver.jl")
include("trafficN.jl")

function create_u_tilde(u, phi, w)
    N = length(u)
    w_phi = w' * phi
    u_tilde = u + [sum(w_phi[n,i] * u[i] for i in 1:N) for n in 1:N]
    return u_tilde
end

# Gradient of u_tilde (modified utilities) wrt w [tensor form]
function u_tilde_i_grad_w(u, phi, i)
    return cat([sum(phi[r][i,j] * u[j] for j in 1:N) for r in 1:length(phi)]..., dims=length(u) + 1)
end

# Jacobian of h^i wrt w
function J_h_i_wrt_w(u, phi, x, i)
    J_u_tilde = u_tilde_i_grad_w(u, phi, i)
    n = length(u)
    list_without_i = [s for s in 1:n]; deleteat!(list_without_i, i)
    pt_cost = J_u_tilde .* prob_prod(x, list_without_i, CartesianIndices(J_u_tilde))
    out = sum(pt_cost, dims=list_without_i)
    out = reshape(out, (length(x[i]), length(phi)))
    return out
end

# Jacobian of softmax(-h^i(x,w)/λ) wrt w
function J_s_i_wrt_w(u, phi, x, λ, i)
    s_i = softmax(x[i])
    return - softmax_jacobian(s_i) * J_h_i_wrt_w(u, phi, x, i) ./ λ
end

# Given a RG, find its entropy_nash solution
function solve_relationship_game(u, phi, w)
    u_tilde = create_u_tilde(u, phi, w)
    res = solve_entropy_nash_general(EntropySolver(), u_tilde)
    return res
end

function evaluate(u, phi, w, V)
    x, info = solve_relationship_game(u, phi, w)
    # full_list = [s for s in 1:length(u)]
    # cost = V .* prob_prod(x, full_list, CartesianIndices(V))
    # return sum(cost)
    return strategy_cost(x, V)
end

#global cost of a strategy profile
function strategy_cost(x, V)
    full_list = [s for s in 1:length(x)]
    cost = V .* prob_prod(x, full_list, CartesianIndices(V))
    return sum(cost)
end

function ChainRulesCore.rrule(::typeof(strategy_cost), x, V)
    res = strategy_cost(x, V)
    function strategy_cost_pullback(∂res)
        N = length(x)

        ∂self = NoTangent()
        ∂V = NoTangent()

        V_repeat = [V for i in 1:N]
        ∂x = [h(i, V_repeat, x)' * ∂res for i in 1:N]

        ∂self, ∂x, ∂V
    end
    res, strategy_cost_pullback
end

function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, phi, w)
    res = solve_relationship_game(u, phi, w)

    function solve_relationship_game_pullback(∂res)
        x = res.x
        x_vec = collect(Iterators.flatten(x))
        ∂x_vec = collect(Iterators.flatten(∂res.x))
        proper_termination, total_iter, max_iter, λ, N = res.info
        # K = size(phi)[3]

        s = [softmax(- h(i, u, x) ./ λ) for i in 1:N]
        s_vec = collect(Iterators.flatten(s))

        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂phi = NoTangent()

        # J_F
        N = length(u)
        J_submatrix(i,j) = (i == j) ? (zeros((size(u[i])[i], size(u[i])[i]))) : (softmax_jacobian(s[j]) * (-g(i,j,u,x) ./ λ)) #(softmax_jacobian(x[j]) * (-g(i,j,u,x) ./ λ))
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
        # CHECK FOR CORRECT SIZE
        m, n = size(J_softmax)
        total_actions = sum(size(u[i])[i] for i in 1:N)
        @assert m == total_actions && n == total_actions

        J_F = I(total_actions) - J_softmax
        @show size(J_F)

        # J_F_wrt_w
        J_F_wrt_w = - vcat([J_s_i_wrt_w(u, phi, x, λ, i) for i in 1:N]...)
        @show size(J_F_wrt_w)
        @show size(∂x_vec)

        #u_tilde = create_u_tilde(u, phi, w)
        #J_h_i_wrt_w = h(i, sum(phi[r,i,j] * u_tilde[j] for j in 1:N), x)
        #J_F_wrt_w = softmax_jacobian(s) ./ λ) ./ λ * J_h_i_wrt_w #softmax_jacobian(-h(i,create_u_tilde(u, phi, w), x) ./ λ) ./ λ * J_h_i_wrt_w

        # ∂x/∂w = - J_F \ J_F_wrt_w
        ∂w = (∂x_vec' * (J_F \ J_F_wrt_w))'

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
