using Zygote, ChainRulesCore
include("entropy_nash_solver_jump.jl")

function create_u_tilde(u, phi, w)
    N = length(u)
    w_phi = w' * phi
    # u_tilde = u + [sum(w_phi[n,i] * u[i] for i in 1:N) for n in 1:N]
    u_tilde = [sum(w_phi[n,i] * u[i] for i in 1:N) for n in 1:N]
    return u_tilde
end

function prob_prod_jump(x, player_indices, cartesian_indices)
    return [prod(x[i, idx[i]] for i in player_indices) for idx in cartesian_indices]
end

function strategy_cost(x, V)
    cost = V .* prob_prod_jump(x, 1:size(x,1), CartesianIndices(V))
    return sum(cost)
end

function solve_relationship_game(u, phi, w, λ)
    u_tilde = create_u_tilde(u, phi, w)
    x = solve_entropy_nash_jump(u_tilde, λ)
    return x
end

function evaluate(u, V, phi, w, λ)
    x = solve_relationship_game(u, phi, w, λ)
    return strategy_cost(x, V)
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

# Derivative of u_tilde (modified utilities) wrt w (tensor form)
function u_tilde_i_grad_w(u, phi, i)
    return cat([sum(phi_r[i,j] * u[j] for j in eachindex(u)) for phi_r in phi]..., dims=length(u) + 1)
end

# Jacobian of h^i wrt w
function J_h_i_wrt_w(u, phi, x, i)
    J_u_tilde = u_tilde_i_grad_w(u, phi, i)
    out = h_except_i(i, x, J_u_tilde)
    return reshape(out, (length(x[i, :]), length(phi)))
end

# Jacobian of softmax(-h^i(x,w)/λ) wrt w
function J_s_i_wrt_w(u, phi, x, λ, i)
    s_i = softmax(-h_except_i(i, x, u[i]) ./ λ)
    return - softmax_jacobian(s_i) * J_h_i_wrt_w(u, phi, x, i) ./ λ
end

function g(i, j, u, x)
    n = size(x)[1]
    if n == 2
        if i < j
            return u[i]
        else
            return u[i]'
        end
    end
    list_without_i_j = [s for s in 1:n]; deleteat!(list_without_i_j, sort([i,j]))
    pt_cost = u[i] .* prob_prod_jump(x, list_without_i_j, CartesianIndices(u[i]))
    out = sum(pt_cost, dims=list_without_i_j)
    out = permutedims(out, (i,j,list_without_i_j...))
    out = reshape(out, (length(x[i, :]), length(x[j, :])))
    return out
end

function h_except_i(i, x, u_i)
    N = size(x, 1)
    full_list_except_i = [s for s in 1:N if s != i]

    # Calculates costs across all action distributions except for player i, then sums across dims other than player i
    costs = sum(u_i .* prob_prod_jump(x, full_list_except_i, CartesianIndices(u_i)), dims=full_list_except_i)
    return vcat(costs...) #We need to flatten the resulting 1D tensor into an array
end

function J_except_i(i, x, u)
    return h_except_i(i, x, u[i])
end

# TO DO: pullback of strategy_cost, solve_entropy_nash_jump, create_u_tilde
# => or make sure strategy_cost and create_u_tilde are compatible with Zygote auto-diff
# then do gradient(evaluate, u, V, phi, w, λ)[4] as ∂w

function ChainRulesCore.rrule(::typeof(strategy_cost), x, V)
    J = strategy_cost(x, V)
    function strategy_cost_pullback(∂J)
        # placeholder for unused partial derivatives ∂V
        ∂self = NoTangent()
        ∂V = NoTangent()

        # ∂x := the partial derivative operator wrt x (∂/∂x)
        # ∂J := the partial derivative operator wrt J (∂/∂J)
        # ∂x = ∂J/∂x * ∂/∂J = [gradient to find] * ∂J
        N = size(x)[1]
        ∂x = vcat([h_except_i(i, x, V)' * ∂J for i in 1:N]...)

        ∂self, ∂x, ∂V
    end
    J, strategy_cost_pullback
end

function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, phi, w, λ)
    res = solve_relationship_game(u, phi, w, λ)

    function solve_relationship_game_pullback(∂res)
        x = res

        # flatten x as a vector z
        ∂z = collect(Iterators.flatten(∂res))

        # placeholder for unused partial derivatives ∂u, ∂phi
        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂phi = NoTangent()
        ∂λ = NoTangent()

        # J_F_wrt_z = ∂z/∂z - ∂s/∂z
        N = length(u)
        # N = size(res, 1)
        J_submatrix(i,j) = (i == j) ? (zeros((size(u[i])[i], size(u[i])[i]))) : (softmax_jacobian(x[j,:]) * (-g(i,j,u,x) ./ λ)) #(softmax_jacobian(x[j]) * (-g(i,j,u,x) ./ λ))
        J_softmax = []
        for i in 1:N
            row = []
            for j in 1:N
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
        total_player_times_action = sum(size(u[i])[i] for i in 1:N)
        @assert m == total_player_times_action && n == total_player_times_action

        # J_F_wrt_z = I .- J_softmax
        J_F_wrt_z = I - J_softmax

        # J_F_wrt_w = - ∂s/∂w
        J_F_wrt_w = - vcat([J_s_i_wrt_w(u, phi, x, λ, i) for i in 1:N]...)

        # ∂z/∂w = - J_F_wrt_z \ J_F_wrt_w
        ∂w = - (∂z' * (J_F_wrt_z\ J_F_wrt_w))'

        ∂self, ∂u, ∂phi, ∂w, ∂λ
    end

    res, solve_relationship_game_pullback
end

# function project(w)
#     # project onto the probability simplex
#     K = length(w)
#     w_abs = abs.(w)
#     w_abs = w_abs - (1/K) * ones(K)
#     w_abs = max.(w_abs, 0)
#     w = w / sum(w_abs)
#     return w
# end

# using HiGHS
# function l1_orthogonal_project(w)
#     model = Model(HiGHS.Optimizer)
#     set_silent(model)

#     @variable(model, y[1:length(w)] >= 0)

#     @constraint(model, sum(abs(y[i]) for i in 1:length(w)) ≤ 1)

#     @objective(model, Min, sum((y - w).^2))

# end

function l2_project(w)
    return w / norm(w)
end

# Gradient Descent of social cost V on weight vector w
function GradientDescent(g, stepsize, max_iter, λ, β)
    w_list = Vector{Vector{Float64}}()
    J_list = Vector{Float64}()
    terminate_step = 0

    # init w
    K = length(g.phi)
    w = randn(K)
    w = w / norm(w)
    #w = [1/sqrt(K) for i in 1:K] # unifrom distribution of length K
    #w = zeros(K); w[1] = 1 # only one relationship

    push!(w_list, w)
    push!(J_list, evaluate(g.u, g.V, g.phi, w, λ))
    println("start with w=($w)")

    for i in 1:max_iter
        ∂w = gradient(evaluate, g.u, g.V, g.phi, w, λ)[4]
        w = w - stepsize .* ∂w

        # project onto the probability simplex
        w = l2_project(w)

        push!(w_list, w)
        current_J = evaluate(g.u, g.V, g.phi, w, λ)
        push!(J_list, current_J)
        if i % 100 == 0
            println("step $(i): $w")
            @show current_J
            println()
        end
        if norm(∂w - (∂w' * w) / (norm(w)^2) * w)  < β # stopping criteria
        #if norm(∂w) < β # stopping criteria
        # if evaluate(g.u, g.phi, w, g.V) - (-1) ≤ 0.01
            println("terminate with w=($w) in $(i) steps, with J=$(current_J)")
            terminate_step = i
            break
        end
        if i == max_iter
            println("Does not converge within ($max_iter) iterations: norm(∂w)=($(norm(∂w)))")
        end
    end

    (;  w = w, J = evaluate(g.u, g.V, g.phi, w, λ), 
        info = (terminate_step = terminate_step, J_list = J_list, w_list = w_list))
end