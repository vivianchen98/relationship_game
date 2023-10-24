using Zygote, ChainRulesCore
include("entropy_nash_solver_jump.jl")

function create_u_tilde(u, phi, w)
    N = length(u)
    w_phi = w' * phi
    u_tilde = u + [sum(w_phi[n,i] * u[i] for i in 1:N) for n in 1:N]
    return u_tilde
end

function prob_prod(x, player_indices, cartesian_indices)
    out = zeros(size(cartesian_indices))
    for idx in cartesian_indices
        out[idx] = prod(x[i, idx[i]] for i in player_indices)
    end
    return out
end

function strategy_cost(x, V)
    full_list = collect(1:size(x)[1])
    cost = V .* prob_prod(x, full_list, CartesianIndices(V))
    return sum(cost)
end

function solve_relationship_game(u, phi, w, λ)
    u_tilde = create_u_tilde(u, phi, w)
    x, = solve_entropy_nash_jump(u_tilde, λ)
    (; x = x)
end

function evaluate(u, V, phi, w, λ)
    x = solve_relationship_game(u, phi, w, λ)
    return strategy_cost(x, V)
end

function extract_x(res)
    return res.x
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

# TO DO: pullback of strategy_cost, solve_entropy_nash_jump, create_u_tilde
# => or make sure strategy_cost and create_u_tilde are compatible with Zygote auto-diff
# then do gradient(evaluate, u, V, phi, w, λ)[4] as ∂w


function ChainRulesCore.rrule(::typeof(strategy_cost), x, V)
    res = strategy_cost(x, V)
    function strategy_cost_pullback(∂res)
        N = size(x)[1]

        ∂self = NoTangent()
        ∂V = NoTangent()

        V_repeat = [V for i in 1:N]
        ∂x = vcat([J_except_i(i, x, V_repeat)' * ∂res for i in 1:N]...)

        ∂self, ∂x, ∂V
    end
    res, strategy_cost_pullback
end

function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, phi, w, λ)
    res = solve_relationship_game(u, phi, w, λ)

    function solve_relationship_game_pullback(∂res)
        x = res.x
        x_vec = collect(Iterators.flatten(x))
        ∂x_vec = collect(Iterators.flatten(∂res.x))
        # proper_termination, total_iter, max_iter, λ, N = res.info
        # K = size(phi)[3]

        # s = [softmax(- h(i, u, x) ./ λ) for i in 1:N]
        # s_vec = collect(Iterators.flatten(s))

        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂phi = NoTangent()

        # J_F
        N = length(u)
        J_submatrix(i,j) = (i == j) ? (zeros((size(u[i])[i], size(u[i])[i]))) : (softmax_jacobian(x[j]) * (-g(i,j,u,x) ./ λ)) #(softmax_jacobian(x[j]) * (-g(i,j,u,x) ./ λ))
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
        #@show size(J_F)

        # J_F_wrt_w
        J_F_wrt_w = - vcat([J_s_i_wrt_w(u, phi, x, λ, i) for i in 1:N]...)
        # @show size(J_F_wrt_w)
        # @show size(∂x_vec)

        #u_tilde = create_u_tilde(u, phi, w)
        #J_h_i_wrt_w = h(i, sum(phi[r,i,j] * u_tilde[j] for j in 1:N), x)
        #J_F_wrt_w = softmax_jacobian(s) ./ λ) ./ λ * J_h_i_wrt_w #softmax_jacobian(-h(i,create_u_tilde(u, phi, w), x) ./ λ) ./ λ * J_h_i_wrt_w

        # ∂x/∂w = - J_F \ J_F_wrt_w
        ∂w = - (∂x_vec' * (J_F \ J_F_wrt_w))'
        # ∂w = (∂x_vec' * (J_F \ J_F_wrt_w))'

        ∂self, ∂u, ∂phi, ∂w
    end

    res, solve_relationship_game_pullback
end


# gradient(evaluate, u, V, phi, w, λ)[4]