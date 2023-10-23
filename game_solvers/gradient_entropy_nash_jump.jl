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

function evaluate(u, V, phi, w, λ)
    u_tilde = create_u_tilde(u, phi, w)
    x, = solve_entropy_nash_jump(u_tilde, λ)
    return strategy_cost(x, V)
end


# TO DO: pullback of strategy_cost, solve_entropy_nash_jump, create_u_tilde
# => or make sure strategy_cost and create_u_tilde are compatible with Zygote auto-diff
# then do gradient(evaluate, u, V, phi, w, λ)[4] as ∂w