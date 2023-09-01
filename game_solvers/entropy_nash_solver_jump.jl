using JuMP, LinearAlgebra, Ipopt
using Zygote, ChainRulesCore


function prob_prod(x, player_indices, cartesian_indices)
    out = zeros(size(cartesian_indices))
    for idx in cartesian_indices
        out[idx] = prod(x[i][idx[i]] for i in player_indices)
    end
    return out
end

function lazy_J_except_i(i, x, u)
    num_actions = size(u[1])[1]
    N = length(x)
    cost = zeros(num_actions)
    for ai in 1:num_actions
        x_alt = copy(x)
        pure_strat_i = zeros(num_actions)
        pure_strat_i[ai] = 1
        x_alt[i] = pure_strat_i
        # full_list = [s for s in 1:length(x_alt)]
        # cost[ai] = sum(u[i] .* prob_prod(x_alt, full_list, CartesianIndices(u[i])))
        cost[ai] = sum(u[i] .* prob_prod(x_alt, collect(1:N), CartesianIndices(u[i])))
    end
    return cost
end

function lazy_J_except_i_jump(i, x, u, N, num_actions)
    # num_actions = size(u[1])[1]
    # N = length(x)
    cost = zeros(num_actions)
    for ai in 1:num_actions
        x_alt = copy(x)
        pure_strat_i = zeros(num_actions)
        pure_strat_i[ai] = 1
        x_alt[i] = pure_strat_i
        # full_list = [s for s in 1:length(x_alt)]
        # cost[ai] = sum(u[i] .* prob_prod(x_alt, full_list, CartesianIndices(u[i])))
        cost[ai] = sum(u[i] .* prob_prod(x_alt, collect(1:N), CartesianIndices(u[i])))
    end
    return cost
end

function solve_entropy_nash_general(u, λ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    N = length(u)
    num_actions = size(u[1])[1]

    # variables
    @variable(model, x[1:N, 1:num_actions] >= 0)


    # slack variable
    @variable(model, p[1:N, 1:num_actions])
    @variable(model, cost[1:N, 1:num_actions])


    # ??? cost = J_except_i(x,u[i])
    for i in 1:N
        # @constraint(model, cost[i, :] == [sum(u[i] .* out[i, :, :, :]) for i in 1:N])

        sum(prod(x[i, ]) .* u[i, j] for j in 1:num_actions)

    end

    # softmax(-cost/λ)
    for i in 1:N, j in 1:num_actions
        @NLconstraint(model, x[i, j] - (exp(-cost[i,j]/λ) / sum(exp(-cost[i,j]/λ) for j in 1:num_actions)) == p[i, j])
    end

    # objective
    @objective(m, Min, sum(sum(p[i, :] .* p[i, :]) for i in 1:N))


    optimize!(m)

    (; x = value.(x))
end