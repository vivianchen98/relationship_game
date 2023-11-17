using JuMP, LinearAlgebra, Ipopt

"""helper functions"""
function prob_prod_jump(x, player_indices, cartesian_indices)
    return [prod(x[i, idx[i]] for i in player_indices) for idx in cartesian_indices]
end

function J_except_i(i, x, u)
    full_list_except_i = [j for j in 1:length(u) if j != i]

    # Calculates costs across all action distributions except for player i, then sums across dims other than player i
    costs = sum(u[i] .* prob_prod_jump(x, full_list_except_i, CartesianIndices(u[i])), dims=full_list_except_i)


    sum([u[i][idx] * prod(x_real[k, idx[k]] for k in full_list_except_i) for idx in cartesian_indices], dims=full_list_except_i)

    return vcat(costs...) #We need to flatten the resulting 1D tensor into an array
end

"""
Compute entropy-regularized Nash mixed strategies using JuMP and Ipopt
Inputs:
- u, utility tensor
- λ, entropy weight

Returns:
- x: mixed eq strategies for each player
"""
function solve_entropy_nash_jump(u, λ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    N = length(u)
    num_actions = size(u[1])[1]

    # variables
    @variable(model, x[1:N, 1:num_actions] >= 0)
    # for i in 1:N, j in 1:num_actions
    #     set_start_value(x[i,j], 1/num_actions)
    # end

    # slack variable
    @variable(model, p[1:N, 1:num_actions])
    @variable(model, cost[1:N, 1:num_actions])
    @variable(model, J[1:N, 1:num_actions])

    # constraints
    for i in 1:N
        # cost_i = -J(x^{-i}, u^i) ./ λ
        full_list_except_i = [j for j in 1:N if j != i]
        cartesian_indices = CartesianIndices(u[i])
        for j in 1:num_actions
            cartesian_indices_j = selectdim(cartesian_indices, i, j)
            @NLconstraint(model, cost[i, j] == - sum(u[i][idx] * prod(x[k, idx[k]] for k in full_list_except_i) for idx in cartesian_indices_j) / λ)
        end

        # p_i = x_i - softmax(cost_i)
        for j in 1:num_actions
            @NLconstraint(model, p[i, j] == x[i, j] - exp(cost[i,j]) / sum(exp(cost[i,j]) for j in 1:num_actions))
        end

        # prob simplex constraint
        @constraint(model, sum(x[i,j] for j in 1:num_actions) == 1)
    end

    # objective
    @objective(model, Min, sum(sum(p[i, :] .* p[i, :]) for i in 1:N))

    optimize!(model)

    print("status: ", termination_status(model), "\n")
    return value.(x)
end