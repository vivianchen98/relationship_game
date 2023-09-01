using JuMP, LinearAlgebra, Ipopt

# stable softmax
function softmax(arr; θ=1)
    e = exp.((arr .- maximum(arr)) * θ)
    return e./ sum(e)
end

function lazy_J_except_i(i, x, u)
    num_actions = size(u[1])[1]
    cost = zeros(num_actions)
    for ai in 1:num_actions
        x_alt = copy(x)
        pure_strat_i = zeros(num_actions)
        pure_strat_i[ai] = 1
        x_alt[i] = pure_strat_i
        full_list = [s for s in 1:length(x_alt)]
        cost[ai] = sum(u[i] .* prob_prod(x_alt, full_list, CartesianIndices(u[i])))
    end
    return cost
end

function solve_entropy_nash_general(u, λ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    N = length(u)
    num_actions = size(u[1])[1]

    # variables
    @variable(model, x[1:N; 1:num_actions] >= 0)


    # slack variable
    @variable(model, p[1:N])

    # constraints
    for i in 1:N
        @NLconstraint(model, x[i, :] - softmax(- lazy_J_except_i(i, x, u) ./ λ) == p[i])
    end

    # objective
    @objective(m, Min, sum(p[i]*p[i] for i in 1:N))


    optimize!(m)

    (; x = value.(x))
end