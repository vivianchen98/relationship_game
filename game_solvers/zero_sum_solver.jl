"""
A zero-sum game solver that casts the game as linear program.
"""
struct ZeroSumSolver end

function solve_mixed_nash(::ZeroSumSolver, A)
    sol1 = solve_mixed_security_strategy(A)
    sol2 = solve_mixed_security_strategy(-A')
    (; x = sol1.x, y = sol2.x)
end

function solve_mixed_security_strategy(player_cost_matrix)
    offset = -minimum(player_cost_matrix) .+ 1

    # Construction of constraints
    A = (player_cost_matrix .+ offset)
    sol = solve_simplex_lp(A)
    x = sol.x / sol.V
    V = 1 / sol.V - offset
    (; x, V)
end

function solve_simplex_lp(A)
    model = JuMP.Model()
    JuMP.set_optimizer(model, OSQP.Optimizer)
    JuMP.set_silent(model)

    m, n = size(A)
    @variable(model, 0 ≤ x[1:m])
    @constraint(model, A' * x .≤ 1)
    @objective(model, Max, sum(x))

    JuMP.optimize!(model)
    @assert JuMP.termination_status(model) == MathOptInterface.OPTIMAL

    (; x = JuMP.value.(x), V = JuMP.objective_value(model))
end
