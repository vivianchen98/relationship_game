include("game_solvers/entropy_nash_solver_general.jl")
include("trafficN.jl")

solver = EntropySolver()
u = generate_traffic(2, [2,2])
@show u

x, info = solve_entropy_nash_general(solver, u)
@show x
@show info.total_iter