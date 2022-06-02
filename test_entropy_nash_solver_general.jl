include("game_solvers/entropy_nash_solver_general.jl")
include("trafficN.jl")

solver = EntropySolver()
u = generate_traffic(3, [2,2, 5])
@show u

x, info = solve_entropy_nash_general(solver, u)
@show x