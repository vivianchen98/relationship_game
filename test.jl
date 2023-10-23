include("examples.jl")
include("game_solvers/gradient_entropy_nash_jump.jl")

# given
u, V, phi = congestion() # in `examples.jl`

# specify parameters
w = [1]
λ = 0.5

# step-wise test
u_tilde = create_u_tilde(u, phi, w)
x, = solve_entropy_nash_jump(u_tilde, λ) # in `game_solvers/entropy_nash_solver_jump.jl`
strategy_cost(x, V)

# combined test: J(x, V) (all other functions in `game_solvers/gradient_entropy_nash_jump.jl`)
evaluate(u, V, phi, w, λ)
