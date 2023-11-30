include("examples.jl")
include("game_solvers/gradient_entropy_nash_jump_new.jl")
using Plots

α = 0.1
λ = 0.5
# w, J, (terminate_step, J_list, w_list) = ProjectedGradientDownstairs(bee_queen(), α, 5000, λ, 1e-3)


w, J, (terminate_step, J_list, w_list) = ProjectedGradientDownstairs(undivided_congestion(3,3), α, 5000, λ, 1e-4)