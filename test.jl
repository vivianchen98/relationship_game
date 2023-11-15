include("examples.jl")
include("game_solvers/gradient_entropy_nash_jump.jl")

# perturbation test of gradient descent
function perturbation_test(w, J, perturbation=0.1, total_test=100)
    println("\nPerturbation test of gradient descent")
    violation = 0
    for _ in 1:total_test
        w_perturbed = w + perturbation * randn(size(w))
        J_perturbed = evaluate(u, V, phi, w_perturbed, λ)
        J_perturbed = J_perturbed + perturbation * 0.001

        @show J_perturbed > J
        if J_perturbed < J
            violation += 1
        end
    end
    println("\nViolation rate: ", violation/total_test)
end

function convexity_test(w, J, perturbation=0.1, total_test=100)
    println("\nConvexity test of gradient descent")
    violation = 0
    for _ in 1:total_test
        dw = perturbation * randn(size(w))
        w_perturbed = w + dw
        J_perturbed = evaluate(u, V, phi, w_perturbed, λ)

        ∂w = gradient(evaluate, u, V, phi, w, λ)[4]
        # @show J_perturbed - J ≥ ∂w' * dw
        if J_perturbed - J < ∂w' * dw
            violation += 1
        end
    end
    println("\nViolation rate: ", violation/total_test)
end

# given
# u, V, phi = congestion() # in `examples.jl`

# # step-wise test
# u_tilde = create_u_tilde(u, phi, w)
# x, = solve_entropy_nash_jump(u_tilde, λ) # in `game_solvers/entropy_nash_solver_jump.jl`
# strategy_cost(x, V)

# # combined test: J(x, V) (all other functions in `game_solvers/gradient_entropy_nash_jump.jl`)
# evaluate(u, V, phi, w, λ)

# # gradient computation
# gradient(evaluate, u, V, phi, w, λ)[4]

""" gradient descent for congestion() """
# GradientDescent(game, step_size, max_iter, λ: entropy weight, β: tolerance)
# w, J, terminate_step = GradientDescent(congestion(), 0.05, 5000, 0.7, 1e-4)

"""gradient descent for bee_queen() """
# w, J, (terminate_step) = GradientDescent(bee_queen(), 0.15, 10000, 0.8, 4e-4)
w, J, (terminate_step) = GradientDescent(bee_queen(), 0.2, 10000, 0.8, 2e-4)

# perturbation_test(w, J, 0.1, 100) # expected violation rate: 0.0
# convexity_test(w, J, 0.1, 100)