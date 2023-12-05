include("examples.jl")
include("game_solvers/gradient_entropy_nash_jump.jl")
using Plots

# given
# name, u, V, phi = congestion() # in `examples.jl`

# # step-wise test
# u_tilde = create_u_tilde(u, phi, w)
# x, = solve_entropy_nash_jump(u_tilde, λ) # in `game_solvers/entropy_nash_solver_jump.jl`
# strategy_cost(x, V)

# # combined test: J(x, V) (all other functions in `game_solvers/gradient_entropy_nash_jump.jl`)
# evaluate(u, V, phi, w, λ)

# # gradient computation
# gradient(evaluate, u, V, phi, w, λ)[4]


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

""" GradientDescent(game, step_size, max_iter, λ: entropy weight, β: tolerance) 
    returns (w, J, info = (terminate_step, J_list, w_list)) 
"""

""" for congestion() """
# w, J, (terminate_step, J_list, w_list) = GradientDescent(congestion(), 0.05, 5000, 0.7, 1e-4)

""" for bee_queen() """
# w, J, (terminate_step, J_list, w_list) = GradientDescent(bee_queen(), 0.2, 10000, 0.8, 2e-4)

# perturbation_test(w, J, 0.1, 100) # expected violation rate: 0.0
# convexity_test(w, J, 0.1, 100)


function experiment(game, α, β, λ_list, max_iter)
    J_dict_of_lists = Dict()
    w_dict_of_lists = Dict()
    for λ in λ_list
        w, J, (terminate_step, J_list, w_list) = GradientDescent(game, α, max_iter, λ, β)
        J_dict_of_lists[λ] = J_list
        w_dict_of_lists[λ] = w_list
    end
    return J_dict_of_lists, w_dict_of_lists
end

# plot J_dict_of_lists in one figure
function plot_J(J_dict_of_lists; xlabel="Iteration", ylabel="J value", title="")
    plot()
    for (λ, J_list) in J_dict_of_lists
        plot!(J_list, label="λ=$λ")
    end
    xlabel!(xlabel)
    ylabel!(ylabel)
    title!(title)
    display(plot!())
end

# plot every 100 iterations
function plot_J_interval(J_dict_of_lists; xlabel="Iteration", ylabel="J value", title="")
    plot()
    for (λ, J_list) in J_dict_of_lists
        # plot at every 100 iterations
        plot!(J_list[1:100:end], label="λ=$λ")
    end
    xlabel!(xlabel)
    ylabel!(ylabel)
    title!(title)
    display(plot!())
end

# functions to save to pickle
using PyCall
py"""
import pickle
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data1 = pickle.load(f)
        data2 = pickle.load(f)
    return data1, data2
def save_pickle(data1, data2, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(data1, f)
        pickle.dump(data2, f)
"""
load_pickle = py"load_pickle"
save_pickle = py"save_pickle"

# set parameters for congestion() experiments
""" !!! frozen, don't touch !!! """
# game = congestion()
# α = 0.05
# β = 1e-4
# λ_list = [0.1, 0.2, 0.3, 0.4, 0.5]
# max_iter = 5000

# J_dict_of_lists, w_dict_of_lists = experiment(game, α, β, λ_list, max_iter)
# plot_J(J_dict_of_lists, title="$(game.name), α=$(α), β=$(β)")
# save_pickle(J_dict_of_lists, w_dict_of_lists, "results/congestion.pkl")
""" !!! frozen, don't touch !!! """


# set parameters for bee_queen() experiments
# game = bee_queen()
# α = 0.5
# β = 2e-4
# λ_list = [1.0]
# max_iter = 5000

# run experiment
# J_dict_of_lists, w_dict_of_lists = experiment(game, α, β, λ_list, max_iter)
# plot_J(J_dict_of_lists, title="$(game.name), α=$(α), β=$(β)")
# save_pickle(J_dict_of_lists, w_dict_of_lists, "results/congestion.pkl")

# w, J, (terminate_step, J_list, w_list) = GradientDescent(bee_queen(), 0.5, 5000, 1.0, 2e-4)

# w, J, (terminate_step, J_list, w_list) = GradientDescent(congestion(), 0.1, 5000, 0.5, 1e-4)


# w, J, (terminate_step, J_list, w_list) = GradientDescent(congestion(), 0.1, 5000, 0.5, 1e-4)


# w, J, (terminate_step, J_list, w_list) = GradientDescent(bee_queen(), 0.1, 5000, 0.5, 1e-4)

# w, J, (terminate_step, J_list, w_list) = GradientDescent(undivided_congestion(4, 2), 0.1, 5000, 0.5, 1e-4)