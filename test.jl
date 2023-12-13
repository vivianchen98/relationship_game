include("examples.jl")
include("game_solvers/gradient_entropy_nash_jump_new.jl")
using Plots

function experiment(game, α, β, λ_list, max_iter, method)
    J_dict_of_lists = Dict()
    w_dict_of_lists = Dict()
    for λ in λ_list
        if method == "minmax"
            w, J, (terminate_step, J_list, w_list) = ProjectedGradientMinMax(game, α, max_iter, λ, β)
        elseif method == "downstairs"
            w, J, (terminate_step, J_list, w_list) = ProjectedGradientDownstairs(game, α, max_iter, λ, β)
        else
            error("method not supported")
        end
        J_dict_of_lists[λ] = J_list
        w_dict_of_lists[λ] = w_list
    end
    return J_dict_of_lists, w_dict_of_lists
end

# plot J_dict_of_lists in one figure
function plot_J(J_dict_of_lists;iter_cap=1000, xlabel="Iteration", ylabel="J value", title="")
    plot()
    for (λ, J_list) in J_dict_of_lists
        if length(J_list) > iter_cap
            J_list = J_list[1:iter_cap]
        end
        plot!(J_list, label="λ=$λ")
    end
    xlabel!(xlabel)
    ylabel!(ylabel)
    title!(title)
    # display(plot!())
    savefig("results/$(title).png")
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


# create dir results if not exists
if !isdir("results")
    mkdir("results")
end

"""run experiments"""
game = bee_queen()
method = "minmax"

α = 0.1
β = 1e-4
J_dict_of_lists, w_dict_of_lists = experiment(game, α, β, [0.3, 0.5, 0.7], 2000, method)


"""plot and save results"""
plot_J(J_dict_of_lists; iter_cap=2000, title="$(game.name)_$(method)")
save_pickle(J_dict_of_lists, w_dict_of_lists, "results/$(game.name)_$(method).pkl")
