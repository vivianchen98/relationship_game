include("../examples.jl")
include("../game_solvers/gradient_entropy_nash_jump_new.jl")
using Plots
using DataFrames, CSV
using PyCall
py"""
import pickle
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data1 = pickle.load(f)
        data2 = pickle.load(f)
    return data1, data2
def save_pickle(data1, data2, data3, fpath):
    with open(fpath, "wb") as f:
        pickle.dump(data1, f)
        pickle.dump(data2, f)
        pickle.dump(data3, f)
"""
load_pickle = py"load_pickle"
save_pickle = py"save_pickle"

# Check if at least one argument is provided
if length(ARGS) < 1
    println("Usage: julia script.jl [game_name]")
    exit()
end

# Access the first argument
game_name = ARGS[1]

# check if game_name is either congestion or bee_queen
if !(game_name in ["congestion", "bee_queen"])
    println("Error: game_name not found in examples.jl")
    exit()
end

function unique_with_tolerance(values, tol::Float64)
    unique_vals = []
    for val in values
        if all(abs(val - unique_val) > tol for unique_val in unique_vals)
            push!(unique_vals, val)
        end
    end
    return unique_vals
end

function sample_J_at(game, w, λ, tol, axis)
    data = []

    # record J for 100 different seeds
    J_list = []
    for se in 1:300
        J = evaluate(game.u, game.V, game.phi, w, λ, se)
        push!(J_list, J)
    end
    J_approx_set = unique_with_tolerance(J_list, tol)  # find unique values with tolerance 1e-5

    # save tuple to data
    for eachj in J_approx_set
        push!(data, (w[axis], eachj))
    end

    return data
end

function sample_on_axis(game, axis, λ; axis_range=0:0.05:1, tol=1e-5)
    dataset = []
    dataset_max = []
    dataset_min = []

    for w_axis in axis_range
        w[axis] = w_axis

        data = sample_J_at(game, w, λ, tol, axis)
        @show data
        push!(dataset, data...)

        # find max J
        max_J = maximum([point[2] for point in data])
        push!(dataset_max, (w_axis, max_J))

        # find min J
        min_J = minimum([point[2] for point in data])
        push!(dataset_min, (w_axis, min_J))
        
    end
    return dataset, dataset_max, dataset_min
end

# set game based on command line argument
if game_name == "congestion"
    game = congestion()
elseif game_name == "bee_queen"
    game = bee_queen()
end

# set parameters
name, u, V, phi = game
K = length(phi)
w = [1/sqrt(K) for i in 1:K]

# create dir if not exist
if !isdir("$(game.name)")
    mkdir("$(game.name)")
end

# sample on each axis and plot
for λ in [1.0], axis in [3]
    # usage: collect dataset
    dataset, dataset_max, dataset_min = sample_on_axis(game, axis, λ; axis_range=0:0.01:3, tol=1e-4)

    # plot scatter
    x_values = [point[1] for point in dataset]
    y_values = [point[2] for point in dataset]
    scatter(x_values, y_values, xlabel="w_axis", ylabel="J of equilibriums", 
            marker=:circle, markercolor = :black, markersize = 1.5)
    savefig("$(game.name)/w$(axis)_lambda$(λ).png")
    save_pickle(dataset, dataset_max, dataset_min, "$(game.name)/w$(axis)_lambda$(λ).pkl")
end