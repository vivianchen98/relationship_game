using LightGraphs
using Plots
# using GraphPlot

# adj_matrix = [0 1 1; 1 0 1; 1 1 0]  # Example adjacency matrix
# graph = SimpleGraph(adj_matrix)
# gplot(graph)
# plot(g, layout=GraphPlot.circular_layout, nodecolor=:blue)



# tensor = strategy_cost(x, V)
# x = []
# y = []
# z = []
# vals = []

# for i in 1:2, j in 1:2, k in 1:2
#     push!(x, i)
#     push!(y, j)
#     push!(z, k)
#     push!(vals, tensor[i,j,k])
# end

# scatter(x, y, z, zcolor=vals, markersize=10, color=:auto)


include("examples.jl")
include("game_solvers/gradient_entropy_nash_jump.jl")

u, V, phi = congestion()

f(w1, w2) = evaluate(u, V, phi, [w1, w2], 0.1)

# Generate x and y values
x = LinRange(-5, 5, 100)
y = LinRange(-5, 5, 100)

# Create a meshgrid for x and y values
X = [i for i in x, j in y]
Y = [j for i in x, j in y]

# Evaluate the function on the meshgrid
Z = f.(X, Y)

plot(X, Y, Z, seriestype=:surface, color=:auto, camera=(30,40))



# adja
