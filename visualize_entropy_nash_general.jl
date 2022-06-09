using Plots
using Zygote, ChainRulesCore
using TensorGames
include("game_solvers/entropy_nash_solver_general.jl")
include("gradient_entropy_nash_general.jl")
include("trafficN.jl")

# E[V] on w
function expectedValue(u, phi, w, V)
    u_tilde = create_u_tilde(u, phi, w)

    if args["nash_type"] == "entropy_nash"
        x, info = solve_entropy_nash_general(EntropySolver(), u_tilde)
    elseif args["nash_type"] == "nash"
        x = compute_equilibrium(u_tilde)
    end

    cost = V .* prob_prod(x, [s for s in 1:length(u)], CartesianIndices(V))
    return sum(cost)
end

# Surface plot
function plot_surface(u, phi, V, output_name, axis_length)
    # axis_length = 10
    w1 = range(-axis_length, stop=axis_length, length=10)
    w2 = range(-axis_length, stop=axis_length , length=10)
    f(w1, w2) = expectedValue(u, phi, [w1, w2], V)
    s = surface(w1, w2, f, camera=(30,40))
    plot(s)
    
    output_path = ""
    if args["nash_type"] == "nash"
        output_path = "general_results/$(output_name)_$(args["nash_type"])_surface.png"
    elseif args["nash_type"] == "entropy_nash"
        output_path = "general_results/$(output_name)_$(args["nash_type"])_λ=$(args["lambda"])_surface.png"
    end

    savefig(output_path)
    println("Surface plot saved to '$(output_path)'")
    println()
end

# Gradient-Heatmap
function plot_heatmap(u, phi, V, output_name, c, axis_length)
    # axis_length = 10
    x = range(-axis_length, stop=axis_length, length=1000)  # w1
    y = range(-axis_length, stop=axis_length, length=1000)  # w2

    # plot heatmap
    g(x, y) = expectedValue(u, phi, [x,y], V)

    xlim = (-axis_length, axis_length)
    ylim = (-axis_length, axis_length)
    xs = range(xlim...; length=200)
    ys = range(ylim...; length=200)

    h = heatmap(xs, ys, g)

    # plot gradient if needed
    if args["plot-gradient"] == true
        f(x, y) = evaluate(u, phi, [x,y], V)

        fdot(x, y) = gradient(f, x, y)
        ∂xf(x, y) = fdot(x, y)[1]
        ∂yf(x, y) = fdot(x, y)[2]

        # c = 0.05 # for visual scaling of the quivers (GRADIENTS ARE MASSIVE!!!? WHY :( )
        x = range(-axis_length, axis_length; length=11)
        y = range(-axis_length, axis_length; length=11)
        X, Y = reim(complex.(x', y)) # meshgrid
        S, T = c*∂xf.(x', y), c*∂yf.(x', y)

        quiver!(vec(X-S/2), vec(Y-T/2); quiver=(vec(S), vec(T)), color=:cyan)
    end

    plot(h; xlim, ylim, xlabel = "w1", ylabel = "w2", title="c=$c", size=(450, 400))

    output_path = ""
    if args["nash_type"] == "nash"
        output_path = "general_results/$(output_name)_$(args["nash_type"])_heatmap.png"
    elseif args["nash_type"] == "entropy_nash"
        output_path = "general_results/$(output_name)_$(args["nash_type"])_λ=$(args["lambda"])_heatmap.png"
    end
    
    savefig(output_path)
    println("Heatmap plot saved to '$(output_path)'")
    println()
end

# General plotting
function plotting(s, plot_type; c=10, axis_length)
    if plot_type == "heatmap"
        println("Plotting heatmap for {$(s.name)}")
        plot_heatmap(s.u, s.phi, s.V, s.name, c, axis_length)
    elseif plot_type == "surface"
        println("Plotting surface for {$(s.name)}")
        plot_surface(s.u, s.phi, s.V, s.name, axis_length)
    end
end