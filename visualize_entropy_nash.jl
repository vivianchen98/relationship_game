using Plots
using Zygote, ChainRulesCore
include("game_solvers/entropy_nash_solver_general.jl")
include("gradient_entropy_nash_general.jl")
include("game_solvers/bimatrix_mixed_nash_solver.jl")
include("trafficN.jl")

# E[V] on w
function expectedValue(u, phi, w, V)
    if args["nash_type"] == "entropy_nash"
        solver = EntropySolver()
        return evaluate(u, phi, w, V)
    elseif args["nash_type"] == "nash" ## 2-players only
        m, n, N = size(u)
        u_tilde = u + [ (phi[k,:,:] * w)' * u[i,j,:] for i in 1:m, j in 1:n, k in 1:N]
        solver = LemkeHowsonGameSolver()
        x, y, info, _ = solve_mixed_nash(solver, u_tilde[:,:,1], u_tilde[:,:,2])
        return x' * V * y
    end
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
        output_path = "clean_results/$(output_name)_$(args["nash_type"])_surface.png"
    elseif args["nash_type"] == "entropy_nash"
        output_path = "clean_results/$(output_name)_$(args["nash_type"])_λ=$(args["lambda"])_surface.png"
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
        output_path = "clean_results/$(output_name)_$(args["nash_type"])_heatmap.png"
    elseif args["nash_type"] == "entropy_nash"
        output_path = "clean_results/$(output_name)_$(args["nash_type"])_λ=$(args["lambda"])_heatmap.png"
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
