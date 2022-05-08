using Plots
using Zygote, ChainRulesCore
include("game_solvers/entropy_nash_solver.jl")
include("gradient_entropy_nash.jl")
include("game_solvers/bimatrix_mixed_nash_solver.jl")

# E[V] on w
function expectedValue(u, phi, w, V)
    m, n, N = size(u)
    u_tilde = u + [ (phi[k,:,:] * w)' * u[i,j,:] for i in 1:m, j in 1:n, k in 1:N]

    if args["nash_type"] == "entropy_nash"
        solver = EntropySolver()
        x, y, info = solve_entropy_nash(solver, u_tilde, actions)
    elseif args["nash_type"] == "nash"
        solver = LemkeHowsonGameSolver()
        x, y, info, _ = solve_mixed_nash(solver, u_tilde[:,:,1], u_tilde[:,:,2])
    end

    return x' * V * y
end

# Surface plot
function plot_surface(u, phi, V, output_name)
    w1 = range(0, stop=1, length=10)
    w2 = range(0, stop=1, length=10)
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
function plot_heatmap(u, actions, phi, V, output_name, c)
    x = range(0, stop=1, length=1000)  # w1
    y = range(0, stop=1, length=1000)  # w2

    # plot heatmap
    g(x, y) = expectedValue(u, phi, [x,y], V)

    xlim = (-0, 1)
    ylim = (-0, 1)
    xs = range(xlim...; length=200)
    ys = range(ylim...; length=200)

    h = heatmap(xs, ys, g)

    # plot gradient if needed
    if args["plot-gradient"] == true
        f(x, y) = evaluate(u, actions, phi, [x,y], V)

        fdot(x, y) = gradient(f, x, y)
        ∂xf(x, y) = fdot(x, y)[1]
        ∂yf(x, y) = fdot(x, y)[2]

        # c = 0.05 # for visual scaling of the quivers (GRADIENTS ARE MASSIVE!!!? WHY :( )
        x = range(-0, 1; length=11)
        y = range(-0, 1; length=11)
        X, Y = reim(complex.(x', y)) # meshgrid
        S, T = c*∂xf.(x', y), c*∂yf.(x', y)

        quiver!(vec(X-S/2), vec(Y-T/2); quiver=(vec(S), vec(T)), color=:cyan)
    end

    plot(h; xlim, ylim, xlabel = "w1", ylabel = "w2",size=(450, 400))

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
function plotting(s, plot_type)
    if plot_type == "heatmap"
        println("Plotting heatmap for {$(s.name)}")
        plot_heatmap(s.u, s.actions, s.phi, s.V, s.name, 1)
    elseif plot_type == "surface"
        println("Plotting surface for {$(s.name)}")
        plot_surface(s.u, s.phi, s.V, s.name)
    end
end

# ************* EXAMPLES *************
# prisoner's dilemma
function prisoner()
    name = "prisoner"
    u = [[1 3; 0 2];;;[1 0; 3 2]]
    actions = [["C", "D"], ["C", "D"]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    V = [-1 0; 0 0]
    (; name=name, u=u, actions=actions, phi=phi, V=V)
end 

# 2-route traffic
function traffic2()
    name = "traffic2"
    u = [[2 4; 1.5 3];;;[2 1.5; 4 3]]
    actions = [["A", "B"], ["A", "B"]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    # V = sum(u[:,:,i] for i in 1:2)
    V = [-1 0; 0 0]
    (; name=name, u=u, actions=actions, phi=phi, V=V)
end

# 3-route traffic
function traffic3()
    name = "traffic3"
    u = [[2 4 4; 1.5 3 1.5; 2.5 2.5 5];;;[2 1.5 2.5; 4 3 2.5; 4 1.5 5]] 
    actions = [["A", "B", "C"], ["A", "B", "C"]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    # V = sum(u[:,:,i] for i in 1:2)
    V = [-1 0 0; 0 0 0; 0 0 0]
    (; name=name, u=u, actions=actions, phi=phi, V=V)
end

## 3 PLAYER EXAMPLE: Clean or Pollute
# u1 = [[1 1; 0 3];;;[1 4; 3 3]]
# u2 = [[1 0; 1 3];;;[1 3; 4 3]]
# u3 = [[1 1; 1 4];;;[0 3; 3 3]]
# u = [u1, u2, u3]
# d = [2,2,2]; N = 3; cost_tensors = [ randn(d...) for i = 1:N]; # random N-player matrix generation

# ************* PLOTTING *************
plotting(traffic3(), "surface")
plotting(traffic3(), "heatmap")