using Zygote, ChainRulesCore
include("entropy_nash_solver_jump.jl")

function create_u_tilde(u, phi, w)
    N = length(u)
    w_phi = w' * phi
    # u_tilde = u + [sum(w_phi[n,i] * u[i] for i in 1:N) for n in 1:N]
    u_tilde = [sum(w_phi[n,i] * u[i] for i in 1:N) for n in 1:N]
    return u_tilde
end

function prob_prod_jump(x, player_indices, cartesian_indices)
    return [prod(x[i, idx[i]] for i in player_indices) for idx in cartesian_indices]
end

function strategy_cost(x, V)
    cost = V .* prob_prod_jump(x, 1:size(x,1), CartesianIndices(V))
    return sum(cost)
end

function solve_relationship_game(u, phi, w, λ, seed)
    u_tilde = create_u_tilde(u, phi, w)
    x = solve_entropy_nash_jump(u_tilde, λ; seed=seed)
    return x
end

function evaluate(u, V, phi, w, λ, seed)
    x = solve_relationship_game(u, phi, w, λ, seed)
    return strategy_cost(x, V)
end

# stable softmax
function softmax(arr; θ=1)
    e = exp.((arr .- maximum(arr)) * θ)
    return e./ sum(e)
end

# jacobian matrix for a softmax distribution (given s = softmax(x))
function softmax_jacobian(s)
    len_s = length(s)
    J_s = zeros(len_s, len_s)
    for i in 1:len_s, j in 1:len_s
        J_s[i,j] = (i == j) ? (s[i] * (1-s[i])) : (-s[i] * s[j])
    end
    return J_s
end

# Derivative of u_tilde (modified utilities) wrt w (tensor form)
function u_tilde_i_grad_w(u, phi, i)
    return cat([sum(phi_r[i,j] * u[j] for j in eachindex(u)) for phi_r in phi]..., dims=length(u) + 1)
end

# Jacobian of h^i wrt w
function J_h_i_wrt_w(u, phi, x, i)
    J_u_tilde = u_tilde_i_grad_w(u, phi, i)
    out = h_except_i(i, x, J_u_tilde)
    return reshape(out, (length(x[i, :]), length(phi)))
end

# Jacobian of softmax(-h^i(x,w)/λ) wrt w
function J_s_i_wrt_w(u, phi, x, λ, i)
    s_i = softmax(-h_except_i(i, x, u[i]) ./ λ)
    return - softmax_jacobian(s_i) * J_h_i_wrt_w(u, phi, x, i) ./ λ
end

function g(i, j, u, x)
    n = size(x)[1]
    if n == 2
        if i < j
            return u[i]
        else
            return u[i]'
        end
    end
    list_without_i_j = [s for s in 1:n]; deleteat!(list_without_i_j, sort([i,j]))
    pt_cost = u[i] .* prob_prod_jump(x, list_without_i_j, CartesianIndices(u[i]))
    out = sum(pt_cost, dims=list_without_i_j)
    out = permutedims(out, (i,j,list_without_i_j...))
    out = reshape(out, (length(x[i, :]), length(x[j, :])))
    return out
end

function h_except_i(i, x, u_i)
    N = size(x, 1)
    full_list_except_i = [s for s in 1:N if s != i]

    # Calculates costs across all action distributions except for player i, then sums across dims other than player i
    costs = sum(u_i .* prob_prod_jump(x, full_list_except_i, CartesianIndices(u_i)), dims=full_list_except_i)
    return vcat(costs...) #We need to flatten the resulting 1D tensor into an array
end

function J_except_i(i, x, u)
    return h_except_i(i, x, u[i])
end

# TO DO: pullback of strategy_cost, solve_entropy_nash_jump, create_u_tilde
# => or make sure strategy_cost and create_u_tilde are compatible with Zygote auto-diff
# then do gradient(evaluate, u, V, phi, w, λ)[4] as ∂w

function ChainRulesCore.rrule(::typeof(strategy_cost), x, V)
    J = strategy_cost(x, V)
    function strategy_cost_pullback(∂J)
        # placeholder for unused partial derivatives ∂V
        ∂self = NoTangent()
        ∂V = NoTangent()

        # ∂x := the partial derivative operator wrt x (∂/∂x)
        # ∂J := the partial derivative operator wrt J (∂/∂J)
        # ∂x = ∂J/∂x * ∂/∂J = [gradient to find] * ∂J
        N = size(x)[1]
        ∂x = vcat([h_except_i(i, x, V)' * ∂J for i in 1:N]...)

        ∂self, ∂x, ∂V
    end
    J, strategy_cost_pullback
end

function ChainRulesCore.rrule(::typeof(solve_relationship_game), u, phi, w, λ, seed)
    res = solve_relationship_game(u, phi, w, λ, seed)

    function solve_relationship_game_pullback(∂res)
        x = res

        # flatten x as a vector z
        ∂z = collect(Iterators.flatten(∂res))

        # placeholder for unused partial derivatives ∂u, ∂phi
        ∂self = NoTangent()
        ∂u = NoTangent()
        ∂phi = NoTangent()
        ∂λ = NoTangent()
        ∂seed = NoTangent()

        # J_F_wrt_z = ∂z/∂z - ∂s/∂z
        N = length(u)
        # N = size(res, 1)
        J_submatrix(i,j) = (i == j) ? (zeros((size(u[i])[i], size(u[i])[i]))) : (softmax_jacobian(x[j,:]) * (-g(i,j,u,x) ./ λ)) #(softmax_jacobian(x[j]) * (-g(i,j,u,x) ./ λ))
        J_softmax = []
        for i in 1:N
            row = []
            for j in 1:N
                if isempty(row)
                    row = J_submatrix(i,j)
                else
                    row = hcat(row, J_submatrix(i,j))
                end
            end
            if isempty(J_softmax)
                J_softmax = row
            else
                J_softmax = vcat(J_softmax, row)
            end
        end
        # CHECK FOR CORRECT SIZE
        m, n = size(J_softmax)
        total_player_times_action = sum(size(u[i])[i] for i in 1:N)
        @assert m == total_player_times_action && n == total_player_times_action

        # J_F_wrt_z = I .- J_softmax
        J_F_wrt_z = I - J_softmax

        # J_F_wrt_w = - ∂s/∂w
        J_F_wrt_w = - vcat([J_s_i_wrt_w(u, phi, x, λ, i) for i in 1:N]...)

        # ∂z/∂w = - J_F_wrt_z \ J_F_wrt_w
        ∂w = - (∂z' * (J_F_wrt_z\ J_F_wrt_w))'

        ∂self, ∂u, ∂phi, ∂w, ∂λ, ∂seed
    end

    res, solve_relationship_game_pullback
end

function l2_project(w)
    return w / norm(w)
end

# projected GD
function GradientDescent(g, α, max_iter, λ, β)
    w_list = Vector{Vector{Float64}}()
    J_list = Vector{Float64}()
    terminate_step = 0

    # init stepsize
    stepsize = α

    # init seed
    seed = 0

    # init w
    K = length(g.phi)
    # w = [1/sqrt(K) for i in 1:K] # unifrom distribution of length K
    w = rand(K) # random distribution of length K
    push!(w_list, w)
    println("start with w=($w)")

    previous_J = evaluate(g.u, g.V, g.phi, w, λ, -1)
    push!(J_list, previous_J)

    for i in 1:max_iter
        # print result in intervals of 100
        if i % 100 == 0
            println("step $(i): $w")
            @show previous_J
            println()
        end

        # # set seed to be the iteration number
        # seed = i

        # compute candidate next w
        ∂w = gradient(evaluate, g.u, g.V, g.phi, w, λ, seed)[4]
        w_candidate = l2_project(w - stepsize .* ∂w)   # project onto the probability simplex

        # update current_J and test if J is decreasing
        current_J = evaluate(g.u, g.V, g.phi, w_candidate, λ, seed)
        # @show current_J
        if current_J > previous_J
            seed = i + 1    # update seed
            continue
        else
            println("update w!")
            w = w_candidate
            push!(w_list, w)
            push!(J_list, current_J)
            previous_J = current_J
        end

        # stopping criteria
        if norm(∂w - (∂w' * w) / (norm(w)^2) * w)  < β # stopping criteria
            println("terminate with w=($w) in $(i) steps, with J=$(current_J)")
            terminate_step = i
            break
        end
        if i == max_iter
            println("Does not converge within ($max_iter) iterations: norm(∂w)=($(norm(∂w)))")
        end
    end

    (;  w = w, J = previous_J, 
        info = (terminate_step = terminate_step, J_list = J_list, w_list = w_list))
end