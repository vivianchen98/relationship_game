using ChainRulesCore
using Tullio
using LinearAlgebra
using Zygote
using Plots
using TensorOperations

# gradient-based approach to max V wrt w

"""
Implements Lemke-Howson pivoting method for bimatrix games.
"""
Base.@kwdef struct LemkeHowsonGameSolver
    "The maximum number of pivots allowed."
    max_pivots::Int = 1000
end

"""
Implements Lemke-Howson pivoting method for Bimatrix games.
Inputs and outputs as defined for the solver interface (see solve_interface.jl).

With special info object:

Inputs:

Returns:
- x: mixed eq strategies for player 1
- y: mixed eq strategies for player 2
- V1: eq game value for player 1 (Note: A is *cost* matrix for P1)
- V2: eq game value for player 2 (Note: B is *cost* matrix for P2)
- info: additional solver info as named tuple of
    - pivots: number of pivots performed to find solution
    - ray_term: true if method exited prematurely with a ray termination
"""
function solve_mixed_nash(solver::LemkeHowsonGameSolver, A, B; ϵ = 0.0)
    @assert size(A) == size(B)

    A_pos = A .+ (1.0 - minimum(A))
    B_pos = B .+ (1.0 - minimum(B))

    m, n = size(A)
    @assert ϵ < 1.0 / m
    @assert ϵ < 1.0 / n

    x_factor = 1.0/(1.0-m*ϵ)
    y_factor = 1.0/(1.0-n*ϵ)

    rhs = [-ones(m) * x_factor; -ones(n) * y_factor]
    T = [I(m + n) [zeros(m, m) -A_pos; -B_pos' zeros(n, n)] rhs]

    basis = 1:(m + n) |> collect

    r = argmin(B_pos'[:, 1])
    pivot!(T, m + r, m + n + 1)
    basis[m + r] = m + n + 1

    s = argmin(A_pos[:, r])
    pivot!(T, s, m + n + m + r)
    basis[s] = m + n + m + r

    entering = n + m + s

    pivots = 0
    ray_term = false
    max_iters = false
    while (1 in basis) && (m + n + 1 in basis)
        d = T[:, entering]
        wrong_dir = d .<= 0
        ratios = T[:, end] ./ d
        ratios[wrong_dir] .= Inf
        t = argmin(ratios)
        if !all(wrong_dir)
            pivot!(T, t, entering)
            exiting = basis[t]
            basis[t] = entering
            if exiting > m + n
                entering = exiting - m - n
            else
                entering = exiting + m + n
            end
        else
            ray_term = true
            break
        end
        pivots += 1
        if pivots >= solver.max_pivots
            max_iters = true
            break
        end
    end

    vars = zeros(2 * (m + n))
    vars[basis] = T[:, end]

    u = vars[1:m]
    v = vars[(m + 1):(m + n)]
    x = vars[(m + n + 1):(m + n + m)]
    y = vars[(m + n + m + 1):(m + n + m + n)]

    x_sum = sum(x)
    y_sum = sum(y)

    x_normalized = (x ./ (x_sum * x_factor)) .+ ϵ
    y_normalized = (y ./ (y_sum * y_factor)) .+ ϵ

    #    V1 = x_normalized' * A * y_normalized
    #    V2 = x_normalized' * B * y_normalized

    (;
        x = x_normalized,
        y = y_normalized,
        #        V1,
        #        V2,
        info = (; pivots, ray_term, max_iters),
        _grad_info = (; A_pos, B_pos, x_sum, y_sum, x_factor, y_factor, x, y),
    )
end

"""
- grads: if gradients computed a named tuple of the following (nothing otherweise):
    - dA: derivative tensor: `dA[i,j,k] = \\frac{ \\partial y_i }{ \\partial A_{j,k} }`
    - dB: derivative tensor: `dB[i,j,k] = \\frac{ \\partial x_i }{ \\partial B'_{j,k} }`
Note:
- Derivatives of x w.r.t. A are 0
- Derivatives of y w.r.t. B are 0
- Temporarily ignoring case when solution is non-isolated
- Temporarily ignoring case when solution is isolated but only directionally differentiable
"""
function ChainRulesCore.rrule(::typeof(solve_mixed_nash), solver::LemkeHowsonGameSolver, A, B; ϵ = 0.0)
    res = solve_mixed_nash(solver, A, B; ϵ)

    function solve_mixed_nash_pullback(∂res)
        m, n = size(A)
        (; A_pos, B_pos, x_sum, y_sum, x_factor, y_factor, x, y) = res._grad_info

        ∂self = NoTangent()
        ∂solver = NoTangent()

        x_pos = x .> 1e-6
        y_pos = y .> 1e-6
        is_reduced_game_square = count(!iszero, x_pos) == count(!iszero, y_pos)

        if !is_reduced_game_square
            ∂A = zero(A)
            ∂B = zero(B)
        else
            # einsum macro for sane notation of tensor derivatives
            ∂A = @thunk maybe_trivial_tangent(∂res.y) do
                d_y_norm = zeros(n, n)
                for i in 1:n
                    d_y_norm[i, i] = 1.0 / (y_sum)
                    d_y_norm[i, :] .-= y[i] / (y_sum * y_sum)
                end
                d_y_norm /= y_factor
                dA = zeros(n, m, n)
                A_reduced = A_pos[x_pos, y_pos]
                Ari = inv(A_reduced)
                for k in 1:n
                    dA[y_pos, x_pos, k] = -Ari * y[k]
                    dA[:, :, k] = d_y_norm * dA[:, :, k]
                end
                @tullio ∂A[j, k] := dA[i, j, k] * ∂res.y[i]
            end
            ∂B = @thunk maybe_trivial_tangent(∂res.x) do
                d_x_norm = zeros(m, m)
                for i in 1:m
                    d_x_norm[i, i] = 1.0 / x_sum
                    d_x_norm[i, :] .-= x[i] / (x_sum * x_sum)
                end
                d_x_norm /= x_factor
                dB = zeros(m, n, m)
                B_reduced = B_pos'[y_pos, x_pos]
                Bri = inv(B_reduced)
                for k in 1:m
                    dB[x_pos, y_pos, k] = -Bri * x[k]
                    dB[:, :, k] = d_x_norm * dB[:, :, k]
                end
                @tullio ∂B[j, k] := dB[i, k, j] * ∂res.x[i]
            end
        end

        ∂self, ∂solver, ∂A, ∂B
    end

    res, solve_mixed_nash_pullback
end

"""
Checks if `upstream_tangent` is ZeroTangent or NoTangent. If so, we can skip the computation of f
and just forward the tangent type
"""
maybe_trivial_tangent(f, upstream_tangent)

function maybe_trivial_tangent(_, upstream_tangent::Union{ZeroTangent,NoTangent})
    upstream_tangent
end

function maybe_trivial_tangent(f, _)
    f()
end

function pivot!(T, row, col)
    pivot = T[row, :] / T[row, col]
    T .-= T[:, col] * pivot'
    T[row, :] = pivot'
    nothing
end

# solver = LemkeHowsonGameSolver()
# A = [2 4; 1.5 3]
# B = [2 1.5; 4 3]
# x, y, info, _grad_info = solve_mixed_nash(solver, A, B)
# V = [3 2; 2 1]


# Given problem
N = 2
A = [[1,2], [1,2]]
# u = [[2 4; 1.5 3], [2 1.5; 4 3]]
# traffic_3_A = [2 4 4; 1.5 3 1.5; 2.5 2.5 5]
# traffic_3_B = [2 1.5 2.5; 4 3 2.5; 4 1.5 5]

rps_A = [0 -1 1; 1 0 -1; -1 1 0]
rps_B = -rps_A
u = [rps_A;;; rps_B]
V = [1 -1 -1; -1 1 -1; -1 -1 1]
# phi = [[0 1; 1 0], [0 1; 0 0], [0 0; 1 0]]
phi = [[0 1; 2 0];;;[0 0; 1 0]]
k = size(phi)[3]

# init w
w = [.3, .7]

function modify_u(u, w, phi)
    w_phi = sum(w[i]*phi[i] for i in eachindex(w))
    u_tilde = u + w_phi * u
    return u_tilde
end

function ChainRulesCore.rrule(::typeof(modify_u), u, w, phi)
    u_tilde = modify_u(u, w, phi)
    
    function modify_u_pullback(∂u_tilde)
        j_max, k_max, i_max = size(u_tilde)

        ∂self = NoTangent()
        ∂u = NoTangent()
        sum = zeros(size(phi)[3])
        for i in 1:i_max, j in 1:j_max, k in 1:k_max
            sum += ∂u_tilde[j,k,i] * phi[i,:,:]' * u[j,k,:]
        end
        ∂w = sum
        ∂phi = NoTangent()

    ∂self, ∂u, ∂w, ∂phi
    end

    return u_tilde, modify_u_pullback
end

function expectedValue(w)
    u_tilde = modify_u(u, w, phi)

    solver = LemkeHowsonGameSolver()
    x, y, info, _grad_info = solve_mixed_nash(solver, u_tilde[:,:,1], u_tilde[:,:,2], ϵ=0.2)
    return x' * V * y
end

function computeV(A, B)
    # u_tilde = modify_u(u, w, phi)

    solver = LemkeHowsonGameSolver()
    x, y, info, _grad_info = solve_mixed_nash(solver, A, B, ϵ=0.2)
    return x' * V * y
end

gr()

w1 = 0:0.01:2
w2 = 0:0.01:2

w1s = [x for x in w1 for y in w2]
w2s = [y for x in w1 for y in w2]

grad = gradient(computeV, rps_A, rps_B)
print(grad)

# f(w1, w2) = gradient(expectedValue, [w1, w2])[1]
# p = quiver(w1s, w2s, quiver=f)
# display(plot(p))


# grad = gradient(expectedValue, w)
# print(grad)
