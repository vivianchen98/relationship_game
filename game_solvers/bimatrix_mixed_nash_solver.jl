
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

function pivot!(T, row, col)
    pivot = T[row, :] / T[row, col]
    T .-= T[:, col] * pivot'
    T[row, :] = pivot'
    nothing
end
