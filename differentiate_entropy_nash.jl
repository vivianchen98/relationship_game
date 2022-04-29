using Plots
using Zygote
using ChainRulesCore
include("entropy_nash_solver.jl")

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

# evaluate effect of w on E[V]
V = [3 2; 2 1]
phi = [[0 1; 1 0], [0 1; 0 0], [0 0; 1 0]]

function expectedValue(w)
    u_tilde = modify_u(u, w, phi)

    solver = EntropySolver()
    x, y, proper_termination, max_iter = solve_entropy_nash(solver, u_tilde, actions)
    return x' * V * y
end

gr()

w1 = range(0, stop=1, length=1000)
w2 = range(0, stop=1, length=1000)
f(w1, w2) = expectedValue([w1, w2])
p = surface(w1, w2, f)
display(plot(p))