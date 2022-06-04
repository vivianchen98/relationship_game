include("game_solvers/entropy_nash_solver_general.jl")
include("gradient_entropy_nash_general.jl")
include("trafficN.jl")

# N-player (<=3), M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    N = N
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]
    if N == 2
        phi = [[0 1; 0 0],[0 0; 1 0]]
    elseif N == 3
        phi = [[0 1 0; 0 0 0; 0 0 0], [0 0 1; 0 0 0; 0 0 0], [0 0 0; 1 0 0; 0 0 0], [0 0 0; 0 0 1; 0 0 0], [0 0 0; 0 0 0; 1 0 0], [0 0 0; 0 0 0; 0 1 0]]
    end
    V = zeros([M for i in 1:N]...)
    if N == 2
        V[1,1] = -1
    elseif N == 3
        V[1,1,1] = -1
    end
    # V = sum(u[i] for i in 1:N)
    @show V
    (; name=name, N=N, u=u, A=A, phi=phi, V=V)
end

# create traffic example
name, N, u, A, phi, V = playerN_trafficM(3,2)
k = length(phi)

x, info = solve_entropy_nash_general(EntropySolver(), u)
@show x
@show info.total_iter

# w = ones(length(phi))
w = [0, 1, 0, 1, 0, 1]
val = evaluate(u, phi, w, V)
@show val

@show gradient(evaluate, u, phi, w, V)
