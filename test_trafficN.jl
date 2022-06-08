include("trafficN.jl")
using TensorGames

N_list = [2,3,4,5]
M_list = [2]

# N-player (<=3), M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    u = generate_traffic(N, [M for i in 1:N])

    (; name=name, u=u)
end

# call order_and_design
for N in N_list, M in M_list
    name, u = playerN_trafficM(N, M)

    @show name
    @show argmin(sum(u))
    @show compute_equilibrium(u).x
    println()
end