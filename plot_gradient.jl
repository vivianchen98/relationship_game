include("visualize_entropy_nash_general.jl")
include("trafficN.jl")

# ************* EXAMPLES *************
# prisoner's dilemma
function prisoner()
    name = "prisoner"
    u = [[1 3; 0 2],[1 0; 3 2]]
    # actions = [["C", "D"], ["C", "D"]]
    phi = [[0 1; 0 0],[0 0; 1 0]]
    V = [0 -1; 0 0]
    (; name=name, u=u, phi=phi, V=V)
end

# M-route traffic
function trafficM(M)
    name = "traffic" * string(M)
    u = generate_traffic(2, [M,M])#; u = [u[1];;;u[2]]
    phi = [[0 1; 0 0],[0 0; 1 0]]
    V = zeros(M,M); V[2,1] = -1; V[1,1] = -1
    # V = sum(u[:,:,i] for i in 1:2)
    (; name=name, u=u, phi=phi, V=V)
end

# N-player (<=4), M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    N = N
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]

    # phi
    phi_list = Matrix{Int64}[]
    for i in 1:N, j in 1:N
        phi = zeros(Int64, N,N)
        if i != j
            phi[i,j] = 1
            push!(phi_list, phi)
        end
    end
    phi_list = [phi_list[1], phi_list[3]]

    # V
    V = sum(u[i] for i in 1:N)

    (; name=name, N=N, u=u, A=A, phi=phi_list, V=V)
end

# ************* PLOTTING *************
# plotting(playerN_trafficM(3,2), "surface", c=1, axis_length=10)
plotting(playerN_trafficM(3,2), "heatmap", c=0.5, axis_length=10)
