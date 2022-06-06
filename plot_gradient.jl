include("visualize_entropy_nash_general.jl")
include("trafficN.jl")


# ************* EXAMPLES *************
# prisoner's dilemma
function prisoner()
    name = "prisoner"
    u = [[1 3; 0 2];;;[1 0; 3 2]]
    # actions = [["C", "D"], ["C", "D"]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    V = [0 -1; 0 0]
    (; name=name, u=u, phi=phi, V=V)
end 

# M-route traffic
function trafficM(M)
    name = "traffic" * string(M)
    u = generate_traffic(2, [M,M]); u = [u[1];;;u[2]]
    phi = [[0 1; 0 0];;;[0 0; 1 0]]
    V = zeros(M,M); V[2,1] = -1; V[1,1] = -1
    # V = sum(u[:,:,i] for i in 1:2)
    (; name=name, u=u, phi=phi, V=V)
end

## 3 PLAYER EXAMPLE: Clean or Pollute
# u1 = [[1 1; 0 3];;;[1 4; 3 3]]
# u2 = [[1 0; 1 3];;;[1 3; 4 3]]
# u3 = [[1 1; 1 4];;;[0 3; 3 3]]
# u = [u1, u2, u3]
# d = [2,2,2]; N = 3; cost_tensors = [ randn(d...) for i = 1:N]; # random N-player matrix generation


# N-player (<=4), M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    N = N
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]
    if N == 2
        phi = [[0 1; 0 0],[0 0; 1 0]]
    elseif N == 3
        # phi = [[0 1 0; 0 0 0; 0 0 0], [0 0 1; 0 0 0; 0 0 0], [0 0 0; 1 0 0; 0 0 0], [0 0 0; 0 0 1; 0 0 0], [0 0 0; 0 0 0; 1 0 0], [0 0 0; 0 0 0; 0 1 0]]
        phi = [[0 1 0; 0 0 0; 0 0 0], [0 0 0; 1 0 0; 0 0 0]]
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

# ************* PLOTTING *************
plotting(playerN_trafficM(2,2), "surface", c=10, axis_length=10)
# plotting(trafficM(3), "heatmap", c=10, axis_length=10)