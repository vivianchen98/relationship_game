include("visualize_entropy_nash.jl")
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

# 3-player M-route traffic
function traffic3M(M)
    name = "traffic3player" * string(M)
    u = generate_traffic(3, [M,M,M]);
    phi = [[0 0 1; 1 0 0; 0 1 0],[0 1 0; 0 0 1; 1 0 0]]
    V = zeros(M,M,M); V[2,1,1] = -1; V[1,1,1] = -1
    # V = sum(u[:,:,i] for i in 1:2)
    (; name=name, u=u, phi=phi, V=V)
end

## 3 PLAYER EXAMPLE: Clean or Pollute
# u1 = [[1 1; 0 3];;;[1 4; 3 3]]
# u2 = [[1 0; 1 3];;;[1 3; 4 3]]
# u3 = [[1 1; 1 4];;;[0 3; 3 3]]
# u = [u1, u2, u3]
# d = [2,2,2]; N = 3; cost_tensors = [ randn(d...) for i = 1:N]; # random N-player matrix generation

# ************* PLOTTING *************
# plotting(trafficM(3), "surface", c=10, axis_length=10)
# plotting(trafficM(3), "heatmap", c=10, axis_length=10)
plotting(traffic3M(3), "surface", c=10, axis_length=10)
plotting(traffic3M(3), "heatmap", c=10, axis_length=10)
