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

# N-player (<=3), M-route traffic
function playerN_trafficM(N, M)
    name = "player"* string(N) * "_traffic" * string(M)
    u = generate_traffic(N, [M for i in 1:N])
    A = [[i for i in 1:M] for j=1:N]

    # phi_individual
    phi_individual = Matrix{Int64}[]
    for i in 1:N, j in 1:N
        phi = zeros(Int64, N,N)
        if i != j
            phi[i,j] = 1
            push!(phi_individual, phi)
        end
    end
    # phi_individual = phi_individual[1:3]

    # phi_all_people
    phi_all_people = Matrix{Int64}[]
    for i in 1:N
        phi = zeros(Int64, N,N)
        for j in 1:N
            if i != j
                phi[i,j] = 1
            end
        end
        push!(phi_all_people, phi)
    end

    # phi_reciprocity
    phi_reciprocity = Matrix{Int64}[]
    for i in 1:N, j in 1:N
        phi = zeros(Int64, N,N)
        if i != j
            phi[i,j] = 1
            phi[j,i] = 1
            if !(phi in phi_reciprocity)    
                push!(phi_reciprocity, phi)
            end
        end
    end


    # V
    V = sum(u[i] for i in 1:N)

    (; name=name, u=u, A=A, phi=phi_individual[1:2], V=V)
end

# ************* PLOTTING *************
#plotting(playerN_trafficM(3,2), "surface", c=0.5, axis_length=10)
plotting(prisoner(), "heatmap", c=0.5, axis_length=1)
