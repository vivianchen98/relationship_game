function rps()
    N = 2; dim=[3,3];

    # create utilities u for each player
    u = [ 0 -1  1;
          1  0 -1;
         -1  1  0 ]
    u = [u , -u]
    
    return N, u
end

function pd()
    N = 2; dim=[2,2];

    # create utilities u for each player
    u = [1 3;
         0 2]
    u = [u , u']

    return N, u
end

# congestion game: two-lane roads of the same direction
function congestion()
    # 3 players on a two-path road
    N = 3; dim=[2,2,2];

    # road delay functions
    a(load) = load
    b(load) = 2*load
    func_list = [a,b]

    # create utilities u for each player
    u = [zeros(dim...) for i = 1:N]
    for i=1:2, j=1:2, k=1:2
        for n=1:N
            player = [i,j,k][n]
            load = count(x->(x==player), [i,j,k])
            u[n][i,j,k] = func_list[player](load)
        end
    end

    # social cost as the sum of utilities
    V = sum(u[i] for i in 1:N)

    # relationship structure imposed by reality
    phi = [ [1 0 0; 0 1 0; 0 0 1],  # each car's selfish utilities
            [0 1 1; 1 0 1; 1 1 0]]  # regular car to regular car
    
    (; name = "congestion", u = u, V = V, phi = phi, λ = 0.7)
end

function bee_queen()
    N = 4; dim=[2,2,2,2];

    # road delay functions
    a(load) = load
    b(load) = 2*load
    func_list = [a,b]

    # create utilities u for each player
    u = [zeros(dim...) for i = 1:N]
    for i=1:2, j=1:2, k=1:2, l=1:2
        for n=1:N
            player = [i,j,k,l][n]
            load = count(x->(x==player), [i,j,k,l])
            u[n][i,j,k,l] = func_list[player](load)
        end
    end

    # social cost as the weighted sum of utilities
    w = [8, 1, 1, 1]
    V = sum(w[i] * u[i] for i in 1:N)

    # relationship structure imposed by reality
    phi = [ [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],   # selifsh utility: identity matrix
            [0 0 0 0; 0 0 1 1; 0 1 0 1; 0 1 1 0],   # regular car to regular car
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0],   # regular car to ambulance
            [0 1 1 1; 0 0 0 0; 0 0 0 0; 0 0 0 0]    # ambulance to regular car
          ]

    (; name = "bee_queen", u = u, V = V, phi = phi)
end

# "undivided road": traffic flows in opposite directions and is typically separated by only lane markings, without any physical barrier or median.
function undivided_congestion(N_east, N_west)
    # N_east = 3; N_west = 3; 
    N = N_east + N_west;
    dim = [2 for i in 1:N]  # 2 actions for each player: 1: straight or 2: swerve

    # create costs u for each player
    delay_c(load) = 2*load
    crash_c = 10
    u = [zeros(dim...) for i = 1:N]
    for i in 1:N
        if i ≤ N_east # player i in east group
            for a in CartesianIndices(u[i])
                if a[i] == 1 ## I go straight
                    a_east = [a[i] for i in 1:N_east]
                    load_east = count(x->(x==1), a_east)
                    u[i][a] = delay_c(load_east)
                else ## I swerve
                    a_west = [a[i] for i in N_east+1:N]
                    any_swerve_in_west = any(x->(x==2), a_west) # check if any west player swerves
                    u[i][a] = any_swerve_in_west ? crash_c : 0
                end
            end
        else # player i in west group
            for a in CartesianIndices(u[i])
                if a[i] == 1 ## I go straight
                    a_west = [a[i] for i in N_east+1:N]
                    load_west = count(x->(x==1), a_west)
                    u[i][a] = delay_c(load_west)
                else ## I swerve
                    a_east = [a[i] for i in 1:N_east]
                    any_swerve_in_east = any(x->(x==2), a_east) # check if any west player swerves
                    u[i][a] = any_swerve_in_east ? crash_c : 0
                end
            end
        end
    end

    # social cost as the sum of utilities
    V = sum(u[i] for i in 1:N)

    # relationship structure imposed by reality
    phi = [zeros(N,N) + I(N),  # selfish
           [ones(N_east, N_east)-I(N_east) zeros(N_east, N_west); zeros(N_west, N_east) ones(N_west, N_west)-I(N_west)],  # within group relationship
           [zeros(N_east, N_east) ones(N_east, N_west); zeros(N_west, N_east) zeros(N_west, N_west)],  # group east to group west
           [zeros(N_east, N_east) zeros(N_east, N_west); ones(N_west, N_east) zeros(N_west, N_west)],  # group west to group east
        ]

    (; name="undivided", u = u, V = V, phi = phi)
end